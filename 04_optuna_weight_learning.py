"""
04_optuna_weight_learning.py
=============================
STEP 4 of the pipeline. Run standalone with:  python 04_optuna_weight_learning.py
(Requires 03_scoring.py to have been run first.)

WHAT "SEED, NOT CONSTRAINT" MEANS HERE (this is the core of what you asked
for): Optuna's TPE sampler builds a probability model of "which weight
combinations produce a low validation loss" from the trials it has already
run. `study.enqueue_trial(...)` below hands it exactly ONE starting trial --
your clinical weights (from Step 3, after the flag-count nudge) -- which it
runs first, as trial #0. Every trial after that is chosen by TPE based on
what it has learned so far, and it is completely free to move away from that
starting point in any direction the data supports. The clinical weights are
a sample, not a boundary.

CONTINUAL LEARNING: if config.LEARNED_WEIGHTS_PATH already exists (i.e. this
script has been run before), THIS run seeds itself from those previously
learned weights instead of the raw clinical baseline -- "the new updated
weight becomes the initial weight" for the next round, exactly as you asked.
Delete that file if you ever want to reset to pure clinical weights.

WHAT OPTUNA IS OPTIMIZING: for each candidate weight set, it computes every
patient's domain scores and total phenotype score (via scoring_engine, the
SAME math Step 3 uses), fits a quick Ridge regression from those scores to
the real 30-day BMI outcome on a training split, and scores that regression
on a held-out validation split. The objective is to MINIMIZE validation
RMSE. So the weights that win are the ones that make the phenotype score
most predictive of what actually happened to real patients -- not the ones
that just look clinically tidy.

Output: models/learned_weights.json (used by 05_train_model.py and by the
next time you run this script), plus outputs/optuna_history.png.
"""

import os
import json
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import config
import utils
import scoring_engine

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep console output readable


def flatten_weights(variable_weights, domain_weights):
    """Turn the nested {domain: {var: weight}} + {domain: weight} structure
    into one flat dict of Optuna parameter_name -> value, for use as an
    enqueue_trial() seed."""
    flat = {}
    for domain, vweights in variable_weights.items():
        for var, w in vweights.items():
            flat[f"{domain}__{var}"] = w
    for domain, w in domain_weights.items():
        flat[f"DOMAIN__{domain}"] = w
    return flat


def unflatten_weights(flat_params, variable_structure):
    """Inverse of flatten_weights(), rebuilding the nested structure Optuna
    needs to hand to scoring_engine, and renormalizing each domain (and the
    domain-weight set) back to summing to 100."""
    variable_weights = {}
    for domain, vweights in variable_structure.items():
        raw = {var: flat_params[f"{domain}__{var}"] for var in vweights}
        variable_weights[domain] = scoring_engine.renormalize_weights(raw)
    domain_weights_raw = {domain: flat_params[f"DOMAIN__{domain}"] for domain in variable_structure}
    domain_weights = scoring_engine.renormalize_weights(domain_weights_raw)
    return variable_weights, domain_weights


def load_seed_weights(df):
    """Where this run's search starts from: previously learned weights if
    they exist, otherwise the clinical baseline + flag-count nudge from
    Step 3."""
    if os.path.exists(config.LEARNED_WEIGHTS_PATH):
        seed = utils.load_json(config.LEARNED_WEIGHTS_PATH)
        utils.log("04_optuna", f"Found previously learned weights at {config.LEARNED_WEIGHTS_PATH} "
                                f"-- using those as this run's starting point (continual learning).")
        return seed["variable_weights"], seed["domain_weights"]
    baseline_path = os.path.join(config.MODEL_DIR, "clinical_baseline_weights.json")
    baseline = utils.load_json(baseline_path)
    utils.log("04_optuna", "No previously learned weights found -- starting from the clinical "
                            "baseline (Step 3 output, i.e. your rationale weights after the flag-count nudge).")
    return baseline["variable_weights"], baseline["domain_weights"]


def build_objective(df, variable_structure, train_idx, val_idx):
    """
    Returns an Optuna objective function closed over the data and the fixed
    train/validation split, so every trial is judged on the SAME held-out
    patients (this is what makes "validation loss" comparable trial to trial).
    """
    y_train = df.loc[train_idx, "OUTCOME_BMI_30D"]
    y_val = df.loc[val_idx, "OUTCOME_BMI_30D"]

    def objective(trial):
        flat = {}
        for domain, vweights in variable_structure.items():
            for var in vweights:
                flat[f"{domain}__{var}"] = trial.suggest_float(f"{domain}__{var}", 0.0, 100.0)
            flat[f"DOMAIN__{domain}"] = trial.suggest_float(f"DOMAIN__{domain}", 0.0, 100.0)

        variable_weights, domain_weights = unflatten_weights(flat, variable_structure)
        domain_scores_df, total_score = scoring_engine.full_scoring_pipeline(df, variable_weights, domain_weights)

        #features = pd.concat([domain_scores_df, total_score.rename("TOTAL")], axis=1).fillna(0.0)
        features = domain_scores_df.fillna(0.0)
        X_train, X_val = features.loc[train_idx], features.loc[val_idx]

        model = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
        #model.fit(X_train, y_train)
        #preds = model.predict(X_val)
        #rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

        model.fit(X_train, y_train)

        # Avoid NumPy/Apple Accelerate large-matrix-multiplication warnings.
        # This is mathematically equivalent to X_val @ model.coef_.
        X_val_np = X_val.to_numpy(dtype=np.float64)
        coef = np.asarray(model.coef_, dtype=np.float64)

        preds = (
            X_val_np[:, 0] * coef[0]
            + X_val_np[:, 1] * coef[1]
            + X_val_np[:, 2] * coef[2]
            + X_val_np[:, 3] * coef[3]
            + model.intercept_
        )

        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

        return rmse

    return objective


def main():
    df = utils.load_dataframe("03_scored")
    utils.log("04_optuna", f"Loaded {len(df):,} rows from Step 3.")

    # variable_structure only needs to record WHICH variables exist per domain
    # (names), not their specs -- the actual weight values come from the seed
    # / Optuna trial suggestions below.
    variable_structure = {d: {v: None for v in scoring_engine.available_variables(d)} for d in config.VARIABLE_RATIONALE}

    seed_variable_weights, seed_domain_weights = load_seed_weights(df)
    seed_flat = flatten_weights(seed_variable_weights, seed_domain_weights)

    # --------------------------------------------------------------
    # Use the permanent 70/15/15 split shared by the entire pipeline.
    #
    # IMPORTANT:
    #   TRAIN      -> used to fit the Ridge model
    #   VALIDATION -> used by Optuna to select phenotype weights
    #   TEST       -> completely untouched during Step 4
    # --------------------------------------------------------------

    split_path = os.path.join(
        config.MODEL_DIR,
        "train_val_test_indices.json"
    )

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Permanent train/validation/test split not found: {split_path}"
        )

    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_idx = np.array(
        split_data["train_indices"],
        dtype=int
    )

    val_idx = np.array(
        split_data["validation_indices"],
        dtype=int
    )

    test_idx = np.array(
        split_data["test_indices"],
        dtype=int
    )

    # --------------------------------------------------------------
    # Safety checks: make sure the split matches this exact dataset.
    # --------------------------------------------------------------

    if (
        len(train_idx)
        + len(val_idx)
        + len(test_idx)
        != len(df)
    ):
        raise ValueError(
            "Saved train/validation/test split does not cover "
            "the current Step 3 dataset exactly."
        )

    if (
        len(set(train_idx) & set(val_idx)) > 0
        or len(set(train_idx) & set(test_idx)) > 0
        or len(set(val_idx) & set(test_idx)) > 0
    ):
        raise ValueError(
            "Saved train/validation/test split contains overlapping patients."
        )

    if (
        set(train_idx)
        | set(val_idx)
        | set(test_idx)
    ) != set(df.index):
        raise ValueError(
            "Saved train/validation/test split does not match "
            "the current dataframe indices."
        )

    utils.log(
        "04_optuna",
        f"Using permanent split: "
        f"{len(train_idx):,} train / "
        f"{len(val_idx):,} validation / "
        f"{len(test_idx):,} test."
    )

    utils.log(
        "04_optuna",
        "TEST SET IS LOCKED AND WILL NOT BE USED BY OPTUNA."
    )

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)  # TPE = Tree-structured Parzen Estimator
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # This is the "use your starting point only as an initial sample" step:
    # trial #0 is exactly the clinical/previously-learned weights. TPE then
    # builds its probability model from that result plus every trial after it.
    study.enqueue_trial(seed_flat)

    objective = build_objective(df, variable_structure, train_idx, val_idx)
    utils.log("04_optuna", f"Running {config.OPTUNA_N_TRIALS} trials (trial 0 = your clinical/previous weights)...")
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, show_progress_bar=False)

    utils.log("04_optuna", f"Best validation RMSE: {study.best_value:.4f} "
                            f"(trial #{study.best_trial.number})")
    seed_trial_value = study.trials[0].value
    utils.log("04_optuna", f"Clinical/seed weights' validation RMSE (trial #0): {seed_trial_value:.4f}")
    if study.best_trial.number == 0:
        utils.log("04_optuna", "Optuna did not find anything better than your clinical starting point -- "
                                "the clinical weights are being kept as-is.")

    best_variable_weights, best_domain_weights = unflatten_weights(study.best_params, variable_structure)

    utils.log("04_optuna", "Learned within-domain weights:")
    for domain, vweights in best_variable_weights.items():
        utils.log("04_optuna", f"  {domain}: " + ", ".join(f"{v}={w:.1f}%" for v, w in
                                                             sorted(vweights.items(), key=lambda x: -x[1])))
    utils.log("04_optuna", "Learned domain weights: " + ", ".join(
        f"{d}={w:.1f}%" for d, w in sorted(best_domain_weights.items(), key=lambda x: -x[1])
    ))

    utils.ensure_output_dirs()
    utils.save_json(
        {"variable_weights": best_variable_weights, "domain_weights": best_domain_weights,
         "best_validation_rmse": study.best_value, "n_trials": config.OPTUNA_N_TRIALS},
        config.LEARNED_WEIGHTS_PATH,
    )
    utils.log("04_optuna", f"Saved learned weights -> {config.LEARNED_WEIGHTS_PATH} "
                            f"(the NEXT run of this script will start from these).")

    # simple, dependency-light optimization-history plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.minimum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, "o", alpha=0.4, label="Trial validation RMSE")
    ax.plot(best_so_far, "-", linewidth=2, label="Best so far")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6, label="Trial 0 = clinical/seed weights")
    ax.set_xlabel("Optuna trial number")
    ax.set_ylabel("Validation RMSE (predicting 30-day BMI)")
    ax.set_title(f"Optuna weight search -- {config.DATASET_LABEL}")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, "optuna_history.png")
    fig.savefig(fig_path, dpi=150)
    utils.log("04_optuna", f"Saved optimization history plot -> {fig_path}")

    # recompute and save the FINAL scored dataset using the learned weights,
    # so 05/06/07 use the Optuna-refined scores rather than the raw clinical ones.
    domain_scores_df, total_score = scoring_engine.full_scoring_pipeline(df, best_variable_weights, best_domain_weights)
    final_df = df.copy()
    for c in domain_scores_df.columns:
        final_df[c] = domain_scores_df[c]
    final_df["TOTAL_PHENOTYPE_SCORE"] = total_score
    utils.save_dataframe(final_df, "04_final_scored")

    return best_variable_weights, best_domain_weights


if __name__ == "__main__":
    main()
