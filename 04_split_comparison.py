"""
04_split_comparison.py

Task #9:

Compare candidate train/validation/test split strategies.

This experiment does NOT modify the normal Step 4 pipeline.

Each split:

    - starts from the same clinical baseline weights
    - uses the same Optuna settings
    - uses the same random seed
    - uses a fixed validation set for Optuna
    - keeps the test set completely locked
    - evaluates the final learned weights once on the test set

Compared strategies:

    1. 70/15/15
    2. 80/10/10
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import optuna

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import config
import utils
import scoring_engine


optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------
# Weight helpers
# ------------------------------------------------------------

def flatten_weights(variable_weights, domain_weights):
    flat = {}

    for domain, vweights in variable_weights.items():
        for var, weight in vweights.items():
            flat[f"{domain}__{var}"] = weight

    for domain, weight in domain_weights.items():
        flat[f"DOMAIN__{domain}"] = weight

    return flat


def unflatten_weights(flat_params, variable_structure):
    variable_weights = {}

    for domain, vweights in variable_structure.items():
        raw = {
            var: flat_params[f"{domain}__{var}"]
            for var in vweights
        }

        variable_weights[domain] = (
            scoring_engine.renormalize_weights(raw)
        )

    domain_weights_raw = {
        domain: flat_params[f"DOMAIN__{domain}"]
        for domain in variable_structure
    }

    domain_weights = scoring_engine.renormalize_weights(
        domain_weights_raw
    )

    return variable_weights, domain_weights


# ------------------------------------------------------------
# Load clinical baseline
# ------------------------------------------------------------

def load_clinical_baseline():
    baseline_path = os.path.join(
        config.MODEL_DIR,
        "clinical_baseline_weights.json"
    )

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Clinical baseline weights not found: {baseline_path}"
        )

    baseline = utils.load_json(baseline_path)

    return (
        baseline["variable_weights"],
        baseline["domain_weights"],
    )


# ------------------------------------------------------------
# Create a fixed split
# ------------------------------------------------------------

def make_split(df, train_fraction, validation_fraction):
    """
    Create a deterministic train/validation/test split using exact
    integer sample counts.

    Examples:

        70/15/15
        80/10/10

    random_state is always config.RANDOM_SEED.
    """

    indices = np.asarray(df.index)
    n = len(indices)

    train_n = int(round(n * train_fraction))
    validation_n = int(round(n * validation_fraction))
    test_n = n - train_n - validation_n

    if train_n <= 0 or validation_n <= 0 or test_n <= 0:
        raise ValueError(
            "All split sizes must contain at least one sample."
        )

    rng = np.random.RandomState(config.RANDOM_SEED)
    shuffled_indices = rng.permutation(indices)

    train_idx = shuffled_indices[:train_n]

    val_idx = shuffled_indices[
        train_n:train_n + validation_n
    ]

    test_idx = shuffled_indices[
        train_n + validation_n:
    ]

    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int),
    )


# ------------------------------------------------------------
# Build Optuna objective
# ------------------------------------------------------------

def build_objective(
    df,
    variable_structure,
    train_idx,
    val_idx,
):
    y_train = df.loc[
        train_idx,
        "OUTCOME_BMI_30D"
    ]

    y_val = df.loc[
        val_idx,
        "OUTCOME_BMI_30D"
    ]

    def objective(trial):
        flat = {}

        for domain, vweights in variable_structure.items():

            for var in vweights:
                flat[f"{domain}__{var}"] = (
                    trial.suggest_float(
                        f"{domain}__{var}",
                        0.0,
                        100.0
                    )
                )

            flat[f"DOMAIN__{domain}"] = (
                trial.suggest_float(
                    f"DOMAIN__{domain}",
                    0.0,
                    100.0
                )
            )

        variable_weights, domain_weights = (
            unflatten_weights(
                flat,
                variable_structure
            )
        )

        domain_scores_df, total_score = (
            scoring_engine.full_scoring_pipeline(
                df,
                variable_weights,
                domain_weights,
            )
        )

        # Use the four domain scores as predictors.
        # TOTAL is deliberately not included because it is
        # derived from the same domain scores.

        features = domain_scores_df.fillna(0.0)

        X_train = features.loc[train_idx]
        X_val = features.loc[val_idx]

        model = Ridge(
            alpha=1.0,
            random_state=config.RANDOM_SEED
        )

        model.fit(
            X_train,
            y_train
        )

        # Apple Accelerate / NumPy can emit RuntimeWarnings
        # during large matrix multiplication even when the
        # resulting predictions are finite.
        #
        # Suppress only these RuntimeWarnings for prediction.
        # The explicit finite-value check below remains active.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                RuntimeWarning
            )

            preds = model.predict(X_val)

        if not np.isfinite(preds).all():
            raise ValueError(
                "Ridge validation predictions contain "
                "non-finite values."
            )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_val,
                    preds
                )
            )
        )

        return rmse

    return objective


# ------------------------------------------------------------
# Run one split experiment
# ------------------------------------------------------------

def run_experiment(
    df,
    name,
    train_fraction,
    validation_fraction,
    n_trials,
):
    print()
    print("=" * 70)
    print(f"RUNNING SPLIT: {name}")
    print("=" * 70)

    train_idx, val_idx, test_idx = make_split(
        df,
        train_fraction,
        validation_fraction,
    )

    print(
        f"Train:      {len(train_idx):,}"
    )

    print(
        f"Validation: {len(val_idx):,}"
    )

    print(
        f"Test:       {len(test_idx):,}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    assert len(
        set(train_idx) & set(val_idx)
    ) == 0

    assert len(
        set(train_idx) & set(test_idx)
    ) == 0

    assert len(
        set(val_idx) & set(test_idx)
    ) == 0

    assert (
        len(train_idx)
        + len(val_idx)
        + len(test_idx)
        == len(df)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Always start from the SAME clinical baseline.
    #
    # Do NOT use learned_weights.json here.
    # --------------------------------------------------------

    seed_variable_weights, seed_domain_weights = (
        load_clinical_baseline()
    )

    seed_flat = flatten_weights(
        seed_variable_weights,
        seed_domain_weights,
    )

    variable_structure = {
        domain: {
            variable: None
            for variable in scoring_engine.available_variables(
                domain
            )
        }
        for domain in config.VARIABLE_RATIONALE
    }

    sampler = optuna.samplers.TPESampler(
        seed=config.RANDOM_SEED
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    # Trial 0 = identical clinical baseline
    study.enqueue_trial(seed_flat)

    objective = build_objective(
        df,
        variable_structure,
        train_idx,
        val_idx,
    )

    print(
        f"Running {n_trials} Optuna trials..."
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    seed_rmse = study.trials[0].value
    best_rmse = study.best_value

    print(
        f"Seed validation RMSE: {seed_rmse:.4f}"
    )

    print(
        f"Best validation RMSE: {best_rmse:.4f}"
    )

    # --------------------------------------------------------
    # Get final learned weights
    # --------------------------------------------------------

    best_variable_weights, best_domain_weights = (
        unflatten_weights(
            study.best_params,
            variable_structure,
        )
    )

    # --------------------------------------------------------
    # LOCKED TEST EVALUATION
    # --------------------------------------------------------

    domain_scores_df, total_score = (
        scoring_engine.full_scoring_pipeline(
            df,
            best_variable_weights,
            best_domain_weights,
        )
    )

    features = domain_scores_df.fillna(0.0)

    X_train = features.loc[train_idx]
    X_test = features.loc[test_idx]

    y_train = df.loc[
        train_idx,
        "OUTCOME_BMI_30D"
    ]

    y_test = df.loc[
        test_idx,
        "OUTCOME_BMI_30D"
    ]

    final_model = Ridge(
        alpha=1.0,
        random_state=config.RANDOM_SEED
    )

    # Fit only on training data.
    final_model.fit(
        X_train,
        y_train
    )

    # Suppress the same non-fatal RuntimeWarnings during
    # the locked test prediction.
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            RuntimeWarning
        )

        test_predictions = final_model.predict(
            X_test
        )

    if not np.isfinite(test_predictions).all():
        raise ValueError(
            "Ridge test predictions contain "
            "non-finite values."
        )

    test_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                test_predictions
            )
        )
    )

    test_mae = float(
        mean_absolute_error(
            y_test,
            test_predictions
        )
    )

    test_r2 = float(
        r2_score(
            y_test,
            test_predictions
        )
    )

    print()
    print("LOCKED TEST RESULTS")

    print(
        f"Test RMSE: {test_rmse:.4f}"
    )

    print(
        f"Test MAE:  {test_mae:.4f}"
    )

    print(
        f"Test R²:   {test_r2:.4f}"
    )

    return {
        "split": name,
        "train_n": int(len(train_idx)),
        "validation_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "seed_validation_rmse": float(seed_rmse),
        "best_validation_rmse": float(best_rmse),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "n_trials": int(n_trials),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    df = utils.load_dataframe(
        "03_scored"
    )

    print(
        f"Loaded {len(df):,} rows from Step 3."
    )

    # Make sure the target exists.
    if "OUTCOME_BMI_30D" not in df.columns:
        raise ValueError(
            "OUTCOME_BMI_30D is missing."
        )

    # Keep only rows with a valid target.
    df = df[
        pd.to_numeric(
            df["OUTCOME_BMI_30D"],
            errors="coerce"
        ).notna()
    ].copy()

    print(
        f"Rows with valid BMI outcome: {len(df):,}"
    )

    results = []

    # --------------------------------------------------------
    # Experiment A: 70 / 15 / 15
    # --------------------------------------------------------

    results.append(
        run_experiment(
            df,
            "70/15/15",
            train_fraction=0.70,
            validation_fraction=0.15,
            n_trials=config.OPTUNA_N_TRIALS,
        )
    )

    # --------------------------------------------------------
    # Experiment B: 80 / 10 / 10
    # --------------------------------------------------------

    results.append(
        run_experiment(
            df,
            "80/10/10",
            train_fraction=0.80,
            validation_fraction=0.10,
            n_trials=config.OPTUNA_N_TRIALS,
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output_path = os.path.join(
        config.OUTPUT_DIR,
        "split_comparison_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 70)
    print("FINAL SPLIT COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved results -> {output_path}"
    )


if __name__ == "__main__":
    main()