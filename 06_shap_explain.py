"""
06_shap_explain.py
===================
STEP 6 of the pipeline. Run standalone with:
    python 06_shap_explain.py                      (global explainability only)
    python 06_shap_explain.py --caseid 1703142      (also explains ONE patient)
(Requires 05_train_model.py to have been run first.)

>>> THREE WAYS TO PICK A PATIENT FOR THE SINGLE-CASE SHAP EXPLANATION <<<
  1. Command line (recommended):    python 06_shap_explain.py --caseid 1703142
  2. Edit config.TARGET_CASEID in config.py to that CASEID (as a string),
     then just run:                 python 06_shap_explain.py
  3. Import and call directly from another script / notebook:
         import importlib
         shap_explain = importlib.import_module("06_shap_explain")
         shap_explain.explain_single_patient("1703142")

What this produces:
  - outputs/shap_global_variablewise_BMI.png / _WEIGHT.png
      Beeswarm-style summary of which VARIABLES move the model's prediction
      most, across a sample of patients.
  - outputs/shap_global_domainwise_BMI.png / _WEIGHT.png
      The same thing grouped up to the 4 phenotype domains (SHAP values are
      additive, so a domain's contribution is just the sum of its
      variables' contributions -- this is a standard, valid aggregation).
  - outputs/clinical_weight_vs_shap_importance.xlsx
      Table comparing your clinically-assigned (Optuna-refined) weights
      against what the trained model actually leans on, per variable and
      per domain -- this is the "feature-attribution analysis compared
      clinically assigned domain weights with model-derived feature
      importance" step from your abstract.
  - outputs/shap_patient_<CASEID>_BMI.png / _WEIGHT.png (only if a CASEID
      is supplied) -- a waterfall plot showing exactly how THAT patient's
      own variable values pushed the prediction up or down from the average,
      plus a printed table with the same numbers.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import utils
import model_utils
import scoring_engine

GLOBAL_SHAP_SAMPLE_SIZE = 300   # how many patients the global summary plots are computed over
BACKGROUND_SAMPLE_SIZE = 60     # background reference set SHAP compares each prediction against


def load_everything():
    df = utils.load_dataframe("04_final_scored")
    X, feature_to_variable, feature_to_domain = model_utils.build_feature_matrix(df)
    scaler = joblib.load(os.path.join(config.MODEL_DIR, "feature_scaler.joblib"))
    feature_columns = utils.load_json(os.path.join(config.MODEL_DIR, "feature_columns.json"))
    X = X[feature_columns]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns, index=X.index)
    model_bmi = joblib.load(os.path.join(config.MODEL_DIR, "mlp_bmi_30d.joblib"))
    model_wgt = joblib.load(os.path.join(config.MODEL_DIR, "mlp_weight_30d.joblib"))
    split = utils.load_json(os.path.join(config.MODEL_DIR, "train_val_split.json"))
    return df, X_scaled, feature_to_variable, feature_to_domain, model_bmi, model_wgt, split


def group_shap_by(shap_values_df, mapping):
    """Sum |SHAP| across feature columns that map to the same variable/domain."""
    grouped = shap_values_df.T.groupby(mapping).apply(lambda g: g.abs().sum())
    return grouped  # index = variable/domain name, columns = patients


def make_beeswarm(shap_explanation, title, out_path):
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(shap_explanation, show=False, max_display=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    utils.log("06_shap", f"Saved {out_path}")


def make_domain_bar(domain_importance_series, title, out_path):
    plt.figure(figsize=(7, 4.5))
    domain_importance_series.sort_values().plot(kind="barh", color="#146661")
    plt.xlabel("Mean |SHAP value| (impact on predicted outcome)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    utils.log("06_shap", f"Saved {out_path}")


def run_global_explanation(df, X_scaled, feature_to_variable, feature_to_domain, model_bmi, model_wgt, split):
    utils.ensure_output_dirs()
    val_idx = [i for i in split["val_idx"] if i in X_scaled.index]
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_idx = rng.choice(val_idx, size=min(GLOBAL_SHAP_SAMPLE_SIZE, len(val_idx)), replace=False)
    background = X_scaled.loc[rng.choice(val_idx, size=min(BACKGROUND_SAMPLE_SIZE, len(val_idx)), replace=False)]
    sample = X_scaled.loc[sample_idx]

    importance_rows = []
    for target_name, model in (("BMI", model_bmi), ("WEIGHT", model_wgt)):
        utils.log("06_shap", f"Computing global SHAP for {target_name} model on {len(sample)} patients "
                              f"(this takes roughly {0.3*len(sample):.0f}s)...")
        explainer = shap.Explainer(model.predict, background)
        sv = explainer(sample)
        sv_df = pd.DataFrame(sv.values, columns=sample.columns, index=sample.index)

        make_beeswarm(sv, f"Variable-wise SHAP -- predicting 30-day {target_name}\n({config.DATASET_LABEL})",
                      os.path.join(config.OUTPUT_DIR, f"shap_global_variablewise_{target_name}.png"))

        domain_grouped = group_shap_by(sv_df, pd.Series(feature_to_domain))
        domain_importance = domain_grouped.mean(axis=1).sort_values(ascending=False)
        make_domain_bar(domain_importance, f"Domain-wise SHAP -- predicting 30-day {target_name}",
                         os.path.join(config.OUTPUT_DIR, f"shap_global_domainwise_{target_name}.png"))

        var_grouped = group_shap_by(sv_df, pd.Series(feature_to_variable))
        var_importance = var_grouped.mean(axis=1)
        for var_name, val in var_importance.items():
            importance_rows.append({"target": target_name, "variable": var_name,
                                     "domain": next((d for f, d in feature_to_domain.items()
                                                      if feature_to_variable.get(f) == var_name), None),
                                     "mean_abs_shap": val})

    importance_df = pd.DataFrame(importance_rows)
    return importance_df


def build_weight_vs_shap_table(importance_df):
    """
    Side-by-side comparison: the clinically-assigned / Optuna-refined weight
    for each variable (as a % of its domain) vs. how much the trained model
    actually relies on it (mean |SHAP|, also rescaled to a within-domain %
    so the two columns are on the same 0-100 footing and easy to compare).
    """
    learned = utils.load_json(config.LEARNED_WEIGHTS_PATH) if os.path.exists(config.LEARNED_WEIGHTS_PATH) \
        else utils.load_json(os.path.join(config.MODEL_DIR, "clinical_baseline_weights.json"))
    variable_weights = learned["variable_weights"]

    rows = []
    for target in importance_df["target"].unique():
        sub = importance_df[importance_df["target"] == target]
        for domain, group in sub.groupby("domain"):
            total_shap = group["mean_abs_shap"].sum()
            for _, r in group.iterrows():
                clinical_pct = variable_weights.get(domain, {}).get(r["variable"], np.nan)
                shap_pct = (r["mean_abs_shap"] / total_shap * 100) if total_shap > 0 else np.nan
                rows.append({
                    "target": target, "domain": domain, "variable": r["variable"],
                    "clinical_weight_pct_within_domain": clinical_pct,
                    "model_shap_importance_pct_within_domain": round(shap_pct, 2) if pd.notna(shap_pct) else np.nan,
                })
    table = pd.DataFrame(rows).sort_values(["target", "domain", "clinical_weight_pct_within_domain"],
                                            ascending=[True, True, False])
    out_path = os.path.join(config.OUTPUT_DIR, "clinical_weight_vs_shap_importance.xlsx")
    table.to_excel(out_path, index=False, engine="openpyxl")
    utils.log("06_shap", f"Saved clinical-weight-vs-SHAP comparison -> {out_path}")
    return table


def explain_single_patient(caseid):
    """
    >>> THIS IS THE FUNCTION THAT ANSWERS "give me the SHAP analysis for
    ONE patient I choose." <<< Pass any CASEID from your dataset (as a
    string or int -- it's matched either way).
    """
    df, X_scaled, feature_to_variable, feature_to_domain, model_bmi, model_wgt, split = load_everything()

    caseid_col = df[config.CASEID_COLUMN].astype(str)
    matches = df.index[caseid_col == str(caseid)]
    if len(matches) == 0:
        utils.log("06_shap", f"CASEID {caseid!r} was not found in the analytic cohort "
                              f"(outputs/04_final_scored.parquet). Double-check the ID, or that this "
                              f"patient had valid 30-day BMI/weight follow-up (required to be in this cohort).")
        return None
    row_idx = matches[0]

    background = X_scaled.sample(min(BACKGROUND_SAMPLE_SIZE, len(X_scaled)), random_state=config.RANDOM_SEED)
    patient_row = X_scaled.loc[[row_idx]]

    utils.ensure_output_dirs()
    breakdown_rows = []
    for target_name, model in (("BMI", model_bmi), ("WEIGHT", model_wgt)):
        explainer = shap.Explainer(model.predict, background)
        sv = explainer(patient_row)

        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(sv[0], show=False, max_display=15)
        plt.title(f"Patient {caseid} -- why the model predicts this 30-day {target_name}")
        plt.tight_layout()
        out_path = os.path.join(config.OUTPUT_DIR, f"shap_patient_{caseid}_{target_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        utils.log("06_shap", f"Saved {out_path}")

        predicted = model.predict(patient_row)[0]
        for feat, val in zip(patient_row.columns, sv.values[0]):
            breakdown_rows.append({
                "target": target_name, "domain": feature_to_domain.get(feat),
                "variable": feature_to_variable.get(feat), "feature": feat,
                "shap_contribution": round(float(val), 4), "predicted_value": round(float(predicted), 2),
            })

    breakdown = pd.DataFrame(breakdown_rows).sort_values(
        ["target", "shap_contribution"], key=lambda s: s.abs() if s.name == "shap_contribution" else s,
        ascending=False
    )
    out_path = os.path.join(config.OUTPUT_DIR, f"shap_patient_{caseid}_breakdown.xlsx")
    breakdown.to_excel(out_path, index=False, engine="openpyxl")
    utils.log("06_shap", f"Saved per-variable contribution table -> {out_path}")
    print(breakdown.to_string(index=False))
    return breakdown


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for the phenotype-scoring model.")
    parser.add_argument("--caseid", type=str, default=None,
                         help="CASEID of a single patient to generate a SHAP waterfall explanation for.")
    parser.add_argument("--skip-global", action="store_true",
                         help="Skip the (slower) global variable-wise / domain-wise SHAP summary plots.")
    args = parser.parse_args()

    if not args.skip_global:
        df, X_scaled, feature_to_variable, feature_to_domain, model_bmi, model_wgt, split = load_everything()
        importance_df = run_global_explanation(df, X_scaled, feature_to_variable, feature_to_domain,
                                                 model_bmi, model_wgt, split)
        build_weight_vs_shap_table(importance_df)

    target_caseid = args.caseid or config.TARGET_CASEID
    if target_caseid:
        explain_single_patient(target_caseid)
    else:
        utils.log("06_shap", "No CASEID given -- skipping single-patient explanation. "
                              "Run again with --caseid <CASEID>, e.g.: python 06_shap_explain.py --caseid 1703142")


if __name__ == "__main__":
    main()
