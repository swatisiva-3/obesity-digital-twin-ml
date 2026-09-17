"""
06_shap_explain.py
===================
STEP 6 of the pipeline.

Run standalone with:

    python 06_shap_explain.py

or:

    python 06_shap_explain.py --caseid 1703142

Requires:
    05_train_model.py

This version loads the PyTorch MLP models produced by Step 5.

The models use the original architecture:

    input -> 256 -> 64 -> 128 -> 1

SHAP is applied through a prediction wrapper that converts
pandas/NumPy feature matrices into PyTorch tensors.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import utils
import model_utils


GLOBAL_SHAP_SAMPLE_SIZE = 300
BACKGROUND_SAMPLE_SIZE = 60


# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------------------------

class BMI_Model(nn.Module):
    """
    Same architecture used by 05_train_model.py and the original
    3_mbsaqip_model.py:

        input -> 256 -> 64 -> 128 -> 1
    """

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 256),
            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(0.05179372697075629),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Dropout(0.05179372697075629),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------------------------------

def load_everything():

    df = utils.load_dataframe(
        "04_final_scored"
    )

    X, feature_to_variable, feature_to_domain = (
        model_utils.build_feature_matrix(df)
    )

    # Load the exact scaler fitted by Step 5.

    scaler = joblib.load(
        os.path.join(
            config.MODEL_DIR,
            "feature_scaler.joblib"
        )
    )

    feature_columns = utils.load_json(
        os.path.join(
            config.MODEL_DIR,
            "feature_columns.json"
        )
    )

    # Reproduce the exact feature ordering used during training.

    X = X[feature_columns]

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_columns,
        index=X.index
    )

    # ---------------------------------------------------------------
    # Load BMI model
    # ---------------------------------------------------------------

    model_bmi = BMI_Model(
        input_features=len(feature_columns)
    )

    bmi_path = os.path.join(
        config.MODEL_DIR,
        "mlp_bmi_30d.pth"
    )

    model_bmi.load_state_dict(
        torch.load(
            bmi_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model_bmi = model_bmi.to(DEVICE)
    model_bmi.eval()

    # ---------------------------------------------------------------
    # Load weight model
    # ---------------------------------------------------------------

    model_wgt = BMI_Model(
        input_features=len(feature_columns)
    )

    weight_path = os.path.join(
        config.MODEL_DIR,
        "mlp_weight_30d.pth"
    )

    model_wgt.load_state_dict(
        torch.load(
            weight_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model_wgt = model_wgt.to(DEVICE)
    model_wgt.eval()

    # ---------------------------------------------------------------
    # Train / validation split
    # ---------------------------------------------------------------

    split = utils.load_json(
        os.path.join(
            config.MODEL_DIR,
            "train_val_split.json"
        )
    )

    return (
        df,
        X_scaled,
        feature_to_variable,
        feature_to_domain,
        model_bmi,
        model_wgt,
        split
    )


# ---------------------------------------------------------------------------
# PYTORCH PREDICTION WRAPPER
# ---------------------------------------------------------------------------

def make_predict_function(model):
    """
    Create a function that SHAP can call.

    SHAP supplies a NumPy array.

    We convert:

        NumPy -> torch.FloatTensor -> model -> NumPy

    The returned array is one prediction per patient.
    """

    def predict(X):

        if isinstance(X, pd.DataFrame):
            values = X.values
        else:
            values = np.asarray(X)

        tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device=DEVICE
        )

        model.eval()

        with torch.no_grad():

            predictions = model(
                tensor
            )

        return predictions.detach().cpu().numpy().reshape(-1)

    return predict


# ---------------------------------------------------------------------------
# GROUP SHAP VALUES
# ---------------------------------------------------------------------------

def group_shap_by(shap_values_df, mapping):
    """
    Sum absolute SHAP values across feature columns that map
    to the same variable/domain.
    """

    grouped = (
        shap_values_df.T
        .groupby(mapping)
        .apply(lambda g: g.abs().sum())
    )

    return grouped


# ---------------------------------------------------------------------------
# BEESWARM
# ---------------------------------------------------------------------------

def make_beeswarm(
    shap_explanation,
    title,
    out_path
):

    plt.figure(
        figsize=(9, 7)
    )

    shap.plots.beeswarm(
        shap_explanation,
        show=False,
        max_display=20
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150
    )

    plt.close()

    utils.log(
        "06_shap",
        f"Saved {out_path}"
    )


# ---------------------------------------------------------------------------
# DOMAIN BAR
# ---------------------------------------------------------------------------

def make_domain_bar(
    domain_importance_series,
    title,
    out_path
):

    plt.figure(
        figsize=(7, 4.5)
    )

    domain_importance_series.sort_values().plot(
        kind="barh"
    )

    plt.xlabel(
        "Mean |SHAP value| (impact on predicted outcome)"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150
    )

    plt.close()

    utils.log(
        "06_shap",
        f"Saved {out_path}"
    )


# ---------------------------------------------------------------------------
# GLOBAL SHAP
# ---------------------------------------------------------------------------

def run_global_explanation(
    df,
    X_scaled,
    feature_to_variable,
    feature_to_domain,
    model_bmi,
    model_wgt,
    split
):

    utils.ensure_output_dirs()

    val_idx = [
        i
        for i in split["val_idx"]
        if i in X_scaled.index
    ]

    rng = np.random.RandomState(
        config.RANDOM_SEED
    )

    sample_idx = rng.choice(
        val_idx,
        size=min(
            GLOBAL_SHAP_SAMPLE_SIZE,
            len(val_idx)
        ),
        replace=False
    )

    background_idx = rng.choice(
        val_idx,
        size=min(
            BACKGROUND_SAMPLE_SIZE,
            len(val_idx)
        ),
        replace=False
    )

    background = X_scaled.loc[
        background_idx
    ]

    sample = X_scaled.loc[
        sample_idx
    ]

    importance_rows = []

    for target_name, model in (
        ("BMI", model_bmi),
        ("WEIGHT", model_wgt)
    ):

        utils.log(
            "06_shap",
            f"Computing global SHAP for {target_name} "
            f"model on {len(sample)} patients."
        )

        predict_function = make_predict_function(
            model
        )

        # SHAP's generic explainer can work with the prediction
        # wrapper. The background DataFrame establishes the reference
        # distribution.

        explainer = shap.Explainer(
            predict_function,
            background
        )

        sv = explainer(
            sample
        )

        sv_df = pd.DataFrame(
            sv.values,
            columns=sample.columns,
            index=sample.index
        )

        # -----------------------------------------------------------
        # VARIABLE-WISE SHAP
        # -----------------------------------------------------------

        make_beeswarm(
            sv,
            f"Variable-wise SHAP -- predicting 30-day {target_name}\n"
            f"({config.DATASET_LABEL})",
            os.path.join(
                config.OUTPUT_DIR,
                f"shap_global_variablewise_{target_name}.png"
            )
        )

        # -----------------------------------------------------------
        # DOMAIN-WISE SHAP
        # -----------------------------------------------------------

        domain_grouped = group_shap_by(
            sv_df,
            pd.Series(feature_to_domain)
        )

        domain_importance = (
            domain_grouped
            .mean(axis=1)
            .sort_values(ascending=False)
        )

        make_domain_bar(
            domain_importance,
            f"Domain-wise SHAP -- predicting 30-day {target_name}",
            os.path.join(
                config.OUTPUT_DIR,
                f"shap_global_domainwise_{target_name}.png"
            )
        )

        # -----------------------------------------------------------
        # VARIABLE-WISE IMPORTANCE TABLE
        # -----------------------------------------------------------

        var_grouped = group_shap_by(
            sv_df,
            pd.Series(feature_to_variable)
        )

        var_importance = (
            var_grouped
            .mean(axis=1)
        )

        for var_name, val in var_importance.items():

            domain = next(
                (
                    d
                    for f, d in feature_to_domain.items()
                    if feature_to_variable.get(f) == var_name
                ),
                None
            )

            importance_rows.append(
                {
                    "target": target_name,
                    "variable": var_name,
                    "domain": domain,
                    "mean_abs_shap": float(val)
                }
            )

    importance_df = pd.DataFrame(
        importance_rows
    )

    return importance_df


# ---------------------------------------------------------------------------
# CLINICAL WEIGHT VS SHAP
# ---------------------------------------------------------------------------

def build_weight_vs_shap_table(
    importance_df
):

    """
    Compare clinically assigned / Optuna-refined weights against
    model-derived SHAP importance.
    """

    if os.path.exists(
        config.LEARNED_WEIGHTS_PATH
    ):

        learned = utils.load_json(
            config.LEARNED_WEIGHTS_PATH
        )

    else:

        learned = utils.load_json(
            os.path.join(
                config.MODEL_DIR,
                "clinical_baseline_weights.json"
            )
        )

    variable_weights = learned[
        "variable_weights"
    ]

    rows = []

    for target in importance_df[
        "target"
    ].unique():

        sub = importance_df[
            importance_df["target"] == target
        ]

        for domain, group in sub.groupby(
            "domain"
        ):

            total_shap = group[
                "mean_abs_shap"
            ].sum()

            for _, r in group.iterrows():

                clinical_pct = (
                    variable_weights
                    .get(domain, {})
                    .get(
                        r["variable"],
                        np.nan
                    )
                )

                shap_pct = (
                    r["mean_abs_shap"]
                    / total_shap
                    * 100
                    if total_shap > 0
                    else np.nan
                )

                rows.append(
                    {
                        "target": target,
                        "domain": domain,
                        "variable": r["variable"],
                        "clinical_weight_pct_within_domain": clinical_pct,
                        "model_shap_importance_pct_within_domain":
                            round(shap_pct, 2)
                            if pd.notna(shap_pct)
                            else np.nan
                    }
                )

    table = pd.DataFrame(
        rows
    ).sort_values(
        [
            "target",
            "domain",
            "clinical_weight_pct_within_domain"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )

    out_path = os.path.join(
        config.OUTPUT_DIR,
        "clinical_weight_vs_shap_importance.xlsx"
    )

    table.to_excel(
        out_path,
        index=False,
        engine="openpyxl"
    )

    utils.log(
        "06_shap",
        f"Saved clinical-weight-vs-SHAP comparison -> {out_path}"
    )

    return table


# ---------------------------------------------------------------------------
# SINGLE PATIENT SHAP
# ---------------------------------------------------------------------------

def explain_single_patient(
    caseid
):

    """
    Generate SHAP waterfall explanations for one patient.
    """

    (
        df,
        X_scaled,
        feature_to_variable,
        feature_to_domain,
        model_bmi,
        model_wgt,
        split
    ) = load_everything()

    caseid_col = df[
        config.CASEID_COLUMN
    ].astype(str)

    matches = df.index[
        caseid_col == str(caseid)
    ]

    if len(matches) == 0:

        utils.log(
            "06_shap",
            f"CASEID {caseid!r} was not found in the "
            f"analytic cohort (outputs/04_final_scored.parquet). "
            f"Double-check the ID, or that this patient had valid "
            f"30-day BMI/weight follow-up."
        )

        return None

    row_idx = matches[0]

    background = X_scaled.sample(
        min(
            BACKGROUND_SAMPLE_SIZE,
            len(X_scaled)
        ),
        random_state=config.RANDOM_SEED
    )

    patient_row = X_scaled.loc[
        [row_idx]
    ]

    utils.ensure_output_dirs()

    breakdown_rows = []

    for target_name, model in (
        ("BMI", model_bmi),
        ("WEIGHT", model_wgt)
    ):

        predict_function = make_predict_function(
            model
        )

        explainer = shap.Explainer(
            predict_function,
            background
        )

        sv = explainer(
            patient_row
        )

        plt.figure(
            figsize=(9, 6)
        )

        shap.plots.waterfall(
            sv[0],
            show=False,
            max_display=15
        )

        plt.title(
            f"Patient {caseid} -- why the model predicts "
            f"this 30-day {target_name}"
        )

        plt.tight_layout()

        out_path = os.path.join(
            config.OUTPUT_DIR,
            f"shap_patient_{caseid}_{target_name}.png"
        )

        plt.savefig(
            out_path,
            dpi=150
        )

        plt.close()

        utils.log(
            "06_shap",
            f"Saved {out_path}"
        )

        predicted = float(
            predict_function(
                patient_row
            )[0]
        )

        for feat, val in zip(
            patient_row.columns,
            sv.values[0]
        ):

            breakdown_rows.append(
                {
                    "target": target_name,
                    "domain": feature_to_domain.get(feat),
                    "variable": feature_to_variable.get(feat),
                    "feature": feat,
                    "shap_contribution": round(
                        float(val),
                        4
                    ),
                    "predicted_value": round(
                        predicted,
                        2
                    )
                }
            )

    breakdown = pd.DataFrame(
        breakdown_rows
    ).sort_values(
        [
            "target",
            "shap_contribution"
        ],
        key=lambda s:
            s.abs()
            if s.name == "shap_contribution"
            else s,
        ascending=False
    )

    out_path = os.path.join(
        config.OUTPUT_DIR,
        f"shap_patient_{caseid}_breakdown.xlsx"
    )

    breakdown.to_excel(
        out_path,
        index=False,
        engine="openpyxl"
    )

    utils.log(
        "06_shap",
        f"Saved per-variable contribution table -> {out_path}"
    )

    print(
        breakdown.to_string(
            index=False
        )
    )

    return breakdown


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SHAP explainability for the "
            "phenotype-scoring model."
        )
    )

    parser.add_argument(
        "--caseid",
        type=str,
        default=None,
        help=(
            "CASEID of a single patient to generate "
            "a SHAP waterfall explanation for."
        )
    )

    parser.add_argument(
        "--skip-global",
        action="store_true",
        help=(
            "Skip the slower global variable-wise / "
            "domain-wise SHAP summary plots."
        )
    )

    args = parser.parse_args()

    if not args.skip_global:

        (
            df,
            X_scaled,
            feature_to_variable,
            feature_to_domain,
            model_bmi,
            model_wgt,
            split
        ) = load_everything()

        importance_df = run_global_explanation(
            df,
            X_scaled,
            feature_to_variable,
            feature_to_domain,
            model_bmi,
            model_wgt,
            split
        )

        build_weight_vs_shap_table(
            importance_df
        )

    target_caseid = (
        args.caseid
        or config.TARGET_CASEID
    )

    if target_caseid:

        explain_single_patient(
            target_caseid
        )

    else:

        utils.log(
            "06_shap",
            "No CASEID given -- skipping single-patient "
            "explanation. Run again with --caseid <CASEID>."
        )


if __name__ == "__main__":
    main()
