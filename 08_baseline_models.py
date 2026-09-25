"""
08_baseline_models.py

Baseline models for comparison with the Step 5 MLP.

Models:
    1. Training-set mean prediction
    2. Ridge regression

Both models use the EXACT same:
    - Step 4 final scored dataset
    - 45-feature model matrix
    - permanent 70/15/15 split
    - train-only severity-feature imputation
    - train-only StandardScaler

The test set is used only once, for final evaluation.

Ridge alpha is selected using the validation set only.
"""

import os
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import config
from model_utils import build_feature_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, predictions):
    """Calculate standard regression metrics."""

    return {
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions
                )
            )
        ),
        "MAE": float(
            mean_absolute_error(
                y_true,
                predictions
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                predictions
            )
        ),
        "n": int(len(y_true))
    }


def print_metrics(label, metrics):
    """Print regression metrics."""

    print(
        f"{label:<30} "
        f"RMSE={metrics['RMSE']:.3f}  "
        f"MAE={metrics['MAE']:.3f}  "
        f"R²={metrics['R2']:.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    os.makedirs(
        config.MODEL_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------------------------
    # Load Step 4 output
    # -----------------------------------------------------------------------

    input_path = os.path.join(
        config.OUTPUT_DIR,
        "04_final_scored.parquet"
    )

    print(
        f"[08_baseline] Loading {input_path}"
    )

    df = pd.read_parquet(
        input_path
    )

    print(
        f"[08_baseline] Loaded {len(df):,} rows"
    )

    # -----------------------------------------------------------------------
    # Build the exact feature matrix used by Step 5
    # -----------------------------------------------------------------------

    X_initial, feature_to_variable, feature_to_domain, _ = (
        build_feature_matrix(df)
    )

    feature_columns = list(
        X_initial.columns
    )

    X = X_initial.copy()

    # Restore NaNs for severity features so that imputation is calculated
    # exclusively from the training split.
    for col in feature_columns:

        if col.endswith("__SEVERITY"):

            variable_name = feature_to_variable[col]

            source_col = (
                f"{variable_name}__SEVERITY"
            )

            if source_col in df.columns:
                X[col] = df[source_col]

    print(
        f"[08_baseline] Features: {X.shape[1]}"
    )

    # -----------------------------------------------------------------------
    # Outcomes
    # -----------------------------------------------------------------------

    y_bmi = pd.to_numeric(
        df["OUTCOME_BMI_30D"],
        errors="coerce"
    )

    y_weight = pd.to_numeric(
        df["OUTCOME_WEIGHT_30D_KG"],
        errors="coerce"
    )

    valid_bmi = y_bmi.notna()
    valid_weight = y_weight.notna()

    # -----------------------------------------------------------------------
    # Load permanent 70/15/15 split
    # -----------------------------------------------------------------------

    split_path = os.path.join(
        config.MODEL_DIR,
        "train_val_test_indices.json"
    )

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Permanent split not found: {split_path}"
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

    # -----------------------------------------------------------------------
    # Safety checks
    # -----------------------------------------------------------------------

    if (
        len(train_idx)
        + len(val_idx)
        + len(test_idx)
        != len(df)
    ):
        raise ValueError(
            "Permanent split does not cover the dataframe exactly."
        )

    if (
        len(set(train_idx) & set(val_idx)) > 0
        or len(set(train_idx) & set(test_idx)) > 0
        or len(set(val_idx) & set(test_idx)) > 0
    ):
        raise ValueError(
            "Permanent split contains overlapping patients."
        )

    if (
        set(train_idx)
        | set(val_idx)
        | set(test_idx)
    ) != set(df.index):
        raise ValueError(
            "Permanent split does not match dataframe indices."
        )

    print(
        f"[08_baseline] Split: "
        f"{len(train_idx):,} train / "
        f"{len(val_idx):,} validation / "
        f"{len(test_idx):,} test"
    )

    print(
        "[08_baseline] TEST SET IS LOCKED."
    )

    # -----------------------------------------------------------------------
    # Train-only imputation
    # -----------------------------------------------------------------------

    imputation_medians = {}

    for col in feature_columns:

        if not col.endswith("__SEVERITY"):
            continue

        train_values = X.loc[
            train_idx,
            col
        ]

        median_value = train_values.median()

        if pd.isna(median_value):
            median_value = 0.0

        imputation_medians[col] = float(
            median_value
        )

    for col, median_value in imputation_medians.items():

        X[col] = X[col].fillna(
            median_value
        )

    X = X.fillna(0.0)

    # -----------------------------------------------------------------------
    # Train-only scaling
    # -----------------------------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X.loc[train_idx]
    )

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        index=X.index,
        columns=X.columns
    )

    # -----------------------------------------------------------------------
    # Evaluation function
    # -----------------------------------------------------------------------

    def run_target(
        target_name,
        y,
        valid_mask
    ):

        target_train_idx = [
            idx for idx in train_idx
            if valid_mask.loc[idx]
        ]

        target_val_idx = [
            idx for idx in val_idx
            if valid_mask.loc[idx]
        ]

        target_test_idx = [
            idx for idx in test_idx
            if valid_mask.loc[idx]
        ]

        X_train = X_scaled.loc[
            target_train_idx
        ].to_numpy(
            dtype=np.float64
        )

        X_val = X_scaled.loc[
            target_val_idx
        ].to_numpy(
            dtype=np.float64
        )

        X_test = X_scaled.loc[
            target_test_idx
        ].to_numpy(
            dtype=np.float64
        )

        y_train = y.loc[
            target_train_idx
        ].to_numpy(
            dtype=np.float64
        )

        y_val = y.loc[
            target_val_idx
        ].to_numpy(
            dtype=np.float64
        )

        y_test = y.loc[
            target_test_idx
        ].to_numpy(
            dtype=np.float64
        )

        print()
        print("=" * 70)
        print(target_name)
        print("=" * 70)

        print(
            f"Train:      {len(y_train):,}"
        )

        print(
            f"Validation: {len(y_val):,}"
        )

        print(
            f"Test:       {len(y_test):,}"
        )

        # -------------------------------------------------------------------
        # Baseline 1: training-set mean
        # -------------------------------------------------------------------

        train_mean = float(
            np.mean(y_train)
        )

        val_mean_predictions = np.full(
            len(y_val),
            train_mean
        )

        test_mean_predictions = np.full(
            len(y_test),
            train_mean
        )

        mean_val_metrics = calculate_metrics(
            y_val,
            val_mean_predictions
        )

        mean_test_metrics = calculate_metrics(
            y_test,
            test_mean_predictions
        )

        print()
        print("MEAN BASELINE")

        print_metrics(
            "Validation",
            mean_val_metrics
        )

        print_metrics(
            "FINAL TEST",
            mean_test_metrics
        )

        # -------------------------------------------------------------------
        # Baseline 2: Ridge regression
        #
        # Alpha is selected using validation only.
        # The test set is NOT used during model selection.
        # -------------------------------------------------------------------

        alphas = [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0
        ]

        best_alpha = None
        best_val_rmse = float("inf")
        best_ridge = None

        print()
        print("RIDGE VALIDATION SEARCH")

        for alpha in alphas:

            ridge = Ridge(
                alpha=alpha
            )

            ridge.fit(
                X_train,
                y_train
            )

            val_predictions = ridge.predict(
                X_val
            )

            val_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_val,
                        val_predictions
                    )
                )
            )

            print(
                f"alpha={alpha:<7} "
                f"validation RMSE={val_rmse:.4f}"
            )

            if val_rmse < best_val_rmse:

                best_val_rmse = val_rmse
                best_alpha = alpha
                best_ridge = ridge

        # -------------------------------------------------------------------
        # Evaluate selected Ridge model
        # -------------------------------------------------------------------

        ridge_val_predictions = best_ridge.predict(
            X_val
        )

        ridge_test_predictions = best_ridge.predict(
            X_test
        )

        ridge_val_metrics = calculate_metrics(
            y_val,
            ridge_val_predictions
        )

        ridge_test_metrics = calculate_metrics(
            y_test,
            ridge_test_predictions
        )

        print()
        print(
            f"Selected Ridge alpha: {best_alpha}"
        )

        print_metrics(
            "Ridge Validation",
            ridge_val_metrics
        )

        print_metrics(
            "Ridge FINAL TEST",
            ridge_test_metrics
        )

        return {
            "mean_baseline": {
                "train_mean": train_mean,
                "validation": mean_val_metrics,
                "test": mean_test_metrics
            },
            "ridge": {
                "selected_alpha": best_alpha,
                "validation": ridge_val_metrics,
                "test": ridge_test_metrics
            }
        }

    # -----------------------------------------------------------------------
    # Run BMI
    # -----------------------------------------------------------------------

    bmi_results = run_target(
        "BMI_30D",
        y_bmi,
        valid_bmi
    )

    # -----------------------------------------------------------------------
    # Run weight
    # -----------------------------------------------------------------------

    weight_results = run_target(
        "WEIGHT_30D_KG",
        y_weight,
        valid_weight
    )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    results = {
        "feature_count": int(X.shape[1]),
        "n_total": int(len(df)),
        "split": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx))
        },
        "BMI_30D": bmi_results,
        "WEIGHT_30D_KG": weight_results
    }

    output_path = os.path.join(
        config.MODEL_DIR,
        "baseline_metrics.json"
    )

    with open(
        output_path,
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print(
        f"[08_baseline] Saved results to {output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
