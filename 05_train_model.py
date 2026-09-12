"""
05_train_model.py
==================
STEP 5 of the pipeline. Run standalone with:   python 05_train_model.py
(Requires 04_optuna_weight_learning.py to have been run first.)

What this does:
  1. Builds the model-ready feature matrix (model_utils.build_feature_matrix)
     from the Optuna-refined scored dataset.
  2. Splits into train/validation (config.TEST_SIZE), fits a StandardScaler
     on the TRAINING split only (so no validation-set information leaks
     into scaling -- a common, easy-to-miss source of over-optimistic
     validation numbers).
  3. Trains TWO feed-forward multilayer perceptrons (config.MLP_HIDDEN_LAYERS
     architecture) -- one predicting 30-day BMI, one predicting 30-day
     weight (kg) -- since SHAP explains one model output at a time cleanly,
     two focused models are easier to interpret than one two-headed model.
  4. Reports RMSE, MAE, and R^2 on the held-out validation split for both.
  5. Saves both models, the scaler, and the feature column order to
     models/ -- 06_shap_explain.py loads these exact files, so it explains
     the model that was actually evaluated here, not a freshly retrained one.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
import utils
import model_utils


def train_one_target(X_train, X_val, y_train, y_val, target_name):
    model = MLPRegressor(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
        max_iter=config.MLP_MAX_ITER,
        random_state=config.RANDOM_SEED,
        early_stopping=True,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    r2 = float(r2_score(y_val, preds))
    utils.log("05_train_model", f"[{target_name}] held-out RMSE={rmse:.3f}  MAE={mae:.3f}  R^2={r2:.3f}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = utils.load_dataframe("04_final_scored")
    utils.log("05_train_model", f"Loaded {len(df):,} rows from Step 4.")

    X, feature_to_variable, feature_to_domain = model_utils.build_feature_matrix(df)
    y_bmi = df["OUTCOME_BMI_30D"]
    y_wgt = df["OUTCOME_WEIGHT_30D_KG"]

    train_idx, val_idx = train_test_split(df.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    scaler = StandardScaler().fit(X.loc[train_idx])
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    model_bmi, metrics_bmi = train_one_target(
        X_scaled.loc[train_idx], X_scaled.loc[val_idx], y_bmi.loc[train_idx], y_bmi.loc[val_idx], "BMI_30D"
    )
    model_wgt, metrics_wgt = train_one_target(
        X_scaled.loc[train_idx], X_scaled.loc[val_idx], y_wgt.loc[train_idx], y_wgt.loc[val_idx], "WEIGHT_30D_KG"
    )

    utils.ensure_output_dirs()
    joblib.dump(model_bmi, os.path.join(config.MODEL_DIR, "mlp_bmi_30d.joblib"))
    joblib.dump(model_wgt, os.path.join(config.MODEL_DIR, "mlp_weight_30d.joblib"))
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, "feature_scaler.joblib"))
    utils.save_json(list(X.columns), os.path.join(config.MODEL_DIR, "feature_columns.json"))
    utils.save_json(feature_to_variable, os.path.join(config.MODEL_DIR, "feature_to_variable.json"))
    utils.save_json(feature_to_domain, os.path.join(config.MODEL_DIR, "feature_to_domain.json"))
    utils.save_json(
        {"BMI_30D": metrics_bmi, "WEIGHT_30D_KG": metrics_wgt, "n_train": len(train_idx), "n_val": len(val_idx)},
        os.path.join(config.OUTPUT_DIR, "model_metrics.json"),
    )
    utils.log("05_train_model", f"Saved models, scaler, and feature metadata -> {config.MODEL_DIR}")

    # save the val/train split indices so 06_shap_explain.py can (optionally)
    # restrict global SHAP summaries to the validation set only
    utils.save_json({"train_idx": [int(i) for i in train_idx], "val_idx": [int(i) for i in val_idx]},
                     os.path.join(config.MODEL_DIR, "train_val_split.json"))

    return model_bmi, model_wgt


if __name__ == "__main__":
    main()
