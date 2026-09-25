import os
import json

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from model_utils import build_feature_matrix


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[05_train] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BMI_Model(nn.Module):
    """
    Preserved from the previous MBSAQIP MLP architecture.

    input
      -> 256
      -> 64
      -> 128
      -> 1
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
# DataLoader
# ---------------------------------------------------------------------------

BATCH_SIZE = 512


def build_dataloader(X, y, shuffle):
    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.0027181131311625325
WEIGHT_DECAY = 3.903509621103742e-05

MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 50

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
MIN_LR = 1e-6


def train_one_target(
    X_train,
    y_train,
    X_val,
    y_val,
    target_name
):

    print()
    print(f"[{target_name}] {X_train.shape[1]} input features")

    train_loader = build_dataloader(
        X_train,
        y_train,
        shuffle=True
    )

    val_loader = build_dataloader(
        X_val,
        y_val,
        shuffle=False
    )

    model = BMI_Model(
        input_features=X_train.shape[1]
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(
        beta=1.0
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        # ---------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------

        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()
            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_losses.append(
                    loss.item()
                )

        val_loss = float(
            np.mean(val_losses)
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if (
            epoch == 1
            or epoch % 10 == 0
            or val_loss < best_val_loss
        ):
            print(
                f"epoch {epoch} "
                f"train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} "
                f"lr={current_lr:.7f}"
            )

        # ---------------------------------------------------------------
        # Best model / early stopping
        # ---------------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

                print(
                    f"[{target_name}] Early stopping "
                    f"epoch {epoch}"
                )

                break

    # Restore best validation model
    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------------------------------------------------
    # Final validation metrics
    # -------------------------------------------------------------------

    model.eval()

    predictions = []

    with torch.no_grad():

        X_val_tensor = torch.tensor(
            X_val,
            dtype=torch.float32
        ).to(DEVICE)

        pred = model(
            X_val_tensor
        ).detach().cpu().numpy().reshape(-1)

        predictions = pred

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_val,
                predictions
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_val,
            predictions
        )
    )

    r2 = float(
        r2_score(
            y_val,
            predictions
        )
    )

    print(
        f"[{target_name}] held-out RMSE={rmse:.3f} "
        f"MAE={mae:.3f} "
        f"R²={r2:.3f}"
    )

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "best_val_loss": float(best_val_loss),
    }

    return model, metrics


# ---------------------------------------------------------------------------
# Final test-set evaluation
#
# This function is deliberately separate from training. The test set is
# NEVER used for early stopping, scheduler decisions, or model selection.
# It is evaluated only after the best validation model has been selected.
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model,
    X_test,
    y_test,
    target_name
):

    model.eval()

    with torch.no_grad():

        X_test_tensor = torch.tensor(
            X_test,
            dtype=torch.float32
        ).to(DEVICE)

        predictions = (
            model(X_test_tensor)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_test,
            predictions
        )
    )

    r2 = float(
        r2_score(
            y_test,
            predictions
        )
    )

    print(
        f"[{target_name}] FINAL TEST "
        f"RMSE={rmse:.3f} "
        f"MAE={mae:.3f} "
        f"R²={r2:.3f}"
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "n_test": int(len(y_test)),
    }



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
        f"[05_train] Loading {input_path}"
    )

    df = pd.read_parquet(
        input_path
    )

    print(
        f"[05_train] Loaded {len(df):,} rows"
    )

    # -----------------------------------------------------------------------
    # Build raw feature matrix WITHOUT performing full-cohort imputation.
    #
    # We temporarily request a feature matrix whose severity features have
    # NaNs. The feature-construction function still defines the exact same
    # 45 features.
    #
    # To achieve this, first obtain the feature structure using the existing
    # function, then restore the original missing values from the source
    # columns where appropriate.
    # -----------------------------------------------------------------------

    X_initial, feature_to_variable, feature_to_domain, _ = (
        build_feature_matrix(df)
    )

    feature_columns = list(X_initial.columns)

    # -----------------------------------------------------------------------
    # Reconstruct the feature matrix with NaNs preserved for severity
    # features. This is necessary because the original build_feature_matrix()
    # historically filled them immediately.
    # -----------------------------------------------------------------------

    X = X_initial.copy()

    for col in feature_columns:

        if col.endswith("__SEVERITY"):

            variable_name = feature_to_variable[col]

            if variable_name in df.columns:
                # The processed dataframe's variable itself is generally not
                # the severity column, so use the generated severity column.
                source_col = f"{variable_name}__SEVERITY"

                if source_col in df.columns:
                    X[col] = df[source_col]

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

    # -----------------------------------------------------------------------
    # Keep only patients with valid outcomes.
    #
    # This filtering occurs BEFORE the train/validation split so that neither
    # target is trained on missing outcome values.
    # -----------------------------------------------------------------------

    valid_bmi = y_bmi.notna()
    valid_weight = y_weight.notna()

    print(
        f"[05_train] Valid BMI outcomes: "
        f"{valid_bmi.sum():,}"
    )

    print(
        f"[05_train] Valid weight outcomes: "
        f"{valid_weight.sum():,}"
    )

    # -----------------------------------------------------------------------
    # IMPORTANT:
    #
    # Use one common split for the entire cohort. Individual targets then
    # select the valid rows from that split.
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # IMPORTANT:
    #
    # Use the permanent 70/15/15 split shared by the entire pipeline.
    #
    # TRAIN      -> model fitting
    # VALIDATION -> early stopping / model selection
    # TEST       -> completely untouched until final evaluation
    # -----------------------------------------------------------------------

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
            "Saved train/validation/test split does not cover "
            "the current dataset exactly."
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

    print(
        f"[05_train] Using permanent split: "
        f"{len(train_idx):,} train / "
        f"{len(val_idx):,} validation / "
        f"{len(test_idx):,} test."
    )

    print(
        "[05_train] TEST SET IS LOCKED AND WILL NOT BE USED FOR TRAINING."
    )

    # -----------------------------------------------------------------------
    # Train-safe normalization for non-severity numeric ML features.
    #
    # Step 2 creates __NORM columns using the full cohort. That is useful for
    # exploratory analysis, but those statistics would leak information from
    # validation/test patients into model training.
    #
    # Recalculate the two non-severity numeric features used by the MLP
    # (height and operation year) using TRAIN rows only.
    # -----------------------------------------------------------------------

    normalization_stats = {}

    for variable_name in ["HGT", "OPYEAR"]:

        source_col = f"{variable_name}__STD"
        norm_col = f"{variable_name}__NORM"

        if source_col not in df.columns:
            raise ValueError(
                f"Expected standardized source column not found: {source_col}"
            )

        if norm_col not in X.columns:
            raise ValueError(
                f"Expected ML feature not found: {norm_col}"
            )

        train_values = pd.to_numeric(
            df.loc[train_idx, source_col],
            errors="coerce"
        )

        train_mean = train_values.mean()
        train_std = train_values.std()

        normalization_stats[variable_name] = {
            "mean": float(train_mean),
            "std": float(train_std),
        }

        if (
            pd.isna(train_std)
            or train_std == 0
        ):
            X[norm_col] = 0.0
        else:
            X[norm_col] = (
                pd.to_numeric(
                    df[source_col],
                    errors="coerce"
                ) - train_mean
            ) / train_std

        print(
            f"[05_train] {norm_col}: "
            f"train mean={train_mean:.6f}, "
            f"train std={train_std:.6f}"
        )

    # -----------------------------------------------------------------------
    # Train-safe imputation.
    #
    # Calculate each severity-feature median ONLY from the training rows.
    # Then apply those exact medians to both training and validation data.
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

    # Apply train-derived medians to the complete feature matrix.
    for col, median_value in imputation_medians.items():

        X[col] = X[col].fillna(
            median_value
        )

    # Any remaining missing values are handled as zero. In the current
    # pipeline these should primarily be absent categorical/norm features.
    X = X.fillna(0.0)

    print(
        f"[05_train] Features: {X.shape[1]}"
    )

    print(
        "[05_train] Train-safe imputation statistics calculated "
        "from training split only."
    )

    # -----------------------------------------------------------------------
    # Standardization
    #
    # Fit ONLY on training rows.
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
    # BMI target
    # -----------------------------------------------------------------------

    bmi_train_idx = [
        idx for idx in train_idx
        if valid_bmi.loc[idx]
    ]

    bmi_val_idx = [
        idx for idx in val_idx
        if valid_bmi.loc[idx]
    ]

    X_bmi_train = X_scaled.loc[
        bmi_train_idx
    ].to_numpy(
        dtype=np.float32
    )

    X_bmi_val = X_scaled.loc[
        bmi_val_idx
    ].to_numpy(
        dtype=np.float32
    )

    bmi_test_idx = [
        idx for idx in test_idx
        if valid_bmi.loc[idx]
    ]

    X_bmi_test = X_scaled.loc[
        bmi_test_idx
    ].to_numpy(
        dtype=np.float32
    )

    y_bmi_test = y_bmi.loc[
        bmi_test_idx
    ].to_numpy(
        dtype=np.float32
    )

    y_bmi_train = y_bmi.loc[
        bmi_train_idx
    ].to_numpy(
        dtype=np.float32
    )

    y_bmi_val = y_bmi.loc[
        bmi_val_idx
    ].to_numpy(
        dtype=np.float32
    )

    bmi_model, bmi_metrics = train_one_target(
        X_bmi_train,
        y_bmi_train,
        X_bmi_val,
        y_bmi_val,
        "BMI_30D"
    )

    bmi_test_metrics = evaluate_test_set(
        bmi_model,
        X_bmi_test,
        y_bmi_test,
        "BMI_30D"
    )

    # -----------------------------------------------------------------------
    # Weight target
    # -----------------------------------------------------------------------

    weight_train_idx = [
        idx for idx in train_idx
        if valid_weight.loc[idx]
    ]

    weight_val_idx = [
        idx for idx in val_idx
        if valid_weight.loc[idx]
    ]

    X_weight_train = X_scaled.loc[
        weight_train_idx
    ].to_numpy(
        dtype=np.float32
    )

    X_weight_val = X_scaled.loc[
        weight_val_idx
    ].to_numpy(
        dtype=np.float32
    )

    weight_test_idx = [
        idx for idx in test_idx
        if valid_weight.loc[idx]
    ]

    X_weight_test = X_scaled.loc[
        weight_test_idx
    ].to_numpy(
        dtype=np.float32
    )

    y_weight_test = y_weight.loc[
        weight_test_idx
    ].to_numpy(
        dtype=np.float32
    )

    y_weight_train = y_weight.loc[
        weight_train_idx
    ].to_numpy(
        dtype=np.float32
    )

    y_weight_val = y_weight.loc[
        weight_val_idx
    ].to_numpy(
        dtype=np.float32
    )

    weight_model, weight_metrics = train_one_target(
        X_weight_train,
        y_weight_train,
        X_weight_val,
        y_weight_val,
        "WEIGHT_30D_KG"
    )

    weight_test_metrics = evaluate_test_set(
        weight_model,
        X_weight_test,
        y_weight_test,
        "WEIGHT_30D_KG"
    )

    # -----------------------------------------------------------------------
    # Save models
    # -----------------------------------------------------------------------

    bmi_model_path = os.path.join(
        config.MODEL_DIR,
        "mlp_bmi_30d.pth"
    )

    weight_model_path = os.path.join(
        config.MODEL_DIR,
        "mlp_weight_30d.pth"
    )

    torch.save(
        bmi_model.state_dict(),
        bmi_model_path
    )

    torch.save(
        weight_model.state_dict(),
        weight_model_path
    )

    print(
        f"[05_train] Saved {bmi_model_path}"
    )

    print(
        f"[05_train] Saved {weight_model_path}"
    )

    # -----------------------------------------------------------------------
    # Save scaler
    # -----------------------------------------------------------------------

    scaler_path = os.path.join(
        config.MODEL_DIR,
        "feature_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        f"[05_train] Saved {scaler_path}"
    )

    # -----------------------------------------------------------------------
    # Save feature metadata
    # -----------------------------------------------------------------------

    feature_metadata = {
        "feature_columns": feature_columns,
        "feature_to_variable": feature_to_variable,
        "feature_to_domain": feature_to_domain,
        "imputation_medians": imputation_medians,
        "normalization_stats": normalization_stats,
    }

    metadata_path = os.path.join(
        config.MODEL_DIR,
        "feature_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            feature_metadata,
            f,
            indent=2
        )

    print(
        f"[05_train] Saved {metadata_path}"
    )

    # -----------------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------------

    metrics = {
        "BMI_30D": bmi_metrics,
        "WEIGHT_30D_KG": weight_metrics,

        "TEST_SET": {
            "BMI_30D": bmi_test_metrics,
            "WEIGHT_30D_KG": weight_test_metrics,
        },

        "n_total": int(len(df)),

        "n_valid_bmi": int(valid_bmi.sum()),
        "n_valid_weight": int(valid_weight.sum()),

        "n_test": int(len(test_idx)),

        "n_bmi_test": int(len(bmi_test_idx)),
        "n_weight_test": int(len(weight_test_idx)),

        "n_bmi_train": int(len(bmi_train_idx)),
        "n_bmi_validation": int(len(bmi_val_idx)),

        "n_weight_train": int(len(weight_train_idx)),
        "n_weight_validation": int(len(weight_val_idx)),

        "n_features": int(len(feature_columns)),

        "train_safe_imputation": True,
        "standard_scaler_fit_on_training_only": True,
    }

    metrics_path = os.path.join(
        config.MODEL_DIR,
        "training_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    print(
        f"[05_train] Saved {metrics_path}"
    )

    # -----------------------------------------------------------------------
    # Save split indices
    # -----------------------------------------------------------------------

    split_data = {
        "train_indices": [
            int(x) for x in train_idx
        ],
        "validation_indices": [
            int(x) for x in val_idx
        ],
        "test_indices": [
            int(x) for x in test_idx
        ],
        "bmi_train_indices": [
            int(x) for x in bmi_train_idx
        ],
        "bmi_validation_indices": [
            int(x) for x in bmi_val_idx
        ],
        "weight_train_indices": [
            int(x) for x in weight_train_idx
        ],
        "weight_validation_indices": [
            int(x) for x in weight_val_idx
        ],
    }

    split_path = os.path.join(
        config.MODEL_DIR,
        "train_val_indices.json"
    )

    with open(
        split_path,
        "w"
    ) as f:

        json.dump(
            split_data,
            f,
            indent=2
        )

    print(
        f"[05_train] Saved {split_path}"
    )

    print()
    print("[05_train] Training complete.")


if __name__ == "__main__":
    main()
