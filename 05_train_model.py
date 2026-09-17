"""
05_train_model.py
==================
STEP 5 of the pipeline. Run standalone with:
    python 05_train_model.py

Requires:
    04_optuna_weight_learning.py

This version migrates the original PyTorch MLP into the new
digital-twin pipeline.

Pipeline connection:
    04_final_scored
        -> model_utils.build_feature_matrix()
        -> training-only StandardScaler
        -> PyTorch MLP
        -> 30-day BMI / 30-day weight

The original MLP architecture from 3_mbsaqip_model.py is preserved:

    input
      |
    256
      |
     64
      |
    128
      |
      1

with ReLU activations, BatchNorm, and Dropout.

Two independent models are trained:
    1. 30-day BMI
    2. 30-day weight (kg)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
import utils
import model_utils


# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

utils.log("05_train_model", f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# ORIGINAL MODEL ARCHITECTURE
# ---------------------------------------------------------------------------

class BMI_Model(nn.Module):
    """
    Original MLP architecture from 3_mbsaqip_model.py.

        input -> 256 -> 64 -> 128 -> 1

    The final layer outputs one continuous prediction.
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
# TRAINING SETTINGS
# ---------------------------------------------------------------------------

BATCH_SIZE = 512
LEARNING_RATE = 0.0027181131311625325
WEIGHT_DECAY = 3.903509621103742e-05
MAX_EPOCHS = 200

EARLY_STOPPING_PATIENCE = 50

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
MIN_LR = 1e-6

DROPOUT = 0.05179372697075629


# ---------------------------------------------------------------------------
# DATASET / DATALOADER
# ---------------------------------------------------------------------------

def build_dataloader(X, y, shuffle):

    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        np.asarray(y).reshape(-1, 1),
        dtype=torch.float32
    )

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
# TRAIN ONE MODEL
# ---------------------------------------------------------------------------

def train_one_target(
    X_train,
    X_val,
    y_train,
    y_val,
    target_name
):

    train_loader = build_dataloader(
        X_train,
        y_train,
        shuffle=True
    )

    validation_loader = build_dataloader(
        X_val,
        y_val,
        shuffle=False
    )

    model = BMI_Model(
        input_features=X_train.shape[1]
    ).to(DEVICE)

    loss_function = nn.SmoothL1Loss(
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

    best_validation_loss = np.inf
    best_state = None
    epochs_without_improvement = 0

    utils.log(
        "05_train_model",
        f"[{target_name}] Starting training with "
        f"{X_train.shape[1]} input features."
    )

    for epoch in range(MAX_EPOCHS):

        # ---------------------------------------------------------------
        # TRAINING
        # ---------------------------------------------------------------

        model.train()

        train_loss = 0.0

        for features, target in train_loader:

            features = features.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(features)

            loss = loss_function(
                prediction,
                target
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for features, target in validation_loader:

                features = features.to(DEVICE)
                target = target.to(DEVICE)

                prediction = model(features)

                loss = loss_function(
                    prediction,
                    target
                )

                validation_loss += loss.item()

        validation_loss /= len(validation_loader)

        scheduler.step(validation_loss)

        # ---------------------------------------------------------------
        # EARLY STOPPING
        # ---------------------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:

            current_lr = optimizer.param_groups[0]["lr"]

            utils.log(
                "05_train_model",
                f"[{target_name}] "
                f"epoch {epoch + 1}/{MAX_EPOCHS} "
                f"train_loss={train_loss:.5f} "
                f"val_loss={validation_loss:.5f} "
                f"lr={current_lr:.7f}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            utils.log(
                "05_train_model",
                f"[{target_name}] Early stopping at epoch {epoch + 1}."
            )

            break

    # Restore best validation model.

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------------------------------------------------
    # FINAL VALIDATION METRICS
    # -------------------------------------------------------------------

    model.eval()

    predictions = []

    with torch.no_grad():

        X_val_tensor = torch.tensor(
            X_val.values,
            dtype=torch.float32
        ).to(DEVICE)

        pred = model(
            X_val_tensor
        ).cpu().numpy().reshape(-1)

        predictions = pred

    y_true = np.asarray(y_val)

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                predictions
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            predictions
        )
    )

    r2 = float(
        r2_score(
            y_true,
            predictions
        )
    )

    utils.log(
        "05_train_model",
        f"[{target_name}] held-out "
        f"RMSE={rmse:.3f}  "
        f"MAE={mae:.3f}  "
        f"R^2={r2:.3f}"
    )

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_validation_loss": float(best_validation_loss),
        "epochs_trained": epoch + 1
    }

    return model, metrics


# ---------------------------------------------------------------------------
# SAVE PYTORCH MODEL
# ---------------------------------------------------------------------------

def save_model(model, path):

    torch.save(
        model.state_dict(),
        path
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    # -----------------------------------------------------------------------
    # LOAD NEW PIPELINE DATA
    # -----------------------------------------------------------------------

    df = utils.load_dataframe(
        "04_final_scored"
    )

    utils.log(
        "05_train_model",
        f"Loaded {len(df):,} rows from Step 4."
    )

    # -----------------------------------------------------------------------
    # BUILD NEW PIPELINE FEATURE MATRIX
    # -----------------------------------------------------------------------

    X, feature_to_variable, feature_to_domain = (
        model_utils.build_feature_matrix(df)
    )

    y_bmi = df[
        "OUTCOME_BMI_30D"
    ]

    y_wgt = df[
        "OUTCOME_WEIGHT_30D_KG"
    ]

    # -----------------------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # -----------------------------------------------------------------------

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    # -----------------------------------------------------------------------
    # SCALE USING TRAINING DATA ONLY
    # -----------------------------------------------------------------------

    scaler = StandardScaler().fit(
        X.loc[train_idx]
    )

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )

    # -----------------------------------------------------------------------
    # TRAIN BMI MODEL
    # -----------------------------------------------------------------------

    model_bmi, metrics_bmi = train_one_target(
        X_scaled.loc[train_idx],
        X_scaled.loc[val_idx],
        y_bmi.loc[train_idx],
        y_bmi.loc[val_idx],
        "BMI_30D"
    )

    # -----------------------------------------------------------------------
    # TRAIN WEIGHT MODEL
    # -----------------------------------------------------------------------

    model_wgt, metrics_wgt = train_one_target(
        X_scaled.loc[train_idx],
        X_scaled.loc[val_idx],
        y_wgt.loc[train_idx],
        y_wgt.loc[val_idx],
        "WEIGHT_30D_KG"
    )

    # -----------------------------------------------------------------------
    # OUTPUT DIRECTORIES
    # -----------------------------------------------------------------------

    utils.ensure_output_dirs()

    # -----------------------------------------------------------------------
    # SAVE MODELS
    # -----------------------------------------------------------------------

    bmi_model_path = os.path.join(
        config.MODEL_DIR,
        "mlp_bmi_30d.pth"
    )

    weight_model_path = os.path.join(
        config.MODEL_DIR,
        "mlp_weight_30d.pth"
    )

    save_model(
        model_bmi,
        bmi_model_path
    )

    save_model(
        model_wgt,
        weight_model_path
    )

    # -----------------------------------------------------------------------
    # SAVE SCALER
    # -----------------------------------------------------------------------

    joblib.dump(
        scaler,
        os.path.join(
            config.MODEL_DIR,
            "feature_scaler.joblib"
        )
    )

    # -----------------------------------------------------------------------
    # SAVE FEATURE METADATA
    # -----------------------------------------------------------------------

    utils.save_json(
        list(X.columns),
        os.path.join(
            config.MODEL_DIR,
            "feature_columns.json"
        )
    )

    utils.save_json(
        feature_to_variable,
        os.path.join(
            config.MODEL_DIR,
            "feature_to_variable.json"
        )
    )

    utils.save_json(
        feature_to_domain,
        os.path.join(
            config.MODEL_DIR,
            "feature_to_domain.json"
        )
    )

    # -----------------------------------------------------------------------
    # SAVE MODEL METRICS
    # -----------------------------------------------------------------------

    utils.save_json(
        {
            "BMI_30D": metrics_bmi,
            "WEIGHT_30D_KG": metrics_wgt,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "architecture": [
                256,
                64,
                128,
                1
            ],
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "device": str(DEVICE)
        },
        os.path.join(
            config.OUTPUT_DIR,
            "model_metrics.json"
        )
    )

    # -----------------------------------------------------------------------
    # SAVE TRAIN / VALIDATION INDICES
    # -----------------------------------------------------------------------

    utils.save_json(
        {
            "train_idx": [
                int(i)
                for i in train_idx
            ],
            "val_idx": [
                int(i)
                for i in val_idx
            ]
        },
        os.path.join(
            config.MODEL_DIR,
            "train_val_split.json"
        )
    )

    utils.log(
        "05_train_model",
        f"Saved BMI model -> {bmi_model_path}"
    )

    utils.log(
        "05_train_model",
        f"Saved weight model -> {weight_model_path}"
    )

    utils.log(
        "05_train_model",
        f"Saved scaler and feature metadata -> {config.MODEL_DIR}"
    )

    return model_bmi, model_wgt


if __name__ == "__main__":
    main()
