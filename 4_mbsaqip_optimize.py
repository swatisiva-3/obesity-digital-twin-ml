# Hyperparameter optimization for the Post-Op BMI Prediction Model

import os
import json
import time
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tkinter import Tk
from tkinter.filedialog import askdirectory

#settings----------------------------------------------------

MODEL_SAVE_PATH = "best_model.pth"

N_TRIALS = 50
#75
MAX_EPOCHS = 40
#200
EARLY_STOPPING_PATIENCE = 8
#20
DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

#Select data folder-------------------------------------------------------------------

root = Tk()
root.withdraw()

print("Select folder containing TRAIN / VALIDATION csv files")

DATA_FOLDER = askdirectory(
    title="Select Processed Data Folder"
)

if DATA_FOLDER == "":
    raise Exception("No folder selected.")

#load data--------------------------------------------------------------------

print("\nLoading datasets...")

X_train = pd.read_csv(
    os.path.join(
        DATA_FOLDER,
        "TRAIN_FEATURES.csv"
    )
)

y_train = pd.read_csv(
    os.path.join(
        DATA_FOLDER,
        "TRAIN_TARGET.csv"
    )
)

X_val = pd.read_csv(
    os.path.join(
        DATA_FOLDER,
        "VALIDATION_FEATURES.csv"
    )
)

y_val = pd.read_csv(
    os.path.join(
        DATA_FOLDER,
        "VALIDATION_TARGET.csv"
    )
)

#convert to tensors-------------------------------------------------------------------------

X_train = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val.values,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val.values,
    dtype=torch.float32
)

#model---------------------------------------------------------------------------

class BMI_Model(nn.Module):

    def __init__(
        self,
        input_size,
        hidden1,
        hidden2,
        hidden3,
        dropout
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                hidden1
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden1,
                hidden2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden2,
                hidden3
            ),

            nn.ReLU(),

            nn.Linear(
                hidden3,
                1
            )

        )

    def forward(
        self,
        x
    ):

        return self.network(x)

#create dataloader-----------------------------------------------------------------

def build_dataloaders(batch_size):

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True

    )

    validation_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False

    )

    return train_loader, validation_loader

#results storage----------------------------------------------------------

history = []

best_model = None

best_score = np.inf

best_parameters = None

#optuna objective function-----------------------------------------------------------

def objective(trial):

    global history
    global best_model
    global best_score
    global best_parameters

    #hyperparameters to optimize----------------------------------------------

    hidden1 = trial.suggest_categorical(
        "hidden1",
        [64, 128, 256, 512]
    )

    hidden2 = trial.suggest_categorical(
        "hidden2",
        [32, 64, 128, 256]
    )

    hidden3 = trial.suggest_categorical(
        "hidden3",
        [16, 32, 64, 128]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-2,
        log=True
    )

    dropout = trial.suggest_float(
        "dropout",
        0.0,
        0.5
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128]
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-2,
        log=True
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "AdamW"]
    )

    #build model---------------------------------------------------------------

    model = BMI_Model(

        input_size=X_train.shape[1],

        hidden1=hidden1,

        hidden2=hidden2,

        hidden3=hidden3,

        dropout=dropout

    ).to(DEVICE)

   #loss function------------------------------------------------------------------------

    loss_function = nn.SmoothL1Loss()

    #optimizer------------------------------------------

    if optimizer_name == "Adam":

        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay

        )

    else:

        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay

        )

    #Scheduler-------------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=8,

        min_lr=1e-6

    )

    #dataloaders---------------------------------------------------------

    train_loader, validation_loader = build_dataloaders(
        batch_size
    )

    #early stopping------------------------------------------------

    best_validation_loss = np.inf

    epochs_without_improvement = 0

    #training loop------------------------------------------------

    for epoch in range(MAX_EPOCHS):

        #Training------------------------------------------------------

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

        #Validation--------------------------------------------------------------

        model.eval()

        validation_loss = 0.0

        predictions = []

        actuals = []

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

                predictions.extend(
                    prediction.cpu().numpy().flatten()
                )

                actuals.extend(
                    target.cpu().numpy().flatten()
                )

        validation_loss /= len(validation_loader)

        scheduler.step(validation_loss)

        #Early stopping------------------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            break

   #Metrics--------------------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            actuals,
            predictions
        )
    )

    mae = mean_absolute_error(
        actuals,
        predictions
    )

    r2 = r2_score(
        actuals,
        predictions
    )

    #Save Trial----------------------------------------------------------

    history.append({

        "trial": trial.number,

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2,

        "epochs": epoch + 1,

        "learning_rate": learning_rate,

        "dropout": dropout,

        "batch_size": batch_size,

        "weight_decay": weight_decay,

        "hidden1": hidden1,

        "hidden2": hidden2,

        "hidden3": hidden3,

        "optimizer": optimizer_name

    })

    #Save best model------------------------------------------------------------

    if rmse < best_score:

        best_score = rmse

        best_model = model.state_dict()

        best_parameters = trial.params.copy()

        os.makedirs(
            os.path.join(DATA_FOLDER, "Optimization"),
            exist_ok=True
        )

        torch.save(
            best_model,
            os.path.join(
                DATA_FOLDER,
                "Optimization",
                "best_model_so_far.pth"
            )
        )

    with open(
        os.path.join(
            DATA_FOLDER,
            "Optimization",
            "best_parameters_so_far.json"
        ),
        "w"
    ) as f:

        json.dump(
            best_parameters,
            f,
            indent=4
        )

    #minimizes rmse
    return rmse

#run optuna-------------------------------------------------------------------

print("\n" + "=" * 70)
print("Starting Hyperparameter Optimization")
print("=" * 70)

start_time = time.time()

study = optuna.create_study(
    direction="minimize",
    study_name="Post_Op_BMI_Optimization"
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)

elapsed = time.time() - start_time

#create output folder titled "optimization"

OUTPUT_FOLDER = os.path.join(
    DATA_FOLDER,
    "Optimization"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

#save the best model---------------------------------------------------------

torch.save(

    best_model,

    os.path.join(
        OUTPUT_FOLDER,
        "best_model.pth"
    )

)

#save best parameters--------------------------------------------------------------

with open(

    os.path.join(
        OUTPUT_FOLDER,
        "best_hyperparameters.json"
    ),

    "w"

) as f:

    json.dump(

        best_parameters,

        f,

        indent=4

    )

#save history of all trials------------------------------------------------

history_df = pd.DataFrame(history)

history_df.sort_values(

    "RMSE",

    inplace=True

)

history_df.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "optimization_results.csv"

    ),

    index=False

)

#save complete optuna history-------------------------------------------------------------

study.trials_dataframe().to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "optuna_history.csv"

    ),

    index=False

)

#print best results---------------------------------------------------------

print("\n")
print("=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)

print(f"\nTrials Completed : {len(study.trials)}")

print(f"Runtime          : {elapsed/60:.2f} minutes")

print(f"Best RMSE        : {study.best_value:.4f}")

print("\nBest Parameters")

for key, value in study.best_params.items():

    print(f"{key:20} : {value}")

#save summary-------------------------------------------------------

summary = {

    "Trials": len(study.trials),

    "Runtime Minutes": elapsed / 60,

    "Best RMSE": study.best_value,

    "Best Parameters": study.best_params

}

with open(

    os.path.join(

        OUTPUT_FOLDER,

        "summary.json"

    ),

    "w"

) as f:

    json.dump(

        summary,

        f,

        indent=4

    )

#feature importance-------------------------------------------------------

importance = optuna.importance.get_param_importances(
    study
)

importance_df = pd.DataFrame({

    "Parameter": importance.keys(),

    "Importance": importance.values()

})

importance_df.sort_values(

    "Importance",

    ascending=False,

    inplace=True

)

importance_df.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "parameter_importance.csv"

    ),

    index=False

)

#optiuonal plots that i found helpful

try:

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))

    plt.plot(history_df["RMSE"])

    plt.xlabel("Ranked Trial")

    plt.ylabel("Validation RMSE")

    plt.title("Optimization Results")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_FOLDER,

            "optimization_curve.png"

        )

    )

    plt.close()

except Exception:

    print("Could not generate optimization plot.")

#FINISHED

print("\nOptimization files saved to:\n")

print(OUTPUT_FOLDER)

print("\nDone.")