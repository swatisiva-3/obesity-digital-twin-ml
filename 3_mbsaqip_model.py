#Purpose: Train, validate, and test feed-forward MLP model on preprocessed MBSAQIP data 
    #and investigate accuracy through error calculations and training graphs
#INPUT: File generated in 2_mbsaqip_preprocess_data that includes all of the training and testing data
#OUTPUT: Best performing model, saved as a .pth, along with testing error analysis and visuals

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory


TRAIN_FEATURES = "TRAIN_FEATURES.csv"
TRAIN_TARGETS = "TRAIN_TARGETS.csv"
VAL_FEATURES = "VALIDATION_FEATURES.csv"
VAL_TARGETS = "VALIDATION_TARGETS.csv"
TEST_FEATURES = "TEST_FEATURES.csv"
TEST_TARGETS = "TEST_TARGETS.csv"
MODEL_SAVE_PATH = "post_op_bmi_model.pth"

BATCH_SIZE = 128
LEARNING_RATE = 0.0027181131311625325 
EPOCHS = 200

#Load data----------------------------------------------------------------------

# Hide the tkinter window
root = Tk()
root.withdraw()

print("Please select the folder containing your preprocessed CSV files.")
DATA_FOLDER = askdirectory(title="Select Folder Containing CSV Files")

if not DATA_FOLDER:
    raise Exception("No folder was selected.")

print(f"Using folder:\n{DATA_FOLDER}")
TRAIN_FEATURES = os.path.join(DATA_FOLDER, "TRAIN_FEATURES.csv")
TRAIN_TARGETS = os.path.join(DATA_FOLDER, "TRAIN_TARGET.csv")
VAL_FEATURES = os.path.join(DATA_FOLDER, "VALIDATION_FEATURES.csv")
VAL_TARGETS = os.path.join(DATA_FOLDER, "VALIDATION_TARGET.csv")
TEST_FEATURES = os.path.join(DATA_FOLDER, "TEST_FEATURES.csv")
TEST_TARGETS = os.path.join(DATA_FOLDER, "TEST_TARGET.csv")
print("Loading datasets...")

X_train = pd.read_csv(TRAIN_FEATURES)
y_train = pd.read_csv(TRAIN_TARGETS)
X_val = pd.read_csv(VAL_FEATURES)
y_val = pd.read_csv(VAL_TARGETS)
X_test = pd.read_csv(TEST_FEATURES)
y_test = pd.read_csv(TEST_TARGETS)

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)
print("Testing samples:", X_test.shape)
print(X_train.shape)
print(y_train.shape)
print(X_train.head())
print(y_train.head())
#Convert to PyTorch tensors------------------------------------------------------

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

X_test = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
)

#Create data loaders----------------------------------------------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)


#define model-----------------------------------------------------------------

class BMI_Model(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 256), #256,64,128
            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(0.05179372697075629), #0.1
            

            nn.Linear(256,64),
            nn.ReLU(),

            nn.Dropout(0.05179372697075629), #0.1


            nn.Linear(64,128),
            nn.ReLU(),


            nn.Linear(128,1)

        )


    def forward(self,x):

        return self.network(x)



input_size = X_train.shape[1]


model = BMI_Model(input_size)


print(model)

#Training setup-------------------------------------------------------------------------

#loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss(beta=1.0)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=3.903509621103742e-05
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
    min_lr=1e-6
)


#early stopping initialization----------------------------------------------------------------
best_validation_loss = np.inf
EARLY_STOPPING_PATIENCE = 50
epochs_without_improvement = 0

training_history = []
validation_history = []

#train model------------------------------------------------------------------

print("\nBeginning training...")


for epoch in range(EPOCHS):
    model.train()
    training_loss = 0

    for X_batch,y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_function(
            predictions,
            y_batch
        )
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss /= len(train_loader)

#validation------------------------------------------------------------
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for X_batch,y_batch in val_loader:
            predictions = model(X_batch)
            loss = loss_function(
                predictions,
                y_batch
            )
            validation_loss += loss.item() * X_batch.size(0)
    validation_loss /= len(val_dataset)
    scheduler.step(validation_loss)
    training_history.append(training_loss)
    validation_history.append(validation_loss)

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        epochs_without_improvement = 0
        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

    else:
        epochs_without_improvement += 1

#Trigger early stopping if lack of improvement exceeds initialized stopping patience
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")
        break

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: "
            f"Train Loss={training_loss:.4f}, "
            f"Validation Loss={validation_loss:.4f}"
        )

print("\nTraining complete.")
print("Best model saved.")

#Testing-------------------------------------------

print("\nTesting model...")
model.load_state_dict(
    torch.load(MODEL_SAVE_PATH)
)
model.eval()
with torch.no_grad():
    predictions = model(X_test)
predictions = predictions.numpy()
actual = y_test.numpy()

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

mae = mean_absolute_error(
    actual,
    predictions
)

r2 = r2_score(
    actual,
    predictions
)

print("--------------------------------")
print("MODEL PERFORMANCE")
print("--------------------------------")

print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R2:   {r2:.3f}")



#Graph for training vs validation------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(training_history,label="Training")
plt.plot(validation_history,label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title(
    "Training Performance"
)
plt.show()

#actual vs predicted scatter plot------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(
    actual, 
    predictions, 
    alpha=0.3, 
    color='#1f77b4', 
    edgecolors='none', 
    s=25, 
    label='Patient Predictions'
)

min_val = min(actual.min(), predictions.min())
max_val = max(actual.max(), predictions.max())
plt.plot(
    [min_val, max_val], 
    [min_val, max_val], 
    color='red', 
    linestyle='--', 
    linewidth=2, 
    label='Ideal Fit (y = x)'
)

plt.xlabel('Actual Post-Op BMI', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Post-Op BMI', fontsize=12, fontweight='bold')
plt.title('Actual vs. Predicted Post-Op BMI', fontsize=14, fontweight='bold', pad=12)
plt.grid(True, linestyle=':', alpha=0.6)

metrics_text = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
plt.gca().text(
    0.05, 0.95, 
    metrics_text, 
    transform=plt.gca().transAxes, 
    fontsize=11, 
    verticalalignment='top', 
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
)

plt.legend(loc='lower right', frameon=True)
plt.tight_layout()