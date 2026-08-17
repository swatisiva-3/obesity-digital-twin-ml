###############################################################################
# 05_predict.py
#
# Predict postoperative BMI for one or more new patients
#
# Uses:
#   post_op_bmi_model.pth
#   feature_columns.pkl
#   scaler.pkl
#   continuous_imputer.pkl
#   binary_imputer.pkl
#
###############################################################################

import os
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

###############################################################################
# MODEL
###############################################################################

class BMI_Model(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features,128),
            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.2),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)

        )

    def forward(self,x):

        return self.network(x)

###############################################################################
# LOAD MODEL OBJECTS
###############################################################################

print("Select the folder containing your trained model.")

root = Tk()
root.withdraw()

MODEL_FOLDER = askdirectory(
    title="Select Model Folder"
)

if not MODEL_FOLDER:
    raise Exception("No model folder selected.")

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "post_op_bmi_model.pth"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_FOLDER,
    "feature_columns.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_FOLDER,
    "scaler.pkl"
)

CONTINUOUS_IMPUTER_PATH = os.path.join(
    MODEL_FOLDER,
    "continuous_imputer.pkl"
)

BINARY_IMPUTER_PATH = os.path.join(
    MODEL_FOLDER,
    "binary_imputer.pkl"
)

###############################################################################
# LOAD PREPROCESSING OBJECTS
###############################################################################

print("\nLoading preprocessing objects...")

with open(FEATURE_COLUMNS_PATH,"rb") as f:
    feature_columns = pickle.load(f)

with open(SCALER_PATH,"rb") as f:
    scaler = pickle.load(f)

with open(CONTINUOUS_IMPUTER_PATH,"rb") as f:
    continuous_imputer = pickle.load(f)

with open(BINARY_IMPUTER_PATH,"rb") as f:
    binary_imputer = pickle.load(f)

print("Loaded preprocessing objects.")

###############################################################################
# DEFINE FEATURE TYPES
###############################################################################

CONTINUOUS_COLUMNS = [

    "AGE",
    "BMI",
    "BMI_HIGH_BAR",
    "ALBUMIN",
    "CREATININE",
    "HEMO",
    "HCT"

]

BINARY_COLUMNS = [

    "HYPERLIPIDEMIA",
    "SLEEP_APNEA",
    "HIP",
    "DIABETES",
    "LIV_DIS",
    "LIV_DIS_SEVERE",
    "RENAL_INSUFFICIENCY",
    "DIALYSIS",
    "COPD"

]

###############################################################################
# LOAD PATIENT CSV
###############################################################################

print("\nSelect the patient CSV.")

PATIENT_FILE = askopenfilename(

    title="Select Patient CSV",

    filetypes=[
        ("CSV Files","*.csv")
    ]

)

if PATIENT_FILE == "":
    raise Exception("No patient file selected.")

patients = pd.read_csv(PATIENT_FILE)

print("\nPatients loaded:")
print(patients.shape)

###############################################################################
# ADD MISSING FEATURES
###############################################################################

for column in feature_columns:

    if column not in patients.columns:

        patients[column] = np.nan

###############################################################################
# KEEP ONLY FEATURES USED BY MODEL
###############################################################################

patients = patients[feature_columns]

###############################################################################
# ENSURE NUMERIC TYPES
###############################################################################

for col in CONTINUOUS_COLUMNS:

    if col in patients.columns:

        patients[col] = pd.to_numeric(
            patients[col],
            errors="coerce"
        )

for col in BINARY_COLUMNS:

    if col in patients.columns:

        patients[col] = pd.to_numeric(
            patients[col],
            errors="coerce"
        )

###############################################################################
# IMPUTE MISSING VALUES
###############################################################################

continuous_present = [

    c for c in CONTINUOUS_COLUMNS

    if c in patients.columns

]

binary_present = [

    c for c in BINARY_COLUMNS

    if c in patients.columns

]

if len(continuous_present) > 0:

    patients[continuous_present] = continuous_imputer.transform(

        patients[continuous_present]

    )

if len(binary_present) > 0:

    patients[binary_present] = binary_imputer.transform(

        patients[binary_present]

    )

###############################################################################
# SCALE CONTINUOUS VARIABLES
###############################################################################

if len(continuous_present) > 0:

    patients[continuous_present] = scaler.transform(

        patients[continuous_present]

    )

###############################################################################
# CONVERT TO TENSOR
###############################################################################

X = torch.tensor(

    patients.values,

    dtype=torch.float32

)

print("\nTensor created.")
print(X.shape)

###############################################################################
# LOAD TRAINED MODEL
###############################################################################

print("\nLoading trained model...")

input_size = len(feature_columns)

model = BMI_Model(input_size)

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

)

model.eval()

print("Model loaded successfully.")

###############################################################################
# MAKE PREDICTIONS
###############################################################################

print("\nPredicting postoperative BMI...")

with torch.no_grad():

    predictions = model(X)

predictions = predictions.numpy().flatten()

###############################################################################
# APPEND PREDICTIONS
###############################################################################

results = pd.read_csv(PATIENT_FILE)

results["PREDICTED_POST_OP_BMI"] = predictions

###############################################################################
# DISPLAY RESULTS
###############################################################################

print("\n========================================")
print("POSTOPERATIVE BMI PREDICTIONS")
print("========================================\n")

for i, prediction in enumerate(predictions):

    print(
        f"Patient {i + 1}: "
        f"{prediction:.2f} kg/m²"
    )

###############################################################################
# SAVE RESULTS
###############################################################################

OUTPUT_FILE = os.path.join(

    os.path.dirname(PATIENT_FILE),

    "PATIENTS_WITH_PREDICTIONS.csv"

)

results.to_csv(

    OUTPUT_FILE,

    index=False

)

###############################################################################
# OPTIONAL SUMMARY STATISTICS
###############################################################################

print("\n========================================")
print("SUMMARY")
print("========================================")

print(f"Patients Processed : {len(results)}")

print(
    f"Average Prediction : "
    f"{predictions.mean():.2f}"
)

print(
    f"Lowest Prediction  : "
    f"{predictions.min():.2f}"
)

print(
    f"Highest Prediction : "
    f"{predictions.max():.2f}"
)

print("\nPredictions saved to:")

print(OUTPUT_FILE)

print("\nPrediction complete.")