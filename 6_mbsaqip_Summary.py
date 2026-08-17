###############################################################################
# 06_explain_prediction.py
#
# Explain a patient's predicted postoperative BMI using Integrated Gradients.
#
# Requires:
#
# post_op_bmi_model.pth
# scaler.pkl
# continuous_imputer.pkl
# binary_imputer.pkl
# feature_columns.pkl
#
###############################################################################

import os
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename

###############################################################################
# MODEL
###############################################################################

class BMI_Model(nn.Module):

    def __init__(self,input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features,128),
            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.20),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)

        )

    def forward(self,x):

        return self.network(x)

###############################################################################
# SELECT MODEL DIRECTORY
###############################################################################

root = Tk()
root.withdraw()

MODEL_FOLDER = askdirectory(
    title="Select the folder containing your trained model"
)

###############################################################################
# PATHS
###############################################################################

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "post_op_bmi_model.pth"
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

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_FOLDER,
    "feature_columns.pkl"
)

###############################################################################
# LOAD PREPROCESSING OBJECTS
###############################################################################

print("Loading preprocessing objects...")

with open(FEATURE_COLUMNS_PATH,"rb") as f:
    feature_columns = pickle.load(f)

with open(SCALER_PATH,"rb") as f:
    scaler = pickle.load(f)

with open(CONTINUOUS_IMPUTER_PATH,"rb") as f:
    continuous_imputer = pickle.load(f)

with open(BINARY_IMPUTER_PATH,"rb") as f:
    binary_imputer = pickle.load(f)

###############################################################################
# LOAD MODEL
###############################################################################

model = BMI_Model(len(feature_columns))

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

)

model.eval()

###############################################################################
# CREATE INTEGRATED GRADIENTS OBJECT
###############################################################################

ig = IntegratedGradients(model)

###############################################################################
# FEATURE GROUPS
###############################################################################

FEATURE_GROUPS = {

    "Adiposity":[

        "BMI",
        "BMI_HIGH_BAR"

    ],

    "Metabolic":[

        "DIABETES",
        "HYPERLIPIDEMIA",
        "ALBUMIN",
        "CREATININE",
        "HEMO",
        "HCT",
        "LIV_DIS",
        "LIV_DIS_SEVERE",
        "RENAL_INSUFFICIENCY",
        "COPD"

    ],

    "Behavioral":[

        "SLEEP_APNEA"

    ]

}

###############################################################################
# CONTINUOUS VARIABLES
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

###############################################################################
# BINARY VARIABLES
###############################################################################

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

PATIENT_FILE = askopenfilename(

    title="Select Patient CSV",

    filetypes=[

        ("CSV Files","*.csv")

    ]

)

patients = pd.read_csv(PATIENT_FILE)

###############################################################################
# ADD ANY MISSING FEATURES
###############################################################################

for col in feature_columns:

    if col not in patients.columns:

        patients[col] = np.nan

###############################################################################
# KEEP ONLY TRAINING FEATURES
###############################################################################

patients = patients[feature_columns]

###############################################################################
# FIX DATA TYPES
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
# IMPUTE
###############################################################################

continuous_present = [

    c for c in CONTINUOUS_COLUMNS

    if c in patients.columns

]

binary_present = [

    c for c in BINARY_COLUMNS

    if c in patients.columns

]

patients[continuous_present] = continuous_imputer.transform(

    patients[continuous_present]

)

patients[binary_present] = binary_imputer.transform(

    patients[binary_present]

)

###############################################################################
# SCALE
###############################################################################

patients[continuous_present] = scaler.transform(

    patients[continuous_present]

)

###############################################################################
# CREATE TENSOR
###############################################################################

X = torch.tensor(

    patients.values,

    dtype=torch.float32

)

