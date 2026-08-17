#Purpose: Consolidate MBSAQIP data by loading the data into a master csv file
#INPUT: MBSAQIP files, in TXT file format.  Simply select the folder containing all the TXT files
#OUTPUT:  Unified data set containing only the variables of interest for every patient included in the files

import pandas as pd
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path

# SETTINGS
root = Tk()
root.withdraw()

# Open File Explorer to choose approproate files
folder = askdirectory(title="Select the folder containing your TXT files")
output = askdirectory(title="Select the output folder that you would like containing the master dataset")
TXT_FOLDER = Path(folder)
CSV_FOLDER = Path(output)
print("Selected input folder:", TXT_FOLDER)
print("Selected output folder:", CSV_FOLDER)
CASE_ID = "CASEID"

# Variables I want to keep
KEEP_COLUMNS = {
    "CASEID",
    "AGE",
    "SEX", 
    "PROCEDURE_TYPE", #initial, conversion, revision
    "SMOKER", 
    "GERD", 
    "HISTORY_DVT", 
    "ASACLASS", #"ASA I - Normal/Healthy", "ASA II - Mild systemic disease", "ASA III - Severe systemic disease", "ASA IV - Severe systemic disease threat to life"
    "OPLENGTH",
    "IMMUNOSUPR_THER", 
    "PREVIOUS_SURGERY",
    "NBHTN_MEDS", 
    "CHRONIC_STEROIDS", 
    "HTN_MEDS", #1,2,3+
    "CPTUNLISTED_REVCONV",#binary, if patient is undergoing revision or conversion surgery
    "BMI",
    "BMI_HIGH_BAR",
    "HYPERLIPIDEMIA",
    "SLEEP_APNEA",
    "HIP",
    "DIABETES",
    "LIV_DIS",
    "LIV_DIS_SEVERE",
    "RENAL_INSUFFICIENCY",
    "DIALYSIS",
    "HEMO",
    "COPD",
    "ALBUMIN",
    "CREATININE",
    "HCT",
    "BMI_CLOSEST30D",
    "BMI_DISCH"
}

DELIMITER = "\t"

#Make master dataset

master = None

for file in TXT_FOLDER.glob("*.txt"):


    #read only columns i care about and skip ones without a caseid

    cols = pd.read_csv(
        file,
        sep=DELIMITER,
        nrows=0
    ).columns

    available = [c for c in cols if c in KEEP_COLUMNS]

    if CASE_ID not in available:
        print("  Skipped (no CASEID)")
        continue

    df = pd.read_csv(
        file,
        sep=DELIMITER,
        usecols=available,
        low_memory=False
    )

    #remove duplicate caseids if applicable

    df = df.drop_duplicates(subset=CASE_ID)

    #initializes master

    if master is None:
        master = df
        continue

    #add new columns

    new_columns = [
        c for c in df.columns
        if c not in master.columns
    ]

    if new_columns:

        master = master.merge(
            df[[CASE_ID] + new_columns],
            on=CASE_ID,
            how="outer"
        )

    #update the existing columns

    existing_columns = [
        c for c in df.columns
        if c in master.columns and c != CASE_ID
    ]

    for col in existing_columns:

        temp = df[[CASE_ID, col]]

        master = master.merge(
            temp,
            on=CASE_ID,
            how="left",
            suffixes=("", "_NEW")
        )

        # Fill only missing values
        master[col] = master[col].combine_first(
            master[col + "_NEW"]
        )

        master.drop(columns=[col + "_NEW"], inplace=True)

#Report any missing values

print("\n=============================")
print("Missing Values")
print("=============================")

print(master.isna().sum())

#save dataset

master.to_csv(CSV_FOLDER / "MASTER_DATASET_MBSAQIP.csv", index=False)

print("\nFinished! The master dataset has been saved to your selected output folder!")
print(master.shape)