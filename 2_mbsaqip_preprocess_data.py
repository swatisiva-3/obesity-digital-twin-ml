#Purpose: Clean and encode clinical variables, create target variable, split data into testing/validation/training sets
#INPUT: Master dataset saved from "1_mbsaqip_load_data.py"
#OUTPUTS:
    #CLEAN_DATASET.csv: cleaned dataset with standardized/encoded variables and the POST_OP_BMI target
    #TRAIN_FEATURES.csv, VALIDATION_FEATURES.csv, TEST_FEATURES.csv: preprocessed feature matrices for model development.
    #TRAIN_TARGET.csv, VALIDATION_TARGET.csv, TEST_TARGET.csv: corresponding postoperative BMI target values
    #Models/feature_columns.pkl: feature names used by the model
    #Saved preprocessing objects for imputation and feature scaling as well

import pandas as pd
import numpy as np
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

CONTINUOUS_COLUMNS = [
    "AGE",
    "BMI",
    "BMI_HIGH_BAR",
    "OPLENGTH",#numerical
    "ALBUMIN",
    "CREATININE",
    "HCT",
    "HEMO",
]

MODELS_DIR = Path("Models")
MODELS_DIR.mkdir(exist_ok=True)

BINARY_COLUMNS = [
    "SEX", #male, female
    "SMOKER", #binary
    "GERD", #binary
    "HISTORY_DVT", #binary
    "ASACLASS", #"ASA I - Normal/Healthy", "ASA II - Mild systemic disease", "ASA III - Severe systemic disease", "ASA IV - Severe systemic disease threat to life"
    "HYPERLIPIDEMIA",
    "SLEEP_APNEA",
    "HIP",
    "DIABETES",
    "LIV_DIS",
    "LIV_DIS_SEVERE",
    "RENAL_INSUFFICIENCY",
    "DIALYSIS",
    "COPD",
    "STEROIDS",
    "REVCONV",
    "HYPER_MEDS"
]

TARGET_COLUMN = "POST_OP_BMI"
CASE_ID = "CASEID"

COLUMN_ENCODINGS = {

    # encode binary variables and variables that have differing names between years
    "SEX":         {
        "MALE": 1,
        "FEMALE": 0
    },

     "REVCONV":         {
        "0":0,
        "1": 1,
        "INITIAL": 0,
        "CONVERSION": 1,
        "REVISION": 1
    },

    "STEROIDS": {
        "YES": 1,
        "NO": 0
    },

    "HYPER_MEDS": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3+": 3,
        "3 OR MORE": 3
        
    },
    
    "SMOKER": {
        "YES": 1,
        "NO": 0
        },

    "GERD": {
        "YES": 1,
        "NO": 0
    },

    "HISTORY_DVT": {
        "YES": 1,
        "NO": 0
    },

    "ASACLASS": {
        "1-NO DISTURB": 0,
        "ASA I - Normal/Healthy": 0,
        "2-MILD DISTURB": 1,
        "ASA II - Mild systemic disease": 1,
        "3-SEVERE DISTURB": 2,
        "ASA III - Severe systemic disease": 2,
        "4-LIFE THREAT": 3,
        "ASA IV - Severe systemic disease threat to life": 3
    },

    "PREVIOUS_SURGERY": {
        "YES": 1,
        "NO": 0
    },

    "SLEEP_APNEA": {
        "YES": 1,
        "NO": 0
    },

    "HYPERLIPIDEMIA": {
        "YES": 1,
        "NO": 0
    },

    "COPD": {
        "YES": 1,
        "NO": 0
    },

    "DIABETES": {
        "NO": 0,
        "NON-INSULIN": 1,
        "INSULIN": 2
    },

    "HIP": {
        "YES": 1,
        "NO": 0
    },

    "RENAL_INSUFFICIENCY": {
        "YES": 1,
        "NO": 0
    },

    "DIALYSIS": {
        "YES": 1,
        "NO": 0
    },

    "LIV_DIS": {
        "YES": 1,
        "NO": 0
    },

    "LIV_DIS_SEVERE": {
        "YES": 1,
        "NO": 0
    },

    }

class PreprocessingPipeline:

    def __init__(self):

        # Configuration
        self.CASE_ID = "CASEID"
        self.MISSING_THRESHOLD = None
        self.RANDOM_STATE = 42
        self.continuous_cols = []
        self.binary_cols = []

        # Will be populated automatically
        self.binary_cols = []
        self.continuous_cols = []

        # Objects
        self.continuous_imputer = None
        self.binary_imputer = None
        self.scaler = None

    # Load data --------------------------------------------------------

    def load_data(self):

        root = Tk()
        root.withdraw()

        file_path = askopenfilename(
            title="Select MASTER_DATASET.csv from previous script",
            filetypes=[("CSV files", "*.csv")]
        )

        if not file_path:
            raise RuntimeError("No input file selected.")

        output_dir = askdirectory(
            title="Select folder to save output files"
        )

        if not output_dir:
            raise RuntimeError("No output folder selected.")

        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "Models"
        self.models_dir.mkdir(exist_ok=True)
        print(f"\nLoading: {file_path}")
        self.df = pd.read_csv(file_path, low_memory=False)
        print(f"Rows: {len(self.df):,}")
        print(f"Columns: {len(self.df.columns)}")
        
    
 
    # Create the post-operative bmi target-----------------------------------------
   

        print("\nCreating POST_OP_BMI target...")

        self.df["POST_OP_BMI"] = (
            self.df["BMI_CLOSEST30D"]
            .combine_first(self.df["BMI_DISCH"])
        )

        # Remove patients without a measured outcome
        before = len(self.df)
        self.df = self.df.dropna(subset=["POST_OP_BMI"])
        after = len(self.df)
        print(f"Removed {before-after:,} patients with no postoperative BMI.")

        # Remove the old outcome columns
        self.df.drop(
            columns=["BMI_CLOSEST30D", "BMI_DISCH"],
            inplace=True,
            errors="ignore"
        )

        #Combine variables with names that change
        self.df["HYPER_MEDS"] = (
            self.df["NBHTN_MEDS"]
            .combine_first(self.df["HTN_MEDS"])
        )

        self.df["STEROIDS"] = (
            self.df["CHRONIC_STEROIDS"]
            .combine_first(self.df["IMMUNOSUPR_THER"])
        )

        self.df["REVCONV"] = (
            self.df["CPTUNLISTED_REVCONV"]
            .combine_first(self.df["PROCEDURE_TYPE"])
        )

        # Remove the old columns
        self.df.drop(
            columns=["NBHTN_MEDS", "HTN_MEDS"],
            inplace=True,
            errors="ignore"
        )
        
        self.df.drop(
            columns=["CHRONIC_STEROIDS", "IMMUNOSUPR_THER"],
            inplace=True,
            errors="ignore"
        )

        self.df.drop(
            columns=["CPTUNLISTED_REVCONV", "PROCEDURE_TYPE"],
            inplace=True,
            errors="ignore"
        )
    
    # Basic cleaning--------------------------------------------------------------------------------

    def standardize_column_names(self):
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.upper()
        )

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=self.CASE_ID)
        removed = before - len(self.df)
        print(f"\nDuplicate CASEIDs removed: {removed}")
    
    # Encode Yes/No-------------------------------------------------------------------

    def encode_categorical_variables(self):
        print("\nEncoding categorical variables...")
        for column, mapping in COLUMN_ENCODINGS.items():
            if column not in self.df.columns:
                continue

            # Convert everything to uppercase strings first
            self.df[column] = (
                self.df[column]
                .astype(str)
                .str.strip()
                .str.upper()
            )

        # Replace only the values in the mapping
            self.df[column] = self.df[column].replace(mapping)

        # Convert to numeric
            self.df[column] = pd.to_numeric(
                self.df[column],
                errors="coerce"
            )
        print(f"  Encoded {column}")
    print("Done.")
  
    # Fix data types------------------------------------------------------------------------
 

    def fix_dtypes(self):

        for col in self.continuous_cols:
            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

        for col in self.binary_cols:
            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

        print("Fixed data types.")

    # Detect binary vs continuous--------------------------------------------

    def initialize_variable_lists(self):

        self.continuous_cols = [
            c for c in CONTINUOUS_COLUMNS
            if c in self.df.columns
        ]

        self.binary_cols = [
            c for c in BINARY_COLUMNS
            if c in self.df.columns
        ]

    # Save clean dataset--------------------------------------------------------

    def save_clean_dataset(self):
        clean_path = self.output_dir / "CLEAN_DATASET.csv"
        self.df.to_csv(clean_path, index=False)
        print(f"\nSaved: {clean_path}")

# Split data-----------------------------------------------------------------

    def split_data(self):

        train_df, temp_df = train_test_split(
            self.df,
            test_size=0.30,
            random_state=self.RANDOM_STATE
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=self.RANDOM_STATE
        )

        print("\nDataset split:")
        print(f"  Train:      {len(train_df):,}")
        print(f"  Validation: {len(val_df):,}")
        print(f"  Test:       {len(test_df):,}")

        return train_df, val_df, test_df

    # Impute + scale---------------------------------------------------------
    
    def preprocess_split(self, train_df, val_df, test_df):

        # Fit imputers on TRAIN ONLY
        if self.continuous_cols:

            self.continuous_imputer = SimpleImputer(
                strategy="median"
            )

            train_df[self.continuous_cols] = (
                self.continuous_imputer.fit_transform(
                    train_df[self.continuous_cols]
                )
            )

            val_df[self.continuous_cols] = (
                self.continuous_imputer.transform(
                    val_df[self.continuous_cols]
                )
            )

            test_df[self.continuous_cols] = (
                self.continuous_imputer.transform(
                    test_df[self.continuous_cols]
                )
            )

        binary_cols = [
            c for c in self.binary_cols
            if c in train_df.columns
        ]

        if binary_cols:

            self.binary_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        #------------------------------------------------------------------------------
        train_df[binary_cols] = self.binary_imputer.fit_transform(
            train_df[binary_cols]
        )

        val_df[binary_cols] = self.binary_imputer.transform(
            val_df[binary_cols]
        )

        test_df[binary_cols] = self.binary_imputer.transform(
            test_df[binary_cols]
        )

        # Scale continuous variables
        if self.continuous_cols:

            self.scaler = StandardScaler()

            train_df[self.continuous_cols] = (
                self.scaler.fit_transform(
                    train_df[self.continuous_cols]
                )
            )

            val_df[self.continuous_cols] = (
                self.scaler.transform(
                    val_df[self.continuous_cols]
                )
            )

            test_df[self.continuous_cols] = (
                self.scaler.transform(
                    test_df[self.continuous_cols]
                )
            )

        X_train = train_df.drop(columns=[CASE_ID, TARGET_COLUMN])
        y_train = train_df[[TARGET_COLUMN]]

        X_val = val_df.drop(columns=[CASE_ID, TARGET_COLUMN])
        y_val = val_df[[TARGET_COLUMN]]

        X_test = test_df.drop(columns=[CASE_ID, TARGET_COLUMN])
        y_test = test_df[[TARGET_COLUMN]]

        feature_columns = [
            c for c in X_train.columns
            if c != CASE_ID
        ]

        joblib.dump(
            feature_columns,
            self.models_dir / "feature_columns.pkl"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Save outputs-----------------------------------------------------------------------
    def save_outputs(self, X_train, X_val, X_test, y_train, y_val, y_test):
        X_train.to_csv(
            self.output_dir / "TRAIN_FEATURES.csv",
            index=False
        )

        X_val.to_csv(
            self.output_dir / "VALIDATION_FEATURES.csv",
            index=False
        )

        X_test.to_csv(
            self.output_dir / "TEST_FEATURES.csv",
            index=False
        )

        y_train.to_csv(
            self.output_dir / "TRAIN_TARGET.csv",
            index=False
        )

        y_val.to_csv(
            self.output_dir / "VALIDATION_TARGET.csv",
            index=False
        )

        y_test.to_csv(
            self.output_dir / "TEST_TARGET.csv",
            index=False
        )

   #Run entire pipeline----------------------------------------------------

    def run(self):

        self.load_data()

        self.standardize_column_names()

        self.initialize_variable_lists()

        self.remove_duplicates()

        self.encode_categorical_variables()

        self.fix_dtypes()


        train_df, val_df, test_df = self.split_data()

        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_split(
            train_df,
            val_df,
            test_df
        )

# Save cleaned dataset AFTER preprocessing
        self.save_clean_dataset()

        self.save_outputs(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

# Main---------------------------------------------------------------------------

if __name__ == "__main__":

    pipeline = PreprocessingPipeline()

    pipeline.run()
