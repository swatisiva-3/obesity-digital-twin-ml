"""
utils.py
========
Small shared helpers used by more than one pipeline step. Nothing in here
is dataset-specific -- all dataset-specific values live in config.py.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import config


def log(step_name, message):
    """Uniform console logging so pipeline output is easy to scan."""
    print(f"[{step_name}] {message}")


def ensure_output_dirs():
    """Create the outputs/ and models/ folders if they don't exist yet."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)


def save_dataframe(df, name):
    """
    Save an intermediate DataFrame as both a .parquet (fast, exact dtypes --
    used by the next script in the pipeline) and a .csv (human-readable, so
    you can open it yourself). `name` is a short key like "01_raw_extract".
    """
    ensure_output_dirs()
    parquet_path = os.path.join(config.OUTPUT_DIR, f"{name}.parquet")
    df.to_parquet(parquet_path, index=False)
    log("utils", f"Saved {len(df):,} rows -> {parquet_path}")
    return parquet_path


def load_dataframe(name):
    """Load a DataFrame previously saved by save_dataframe(). Raises a clear
    error telling you which earlier script to run if the file is missing."""
    parquet_path = os.path.join(config.OUTPUT_DIR, f"{name}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Could not find {parquet_path}. You need to run the earlier "
            f"pipeline step that produces '{name}' before this script."
        )
    return pd.read_parquet(parquet_path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
def weight_to_kg(value, unit):
    """Convert a weight value to kilograms given its unit string ('lbs'/'kg')."""
    if pd.isna(value):
        return np.nan
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return np.nan
    unit = str(unit).strip().lower()
    if unit in ("kg", "kgs", "kilogram", "kilograms"):
        return float(value)
    if unit in ("lb", "lbs", "pound", "pounds"):
        return float(value) * config.LBS_TO_KG
    return np.nan  # unrecognized unit -- treated as missing, not guessed


def height_to_cm(value, unit):
    """Convert a height value to centimeters given its unit string ('in'/'cm')."""
    if pd.isna(value):
        return np.nan
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return np.nan
    unit = str(unit).strip().lower()
    if unit in ("cm", "centimeter", "centimeters"):
        return float(value)
    if unit in ("in", "ins", "inch", "inches"):
        return float(value) * config.IN_TO_CM
    return np.nan


def devine_ideal_body_weight_kg(height_cm, sex):
    """
    Devine formula for ideal body weight, the standard clinical formula used
    to give an individualized 'ideal weight' reference point for a person of
    a given height and sex (rather than one fixed number for everybody).
        Men:   50.0 kg + 2.3 kg per inch over 5 feet (60 inches)
        Women: 45.5 kg + 2.3 kg per inch over 5 feet (60 inches)
    Returns NaN if height or sex is missing/unrecognized (the caller should
    then skip flagging for that patient's weight variables rather than guess).
    """
    if pd.isna(height_cm) or height_cm <= 0 or sex is None or (isinstance(sex, float) and pd.isna(sex)):
        return np.nan
    height_in = height_cm / config.IN_TO_CM
    extra_inches = max(0.0, height_in - 60.0)
    sex = str(sex).strip().lower()
    if sex.startswith("m"):
        return 50.0 + 2.3 * extra_inches
    if sex.startswith("f"):
        return 45.5 + 2.3 * extra_inches
    return np.nan  # e.g. "Non-binary" in this dataset -- no validated Devine variant, skip
