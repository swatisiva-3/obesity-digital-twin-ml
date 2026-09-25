"""
02_preprocessing.py
====================
STEP 2 of the pipeline. Run standalone with:   python 02_preprocessing.py
(Requires 01_data_loader.py to have been run first.)

What this does, matching your three preprocessing layers:

  1. STANDARDIZATION OF UNITS
     Every height goes to centimeters, every weight goes to kilograms,
     regardless of which unit the original row used ("in"/"cm", "lbs"/"kg").

  2. NORMALIZATION
     Every numeric variable also gets a z-score-normalized column (mean 0,
     std 1) for use as a machine-learning feature later. This is SEPARATE
     from flagging (flagging below uses real clinical units, because a
     clinical threshold like "BMI > 30" only means something in real BMI
     units, not in a normalized/rescaled space).

  3. FLAGGING
     For every variable that has a defined "ideal" reference value (see
     config.VARIABLE_RATIONALE), each patient's value is compared to that
     ideal. If it's within `tolerance` units, it's fine. Outside that band,
     the patient is flagged, with a reason string in exactly the format you
     asked for: "High than normal" or "Low than normal", plus how far apart
     as a percentage. Binary/ordinal variables (DIABETES, SMOKER, ASACLASS,
     etc.) are flagged instead by category (e.g. "Yes" = flagged), since
     there's no numeric percent-deviation for a category.
     Variables in config.FLAG_INELIGIBLE_VARIABLES (age, height, race, sex,
     Hispanic ethnicity, operative year, surgeon specialty) are never
     flagged -- see the comment on that set in config.py for why.

Output: outputs/master_flagged_data.parquet (full data, used by later
steps), outputs/master_flagged_data.csv (full data, human-readable/portable),
and outputs/master_flagged_data_PREVIEW.xlsx (first N rows, a real Excel
file you can open directly -- see config.EXCEL_PREVIEW_ROWS. The full
dataset is ~215k rows x ~90 columns, which Excel can technically open but
is slow to work with interactively; the CSV/parquet are the full data).
"""
import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

import os
import numpy as np
import pandas as pd

import config
import utils

EXCEL_PREVIEW_ROWS = 3000  # how many rows go into the .xlsx you can open directly


def flag_numeric_variable(values, idealvals, tolerance, direction):
    """
    Vectorized flagging for one numeric variable across the whole cohort.
    Returns four Series: flagged (bool), reason (str), pct_deviation (float),
    severity_0_100 (float) -- severity is computed by the caller (needs
    severity_span), this function returns the raw signed deviation for that.
    """
    diff = values - idealvals  # positive = above ideal, negative = below ideal

    if direction == "high_is_bad":
        is_flag = diff > tolerance
        bad_side = diff.clip(lower=0)          # only "too high" counts as severity
    elif direction == "low_is_bad":
        is_flag = diff < -tolerance
        bad_side = (-diff).clip(lower=0)       # only "too low" counts as severity
    else:  # two_sided
        is_flag = diff.abs() > tolerance
        bad_side = diff.abs()

    # % deviation from ideal, guarding against ideal == 0 (e.g. HTN_MEDS)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(idealvals != 0, (diff.abs() / idealvals.abs()) * 100, np.nan)
    pct = pd.Series(pct, index=values.index)

    reason = pd.Series(np.where(is_flag, np.where(diff > 0, "High than normal", "Low than normal"),
                                 "Within normal range"), index=values.index)
    # Append the percent-apart figure where we have one; otherwise report the
    # raw absolute difference (this only happens when ideal == 0, e.g. medication counts).
    pct_text = np.where(pct.notna(), pct.round(1).astype(str) + "% apart from ideal",
                         diff.abs().round(2).astype(str) + " units apart from ideal (ideal = 0)")
    reason_full = pd.Series(np.where(is_flag, reason + " by " + pd.Series(pct_text, index=values.index), reason),
                             index=values.index)
    reason_full = reason_full.where(values.notna() & idealvals.notna(), "Not assessable (missing data)")
    is_flag = is_flag.where(values.notna() & idealvals.notna(), False)

    return is_flag.fillna(False), reason_full, pct, bad_side


def process_numeric_variable(df, domain_name, var_name, spec):
    """Standardize units (if needed), flag, and score one numeric variable."""
    col = spec["column"]
    raw = pd.to_numeric(df[col], errors="coerce")

    # --- 1. unit standardization ---
    # unit_kind ("height"/"weight") tells us which conversion applies; it's
    # an explicit config field (not guessed from the column name) so this
    # keeps working correctly if you rename columns for a new dataset.
    unit_kind = spec.get("unit_kind")
    if spec.get("unit_column") and unit_kind == "height":
        std_values = df.apply(lambda r: utils.height_to_cm(r[col], r[spec["unit_column"]]), axis=1)
    elif spec.get("unit_column") and unit_kind == "weight":
        std_values = df.apply(lambda r: utils.weight_to_kg(r[col], r[spec["unit_column"]]), axis=1)
    else:
        std_values = raw

    df[f"{var_name}__STD"] = std_values

    flag_eligible = var_name not in config.FLAG_INELIGIBLE_VARIABLES and spec.get("tolerance") is not None
    severity = pd.Series(np.nan, index=df.index)

    if flag_eligible:
        if spec.get("ideal") == "devine_ibw":
            ideal_series = df["IDEAL_BODY_WEIGHT_KG"]
        else:
            ideal_series = pd.Series(spec["ideal"], index=df.index)
        is_flag, reason, pct, bad_side = flag_numeric_variable(
            std_values, ideal_series, spec["tolerance"], spec["direction"]
        )
        df[f"{var_name}__FLAG"] = is_flag
        df[f"{var_name}__FLAG_REASON"] = reason
        df[f"{var_name}__PCT_DEVIATION"] = pct
        if spec.get("contributes_to_severity_score") and spec.get("severity_span"):
            severity = (bad_side / spec["severity_span"] * 100).clip(lower=0, upper=100)
            severity = severity.where(std_values.notna() & ideal_series.notna())
    else:
        df[f"{var_name}__FLAG"] = False
        df[f"{var_name}__FLAG_REASON"] = "Not flagged (no clinical abnormal threshold for this variable)"
        df[f"{var_name}__PCT_DEVIATION"] = np.nan
        # AGE is the one flag-ineligible variable that still contributes a mild
        # severity number (see config.py). Everything else in
        # FLAG_INELIGIBLE_VARIABLES has contributes_to_severity_score = False.
        if spec.get("contributes_to_severity_score") and spec.get("ideal") is not None and spec.get("severity_span"):
            diff = std_values - spec["ideal"]
            bad_side = diff.clip(lower=0) if spec["direction"] == "high_is_bad" else diff.abs()
            severity = (bad_side / spec["severity_span"] * 100).clip(lower=0, upper=100)

    df[f"{var_name}__SEVERITY"] = severity
    return df


def process_categorical_variable(df, domain_name, var_name, spec):
    """Flag and score one binary/ordinal (categorical) variable."""
    col = spec["column"]
    raw = df[col]
    category_scores = spec.get("category_scores")
    flag_eligible = var_name not in config.FLAG_INELIGIBLE_VARIABLES and category_scores is not None

    if category_scores is not None:
        severity = raw.map(category_scores)  # unmapped/unknown categories -> NaN (excluded, not zero)
    else:
        severity = pd.Series(np.nan, index=df.index)

    if flag_eligible:
        # Flagged = maps to a severity score above 0 (i.e. an "abnormal"/risk-positive category).
        is_flag = severity.fillna(0) > 0
        reason = pd.Series(np.where(is_flag, "Abnormal / risk-positive category: " + raw.astype(str),
                                     "Within normal category"), index=df.index)
        reason = reason.where(raw.notna() & severity.notna(), "Not assessable (missing or unmapped category)")
        is_flag = is_flag.where(raw.notna() & severity.notna(), False)
    else:
        is_flag = pd.Series(False, index=df.index)
        reason = pd.Series("Not flagged (context variable, no clinical abnormal category)", index=df.index)
        if not spec.get("contributes_to_severity_score", True):
            severity = pd.Series(np.nan, index=df.index)

    df[f"{var_name}__STD"] = raw  # categoricals have no unit conversion; STD = raw category
    df[f"{var_name}__FLAG"] = is_flag
    df[f"{var_name}__FLAG_REASON"] = reason
    df[f"{var_name}__PCT_DEVIATION"] = np.nan  # categorical -> no percent-apart concept
    df[f"{var_name}__SEVERITY"] = severity
    return df


def main():
    df = utils.load_dataframe("01_raw_extract")
    utils.log("02_preprocessing", f"Loaded {len(df):,} rows from Step 1.")

    # --- Devine ideal-body-weight, computed once, reused by WGT_HIGH_BAR / WGT_CLOSEST ---
    hgt_spec = config.VARIABLE_RATIONALE["Adiposity"]["variables"]["HGT"]
    if hgt_spec["column"] and hgt_spec["column"] in df.columns:
        df["HGT__STD_TEMP"] = df.apply(
            lambda r: utils.height_to_cm(r[hgt_spec["column"]], r[hgt_spec["unit_column"]]), axis=1
        )
        sex_col = config.VARIABLE_RATIONALE["Socio-Environmental"]["variables"]["SEX"]["column"]
        df["IDEAL_BODY_WEIGHT_KG"] = df.apply(
            lambda r: utils.devine_ideal_body_weight_kg(r["HGT__STD_TEMP"], r[sex_col]), axis=1
        )
    else:
        df["IDEAL_BODY_WEIGHT_KG"] = np.nan
        utils.log("02_preprocessing", "WARNING: height column unavailable -- weight-based flags will be skipped.")

    # --- walk every variable in the rationale and process it ---
    for domain_name, domain in config.VARIABLE_RATIONALE.items():
        for var_name, spec in domain["variables"].items():
            if spec["column"] is None or spec["column"] not in df.columns:
                continue  # not available in this dataset -- handled in scoring step's weight redistribution
            if spec["var_type"] == "numeric":
                df = process_numeric_variable(df, domain_name, var_name, spec)
            else:
                df = process_categorical_variable(df, domain_name, var_name, spec)

    df.drop(columns=["HGT__STD_TEMP"], errors="ignore", inplace=True)

    # --- standardize the outcome columns too (same unit logic, no flagging) ---
    bmi_spec, wgt_spec = config.OUTCOME_VARS["bmi_30d"], config.OUTCOME_VARS["weight_30d"]
    df["OUTCOME_BMI_30D"] = pd.to_numeric(df[bmi_spec["column"]], errors="coerce")
    df["OUTCOME_WEIGHT_30D_KG"] = df.apply(
        lambda r: utils.weight_to_kg(r[wgt_spec["column"]], r[wgt_spec["unit_column"]]), axis=1
    )

    # --- 2. NORMALIZATION: z-score every numeric __STD column for ML use later ---
    # (Fit on the full cohort here for the exploratory/master-excel view. The
    # actual model-training step re-fits its own scaler on the TRAIN split
    # only, to avoid leaking validation-set statistics into training -- see
    # 05_train_model.py.)
    std_numeric_cols = [c for c in df.columns if c.endswith("__STD") and pd.api.types.is_numeric_dtype(df[c])]
    for c in std_numeric_cols:
        mean, std = df[c].mean(), df[c].std()
        norm_col = c.replace("__STD", "__NORM")
        df[norm_col] = (df[c] - mean) / std if std and not np.isnan(std) and std != 0 else 0.0

    utils.save_dataframe(df, "02_master_flagged")

    # --- write outputs: full CSV + full parquet (already done) + a real, openable Excel preview ---
    utils.ensure_output_dirs()
    csv_path = os.path.join(config.OUTPUT_DIR, "master_flagged_data.csv")
    df.to_csv(csv_path, index=False)
    utils.log("02_preprocessing", f"Wrote full data ({len(df):,} rows) -> {csv_path}")

    preview_path = os.path.join(config.OUTPUT_DIR, "master_flagged_data_PREVIEW.xlsx")
    df.head(EXCEL_PREVIEW_ROWS).to_excel(preview_path, index=False, engine="openpyxl")
    utils.log("02_preprocessing", f"Wrote first {min(EXCEL_PREVIEW_ROWS, len(df)):,} rows as an openable "
                                   f".xlsx -> {preview_path} (full data is in the .csv / .parquet -- "
                                   f"opening 215k rows x ~90 columns directly in Excel is slow, so this "
                                   f"preview is what you'd actually want to eyeball by hand).")

    # --- quick flag-rate summary, printed to console, useful sanity check ---
    flag_cols = [c for c in df.columns if c.endswith("__FLAG")]
    summary = pd.DataFrame({
        "variable": [c.replace("__FLAG", "") for c in flag_cols],
        "n_flagged": [int(df[c].sum()) for c in flag_cols],
        "pct_flagged": [round(100 * df[c].mean(), 2) for c in flag_cols],
    }).sort_values("n_flagged", ascending=False)
    utils.log("02_preprocessing", "Flag-rate summary (top 10):\n" + summary.head(10).to_string(index=False))
    summary.to_csv(os.path.join(config.OUTPUT_DIR, "flag_rate_summary.csv"), index=False)

    return df


if __name__ == "__main__":
    main()
