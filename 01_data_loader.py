"""
01_data_loader.py
==================
STEP 1 of the pipeline. Run standalone with:   python 01_data_loader.py

What this does:
  A. Reads config.FILE_PATHS and, for each file, loads ONLY the columns that
     the variable rationale (config.VARIABLE_RATIONALE) and the outcome
     definitions (config.OUTCOME_VARS) actually reference -- not the whole
     file. This is what makes it "only fetch data columns that the variable
     names relate to."
  B. Builds the literal "CASEID present in all 4 files" list you asked for,
     with each such patient's 30-day BMI and weight, and writes it to its
     own Excel file: outputs/caseids_in_all_4_files.xlsx. This happens
     regardless of which LINK_MODE you've chosen for modeling (see the long
     comment in config.py Section 2 for why that cohort is tiny and why the
     model itself uses a different cohort by default).
  C. Assembles the analytic dataset used by every later step, according to
     config.LINK_MODE, and saves it (outputs/01_raw_extract.parquet) for
     02_preprocessing.py to pick up.

WHAT TO CHANGE WHEN YOU SWITCH DATASETS: nothing in this file. Everything
it does is driven by config.py. See ADAPTATION_GUIDE.md.
"""

import os
import pandas as pd
import numpy as np

import config
import utils


def collect_columns_by_file():
    """
    Walk the variable rationale + outcome definitions and figure out, per
    source file, exactly which columns need to be read. Variables whose
    "column" is None (not available in this dataset -- see config.py notes
    on MOBILITY_DEVICE / PRIORITY) are automatically skipped here; later
    steps detect their absence and redistribute their weight.
    """
    columns_by_file = {key: {config.CASEID_COLUMN} for key in config.FILE_PATHS}

    for domain_name, domain in config.VARIABLE_RATIONALE.items():
        for var_name, spec in domain["variables"].items():
            if spec["column"] is None:
                utils.log("01_data_loader", f"SKIP (not in dataset): {domain_name} / {var_name} "
                                             f"-- column is None in config.py, weight will be redistributed later.")
                continue
            src = spec["source_file"]
            columns_by_file[src].add(spec["column"])
            if spec.get("unit_column"):
                columns_by_file[src].add(spec["unit_column"])

    for outcome_name, spec in config.OUTCOME_VARS.items():
        columns_by_file["main"].add(spec["column"])
        if spec.get("unit_column"):
            columns_by_file["main"].add(spec["unit_column"])

    # A few extra main-file columns that are useful context even though they
    # aren't in the rationale: REOP30/READ30/INTV30 tell you (per-patient,
    # without needing to join the 3 detail files) whether that patient had
    # a reoperation/readmission/intervention within 30 days.
    for extra in ("REOP30", "READ30", "INTV30"):
        columns_by_file["main"].add(extra)

    return {k: sorted(v) for k, v in columns_by_file.items()}


def read_one_file(file_key):
    path = config.FILE_PATHS[file_key]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing source file: {path}")

    wanted_columns = collect_columns_by_file()[file_key]

    # MBSAQIP files may be supplied as either the original tab-separated
    # text exports or as Excel workbooks.
    if path.lower().endswith((".xlsx", ".xls")):
        header_df = pd.read_excel(path, sheet_name=0, nrows=0)
        header = header_df.columns.tolist()

        missing = [c for c in wanted_columns if c not in header]
        if missing:
            utils.log("01_data_loader", f"[{file_key}] Missing requested columns ({len(missing)}): " + ", ".join(missing))

        usable_columns = [c for c in wanted_columns if c in header]

        df = pd.read_excel(
            path,
            sheet_name=0,
            usecols=usable_columns,
            dtype=str,
        )
    else:
        header = pd.read_csv(
            path,
            sep=config.FILE_DELIMITER,
            nrows=0,
        ).columns.tolist()

        missing = [c for c in wanted_columns if c not in header]
        if missing:
            utils.log("01_data_loader", f"[{file_key}] Missing requested columns ({len(missing)}): " + ", ".join(missing))

        usable_columns = [c for c in wanted_columns if c in header]

        df = pd.read_csv(
            path,
            sep=config.FILE_DELIMITER,
            usecols=usable_columns,
            dtype=str,
            low_memory=False,
        )

    utils.log("01_data_loader", f"[{file_key}] Loaded {len(df):,} rows x {len(df.columns):,} columns from {os.path.basename(path)}")

    return df

def build_all_4_files_linkage_sheet(main_df, intv_ids, reop_ids, read_ids):
    """
    Build the literal deliverable you asked for: an Excel file listing every
    CASEID that appears in ALL FOUR files, with that patient's 30-day BMI
    and 30-day weight (each in its own column, as requested).
    """
    main_ids = set(main_df[config.CASEID_COLUMN])
    in_all_4 = main_ids & intv_ids & reop_ids & read_ids

    bmi_col = config.OUTCOME_VARS["bmi_30d"]["column"]
    wgt_col = config.OUTCOME_VARS["weight_30d"]["column"]
    wgt_unit_col = config.OUTCOME_VARS["weight_30d"]["unit_column"]

    subset = main_df[main_df[config.CASEID_COLUMN].isin(in_all_4)].copy()
    subset["BMI_30_days_later"] = pd.to_numeric(subset[bmi_col], errors="coerce")
    subset["WEIGHT_30_days_later_kg"] = subset.apply(
        lambda r: utils.weight_to_kg(r[wgt_col], r[wgt_unit_col]), axis=1
    )
    out = subset[[config.CASEID_COLUMN, "BMI_30_days_later", "WEIGHT_30_days_later_kg"]].rename(
        columns={config.CASEID_COLUMN: "CASEID"}
    )

    out_path = os.path.join(config.OUTPUT_DIR, "caseids_in_all_4_files.xlsx")
    utils.ensure_output_dirs()
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="CaseIDs in all 4 files", index=False)
        summary = pd.DataFrame({
            "File": ["main", "intv", "reop", "read", "IN ALL 4 (this cohort)"],
            "Unique CASEIDs": [len(main_ids), len(intv_ids), len(reop_ids), len(read_ids), len(in_all_4)],
        })
        summary.to_excel(writer, sheet_name="Coverage summary", index=False)
    utils.log("01_data_loader", f"Wrote {len(out):,} cases present in all 4 files -> {out_path}")
    utils.log("01_data_loader", f"NOTE: this is a small, complication-selected cohort by construction "
                                 f"(patients with an intervention AND reoperation AND readmission). "
                                 f"See config.py Section 2 for why the model below uses a different, "
                                 f"much larger cohort by default (LINK_MODE = 'main_only').")
    return in_all_4


def main():
    utils.ensure_output_dirs()

    main_df = read_one_file("main")
    intv_df = read_one_file("intv")
    reop_df = read_one_file("reop")
    read_df = read_one_file("read")

    intv_ids = set(intv_df[config.CASEID_COLUMN])
    reop_ids = set(reop_df[config.CASEID_COLUMN])
    read_ids = set(read_df[config.CASEID_COLUMN])

    # Deliverable B: the literal "caseid in all 4 files" Excel, with BMI/weight at 30 days.
    in_all_4 = build_all_4_files_linkage_sheet(main_df, intv_ids, reop_ids, read_ids)

    # Deliverable C: the analytic cohort actually used downstream.
    bmi_col = config.OUTCOME_VARS["bmi_30d"]["column"]
    wgt_col = config.OUTCOME_VARS["weight_30d"]["column"]
    has_outcomes = pd.to_numeric(main_df[bmi_col], errors="coerce").notna() & main_df[wgt_col].notna()

    if config.LINK_MODE == "strict_intersection":
        keep_mask = main_df[config.CASEID_COLUMN].isin(in_all_4) & has_outcomes
        utils.log("01_data_loader", "LINK_MODE = 'strict_intersection' -- modeling cohort restricted to "
                                     "CASEIDs found in all 4 files. This will be a SMALL cohort.")
    elif config.LINK_MODE == "main_only":
        keep_mask = has_outcomes
    else:
        raise ValueError(f"Unrecognized config.LINK_MODE = {config.LINK_MODE!r}. "
                          f"Must be 'main_only' or 'strict_intersection'.")

    analytic_df = main_df[keep_mask].copy()

    # Attach informational flags (not used as filters) recording whether each
    # kept patient also shows up in the 3 event-detail files -- useful context
    # for the model/SHAP step without throwing away the rest of the cohort.
    analytic_df["HAD_INTV_LINKED_RECORD"] = analytic_df[config.CASEID_COLUMN].isin(intv_ids)
    analytic_df["HAD_REOP_LINKED_RECORD"] = analytic_df[config.CASEID_COLUMN].isin(reop_ids)
    analytic_df["HAD_READ_LINKED_RECORD"] = analytic_df[config.CASEID_COLUMN].isin(read_ids)

    if config.SAMPLE_SIZE is not None and len(analytic_df) > config.SAMPLE_SIZE:
        analytic_df = analytic_df.sample(n=config.SAMPLE_SIZE, random_state=config.RANDOM_SEED)
        utils.log("01_data_loader", f"config.SAMPLE_SIZE is set -- subsampled to {len(analytic_df):,} rows "
                                     f"for a faster test run. Set SAMPLE_SIZE = None in config.py for the full run.")

    utils.log("01_data_loader", f"Final analytic cohort: {len(analytic_df):,} patients "
                                 f"(LINK_MODE = '{config.LINK_MODE}').")

    utils.save_dataframe(analytic_df, "01_raw_extract")
    return analytic_df


if __name__ == "__main__":
    main()
