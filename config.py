"""
config.py
=========
THIS IS THE ONE FILE YOU EDIT WHEN YOU SWITCH DATASETS.

Every other script (01 through 07) imports from here and does not hard-code
any file path, column name, weight, or clinical threshold. If you plug in a
new MBSAQIP year, a different PUF, or NHANES, you change values in THIS file
only. See ADAPTATION_GUIDE.md for a walkthrough of exactly which sections to
touch and why.

Sections in this file, in order:
  1. Dataset identity & file paths
  2. Case-linkage settings (how the 4 files are joined)
  3. Outcome variables (what we're predicting)
  4. The variable rationale (domains, weights, column mapping, clinical ranges)
  5. Domain-level weights (how the 4 domain scores combine into a total score)
  6. Flagging / tolerance rules
  7. Optuna weight-learning settings
  8. Model + SHAP + PhenoGraph settings
"""

import os

# ---------------------------------------------------------------------------
# 1. DATASET IDENTITY & FILE PATHS
# ---------------------------------------------------------------------------
# DATASET_LABEL is just used in filenames/titles/log messages so you can tell
# your output files apart when you've run this on multiple years or sources.
DATASET_LABEL = "MBSAQIP_2023_PUF"

# Folder that holds your raw input files. Change this if you move the data.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Folder everything gets written to (Excel files, plots, model files).
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# The 4 source files for this run. KEYS ("main", "intv", "reop", "read") are
# used by name throughout the code -- don't rename the keys, only the paths.
# >>> WHEN YOU SWITCH DATASETS: change these paths (and see NHANES note below) <<<
FILE_PATHS = {
    "main": os.path.join(DATA_DIR, "MBSAQIP_MAIN_2017_2020_2023_NEEDED.tsv"),
    "intv": os.path.join(DATA_DIR, "PUF_INTV_2023.tsv"),
    "reop": os.path.join(DATA_DIR, "PUF_REOP_2023.tsv"),
    "read": os.path.join(DATA_DIR, "PUF_READ_2023.tsv"),
}

# Delimiter used in the raw files. MBSAQIP PUF text exports are tab-separated.
# >>> If your new file is comma-separated, change this to "," <<<
FILE_DELIMITER = "\t"

# The patient identifier column name, as it appears in the raw files.
# It must be spelled identically in main/intv/reop/read for the join to work.
CASEID_COLUMN = "CASEID"

# NHANES NOTE (for when you adapt this later):
# NHANES does not have "main/intv/reop/read" -- it has one file per
# questionnaire/exam component (e.g. BMX, DEMO, DIQ) linked by SEQN instead
# of CASEID. To adapt: (a) add each NHANES component file as its own entry
# in FILE_PATHS, (b) set CASEID_COLUMN = "SEQN", (c) in the variable
# rationale below, set "source_file" to whichever key holds that variable.
# The data loader (01_data_loader.py) merges every file in FILE_PATHS on
# CASEID_COLUMN automatically -- it does not care how many files you list.

# ---------------------------------------------------------------------------
# 2. CASE-LINKAGE SETTINGS
# ---------------------------------------------------------------------------
# You asked for "an excel with caseid that are in all 4 files." Read this
# carefully before you run the pipeline, because it changes your cohort size
# by orders of magnitude:
#
#   - intv/reop/read are NOT parallel patient rosters. They are EVENT detail
#     files that only contain a row for a patient who actually had an
#     intervention / reoperation / readmission within 30 days. Most patients
#     (the ones who did fine) never appear in those 3 files at all.
#   - In the 2023 PUF you gave me: intv has 1,877 unique patients, reop has
#     2,588, read has 7,500 -- and patients appearing in ALL THREE of those
#     (before even joining to main) number only 240. That's a cohort of
#     patients who had an intervention AND a reoperation AND a readmission,
#     all within 30 days -- the most complicated ~0.1% of cases, not a
#     usable general training set for "predict this patient's 30-day BMI."
#   - The outcome you actually want (BMI_CLOSEST30D / WGT_CLOSEST30D) lives
#     in the MAIN file alone, for 215,101 of 230,707 patients (93%).
#
# So LINK_MODE controls which cohort feeds the model:
#   "main_only"          -> (RECOMMENDED, DEFAULT) every case in main with a
#                            valid 30-day BMI/weight. This is your real
#                            training/validation set (~215k patients).
#   "strict_intersection" -> only cases present in ALL FOUR physical files.
#                            This is what "caseid that are in all 4 files"
#                            means literally. The pipeline still produces
#                            this as its own labeled Excel sheet
#                            (see 01_data_loader.py) no matter which mode
#                            you choose below, so you always get that exact
#                            list -- it's just not used to train the model
#                            unless you set LINK_MODE to it.
LINK_MODE = "main_only"

# ---------------------------------------------------------------------------
# 3. OUTCOME VARIABLES (what the model predicts)
# ---------------------------------------------------------------------------
# Column names as they appear in FILE_PATHS["main"], plus their unit column
# (None if the column has no separate unit field).
OUTCOME_VARS = {
    "bmi_30d": {
        "column": "BMI_CLOSEST30D",
        "unit_column": None,          # BMI is unitless (kg/m^2), always
        "label": "BMI at 30-day follow-up (kg/m^2)",
    },
    "weight_30d": {
        "column": "WGT_CLOSEST30D",
        "unit_column": "WGTUNIT_CLOSEST30D",   # values are "lbs" or "kg"
        "label": "Weight at 30-day follow-up (kg)",
        "standard_unit": "kg",
    },
}
# A row is only usable for training if BOTH outcomes are present (non-null).
# >>> WHEN YOU SWITCH DATASETS: point these at your new outcome columns. <<<

# ---------------------------------------------------------------------------
# 4. VARIABLE RATIONALE
# ---------------------------------------------------------------------------
# This is your rationale document, translated into code. Each variable has:
#   rank            - as in your rationale doc (for display only)
#   weight          - % weight WITHIN its domain (each domain's weights sum
#                      to 100 -- this is checked automatically at startup)
#   column          - the column name in the raw file, or None if this
#                      variable does not exist in the current dataset
#   unit_column     - column holding the unit string, or None
#   source_file     - which key in FILE_PATHS this column lives in
#   var_type        - "numeric" | "binary" | "ordinal"
#   direction       - "two_sided" | "high_is_bad" | "low_is_bad"
#                      (controls which side of "ideal" counts as a flag,
#                      and whether the flag reason reads "High" or "Low")
#   ideal           - the reference/healthy value. For weight variables this
#                      is computed per-patient (Devine ideal-body-weight
#                      formula) instead of a fixed number -- see
#                      preprocessing.py, function `ideal_value_for`.
#   tolerance       - +/- band around `ideal` that counts as normal, in the
#                      variable's own units. Your instruction was "+/- 5
#                      units is okay." That works directly for BMI and for
#                      weight in kg. It does NOT work for lab values like
#                      Albumin (whole healthy range is 3.5-5.0) or
#                      Creatinine (whole healthy range is ~0.6-1.2) -- +/-5
#                      there would flag almost nothing or almost everything.
#                      So each numeric variable has its OWN tolerance below,
#                      defaulting to 5 only where 5 is the clinically
#                      correct unit scale. Change any of these freely.
#   severity_span   - distance from `ideal` (in the variable's units) that
#                      maps to a severity score of 100 (fully flagged /
#                      maximally severe). Used for scoring, not flagging.
#   category_scores - for ordinal/binary variables: dict mapping each raw
#                      category string to a severity score 0-100.
#   contributes_to_severity_score - True/False. See note on identity/context
#                      variables below (Section 4b).
#   flag_direction_logic - "increase" | "decrease". Used by the rule in
#                      03_scoring.py: "if more than N cases are flagged on
#                      this variable, nudge its weight up or down before
#                      Optuna ever sees it." You get to decide, per
#                      variable, whether frequent flagging on that variable
#                      should raise or lower its clinical weight. Defaulted
#                      to "increase" (a variable that's flagging often is
#                      actively discriminating real signal in THIS cohort)
#                      -- change any variable to "decrease" if you believe
#                      high prevalence dilutes its meaning instead.

# 4a'. Variables that should NEVER be reported as "flagged / abnormal", even
# though some of them still contribute a severity number to their domain.
# Two different reasons land a variable here:
#   - No genuine clinical "abnormal value" exists (AGE, HGT: there's no
#     pathological age or height the way there's a pathological BMI). AGE
#     still contributes a mild severity number (older = more cumulative
#     physiologic exposure) -- it just never gets a "High than normal" flag
#     printed next to it, which would misleadingly imply age itself is a
#     disease state.
#   - Identity/context variables (RACE, SEX, HISPANIC, OPYEAR,
#     SURGSPECIALTY_BAR) -- see the long comment on "RACE" in
#     VARIABLE_RATIONALE below for why these don't get a severity number OR
#     a flag, only raw extraction + availability to the ML model.
FLAG_INELIGIBLE_VARIABLES = {"AGE", "HGT", "RACE", "SEX", "HISPANIC", "OPYEAR", "SURGSPECIALTY_BAR"}

# 4a. FLAG_THRESHOLD_COUNT: the "more than 3 cases" rule you specified.
FLAG_THRESHOLD_COUNT = 3
# How much to nudge a variable's within-domain weight (in percentage points)
# when its flag count crosses the threshold above, before Optuna starts.
FLAG_WEIGHT_STEP = 2.0

VARIABLE_RATIONALE = {
    "Adiposity": {
        "domain_weight_prior": 25.0,  # see Section 5 -- initial guess, Optuna-tunable
        "variables": {
            "BMI_HIGH_BAR": {
                "rank": 1, "weight": 30, "column": "BMI_HIGH_BAR", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 21.7, "tolerance": 5, "severity_span": 25,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "BMI": {
                "rank": 2, "weight": 25, "column": "BMI", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "two_sided",
                "ideal": 21.7, "tolerance": 5, "severity_span": 25,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "WGT_HIGH_BAR": {
                "rank": 3, "weight": 20, "column": "WGT_HIGH_BAR", "unit_column": "WGT_HIGH_UNIT_BAR",
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad", "unit_kind": "weight",
                # "ideal" is computed per patient from height/sex (Devine IBW) -- see preprocessing.py
                "ideal": "devine_ibw", "tolerance": 5, "severity_span": 40,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "WGT_CLOSEST": {
                "rank": 4, "weight": 10, "column": "WGT_CLOSEST", "unit_column": "WGTUNIT_CLOSEST",
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad", "unit_kind": "weight",
                "ideal": "devine_ibw", "tolerance": 5, "severity_span": 40,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "MOBILITY_DEVICE": {
                "rank": 5, "weight": 8, "column": None,  # NOT FOUND in 2023 MAIN PUF -- see note below
                "unit_column": None, "source_file": "main", "var_type": "binary",
                "direction": "high_is_bad", "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
                # NOTE: I searched every column header in the 2023 MAIN PUF you gave me and
                # could not find a pre-op ambulatory-aid / mobility-device field (only
                # VENT_DEVICE, which is ventilator-related, not mobility). This variable is
                # not collected in this PUF cycle. The pipeline detects column=None
                # automatically, prints a warning, EXCLUDES it from scoring, and
                # redistributes its 8% weight proportionally across the domain's other
                # variables. If a future PUF year adds this field back, just fill in its
                # real column name here and it will start being used automatically.
            },
            "VENOUS_STASIS": {
                "rank": 6, "weight": 4, "column": "VENOUS_STASIS", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "HGT": {
                "rank": 7, "weight": 3, "column": "HGT", "unit_column": "HGTUNIT", "unit_kind": "height",
                "source_file": "main", "var_type": "numeric", "direction": "two_sided",
                "ideal": None, "tolerance": None, "severity_span": None,
                # Height has no "unhealthy" value -- it's included (inverted) only so the
                # domain reflects adiposity relative to frame size. It is NOT flag-eligible
                # (no clinical "abnormal height") but it IS still extracted, standardized to
                # cm, and available to the ML model as a raw feature.
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
        },
    },
    "Metabolic": {
        "domain_weight_prior": 25.0,
        "variables": {
            "DIABETES": {
                "rank": 1, "weight": 25, "column": "DIABETES", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"No": 0, "Yes, non-insulin": 60, "Yes, insulin": 100},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "HTN_MEDS": {
                "rank": 2, "weight": 18, "column": "NBHTN_MEDS", "unit_column": None,
                # NOTE: rationale name "HTN_MEDS" maps to raw column "NBHTN_MEDS"
                # ("Number of B/P HTN medications") in the 2023 PUF.
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 0, "tolerance": 0, "severity_span": 3,
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "HYPERLIPIDEMIA": {
                "rank": 3, "weight": 15, "column": "HYPERLIPIDEMIA", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "SLEEP_APNEA": {
                "rank": 4, "weight": 12, "column": "SLEEP_APNEA", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "RENAL_INSUFFICIENCY": {
                "rank": 5, "weight": 10, "column": "RENAL_INSUFFICIENCY", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "GERD": {
                "rank": 6, "weight": 7, "column": "GERD", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "ALBUMIN": {
                "rank": 7, "weight": 6, "column": "ALBUMIN", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "low_is_bad",
                "ideal": 4.25, "tolerance": 0.5, "severity_span": 2.0,  # g/dL scale, NOT +/-5
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "CREATININE": {
                "rank": 8, "weight": 4, "column": "CREATININE", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 0.85, "tolerance": 0.2, "severity_span": 2.0,  # mg/dL scale, NOT +/-5
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "DIALYSIS": {
                "rank": 9, "weight": 3, "column": "DIALYSIS", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
        },
    },
    "Behavioral": {
        "domain_weight_prior": 25.0,
        "variables": {
            "FUNSTATPRESURG": {
                "rank": 1, "weight": 30, "column": "FUNSTATPRESURG", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {
                    "Independent": 0, "Partially dependent": 60,
                    "Totally dependent": 100, "Unknown": None,  # None -> excluded, not scored 0
                },
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "SMOKER": {
                "rank": 2, "weight": 25, "column": "SMOKER", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "MOBILITY_DEVICE": {
                "rank": 3, "weight": 15, "column": None,  # see Adiposity note -- not in this PUF
                "unit_column": None, "source_file": "main", "var_type": "binary",
                "direction": "high_is_bad", "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "CHRONIC_STEROIDS": {
                "rank": 4, "weight": 10, "column": "IMMUNOSUPR_THER", "unit_column": None,
                # NOTE: rationale name "CHRONIC_STEROIDS" maps to "IMMUNOSUPR_THER"
                # ("chronic immunosuppressive therapy") in the 2023 PUF -- the closest
                # available proxy for chronic steroid/immunosuppressant use.
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "PREVIOUS_SURGERY": {
                "rank": 4, "weight": 10, "column": "PREVIOUS_SURGERY", "unit_column": None,
                "source_file": "main", "var_type": "binary", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "PRIORITY": {
                "rank": 4, "weight": 10, "column": None,  # NOT reliably available -- see note
                "unit_column": None, "source_file": "main", "var_type": "binary",
                "direction": "high_is_bad", "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {"Yes": 100, "No": 0},
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
                # NOTE: the only "emergency case" flag in the 2023 MAIN PUF is
                # REV_CONV_EMERGCASE, which is ONLY populated for revision/conversion
                # procedures (blank for ~89% of all cases, including every "Initial"
                # procedure). It is not a general emergency/priority flag for the whole
                # cohort, so using it here would systematically mis-flag almost everyone
                # as "not emergency" rather than "not applicable." Left as column=None
                # (excluded, weight redistributed) until a real general priority field is
                # identified. If you find one in a future PUF, put its column name here.
            },
        },
    },
    "Socio-Environmental": {
        "domain_weight_prior": 25.0,
        "variables": {
            "AGE": {
                "rank": 1, "weight": 20, "column": "AGE", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "high_is_bad",
                "ideal": 13, "tolerance": 0, "severity_span": 67,  # mild linear age-risk scaling
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "ASACLASS": {
                "rank": 1, "weight": 20, "column": "ASACLASS", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None,
                "category_scores": {
                    "ASA I - Normal/Healthy": 0, "ASA II - Mild systemic disease": 25,
                    "ASA III - Severe systemic disease": 60,
                    "ASA IV - Severe systemic disease threat to life": 85,
                    "ASA V - Moribund": 100, "None assigned": None,
                },
                "contributes_to_severity_score": True, "flag_direction_logic": "increase",
            },
            "RACE": {
                "rank": 1, "weight": 20, "column": "RACE_PUF", "unit_column": None,
                # NOTE: rationale name "RACE" maps to raw column "RACE_PUF."
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                # IMPORTANT DESIGN CHOICE -- read this before changing it:
                # RACE, SEX, and HISPANIC are included in the domain (as your rationale
                # states) to represent the *health-equity/context* dimension of the
                # phenotype, not because any category is intrinsically "unhealthy."
                # Assigning a hand-picked severity number to a race or sex category would
                # bake in a bias the data itself hasn't earned. So these three variables
                # are set to contributes_to_severity_score = False: they are NOT flagged,
                # NOT given a 0-100 severity number, and their domain weight is
                # redistributed to the variables that ARE clinically/objectively scored
                # (AGE, ASACLASS, OPYEAR, SURGSPECIALTY_BAR). They are still extracted,
                # still written to the master Excel, and still available as raw features
                # to the machine-learning model in Step 05 -- so if race, sex, or
                # ethnicity genuinely predict 30-day BMI/weight trajectory in your data,
                # the model and its SHAP output (Step 06) will surface that honestly,
                # instead of the phenotype score asserting it up front.
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
            "SEX": {
                "rank": 2, "weight": 15, "column": "SEX", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
            "HISPANIC": {
                "rank": 3, "weight": 10, "column": "HISPANIC", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
            },
            "OPYEAR": {
                "rank": 3, "weight": 10, "column": "OPYEAR", "unit_column": None,
                "source_file": "main", "var_type": "numeric", "direction": "two_sided",
                "ideal": None, "tolerance": None, "severity_span": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "decrease",
                # Pure temporal/administrative context, not a risk signal -- excluded from
                # severity scoring, weight redistributed, still extracted as a raw feature.
            },
            "SURGSPECIALTY_BAR": {
                "rank": 4, "weight": 5, "column": "SURGSPECIALTY_BAR", "unit_column": None,
                "source_file": "main", "var_type": "ordinal", "direction": "high_is_bad",
                "ideal": None, "tolerance": None, "severity_span": None, "category_scores": None,
                "contributes_to_severity_score": False, "flag_direction_logic": "increase",
                # System-access/environment context, not a patient risk factor -- excluded
                # from severity scoring for the same reason as RACE/SEX/HISPANIC above.
            },
        },
    },
}

# ---------------------------------------------------------------------------
# 5. DOMAIN-LEVEL WEIGHTS (how the 4 domain scores combine into ONE total)
# ---------------------------------------------------------------------------
# Your rationale document specifies weights WITHIN each domain (they each sum
# to 100) but does not specify how the 4 domains themselves combine into a
# single total score. I defaulted to equal weighting (25% each) as a neutral
# starting point -- this is exactly the kind of "clinician-set starting
# point" your Week 4 deck discussed. Change these to match Dr. Mancini's
# guidance whenever you have it; Optuna treats these 4 numbers the same way
# it treats the within-domain weights (a seed, not a constraint).
DOMAIN_WEIGHTS_PRIOR = {
    "Adiposity": 25.0,
    "Metabolic": 25.0,
    "Behavioral": 25.0,
    "Socio-Environmental": 25.0,
}

# ---------------------------------------------------------------------------
# 6. UNIT STANDARDIZATION
# ---------------------------------------------------------------------------
# Canonical units everything gets converted to before flagging/scoring.
STANDARD_WEIGHT_UNIT = "kg"
STANDARD_HEIGHT_UNIT = "cm"
LBS_TO_KG = 0.45359237
IN_TO_CM = 2.54

# ---------------------------------------------------------------------------
# 7. OPTUNA WEIGHT-LEARNING SETTINGS
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = 50                 # matches the "50-trial Optuna search" from your abstract
OPTUNA_SAMPLER = "TPE"               # Tree-structured Parzen Estimator
LEARNED_WEIGHTS_PATH = os.path.join(MODEL_DIR, "learned_weights.json")
# ^ After each run, the Optuna-refined weights are saved here. The NEXT run
# automatically loads this file (if it exists) and uses it as the starting
# seed instead of the clinical weights above -- this is the "new updated
# weight becomes the initial weight" continual-learning behavior you asked
# for. Delete this file if you ever want to reset back to pure clinical
# weights.

# ---------------------------------------------------------------------------
# 8. MODEL / SHAP / PHENOGRAPH SETTINGS
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2                      # train/validation split fraction
MLP_HIDDEN_LAYERS = (64, 32)         # feed-forward MLP architecture
MLP_MAX_ITER = 500

# Set to an integer (e.g. 5000) to run the whole pipeline on a random subset
# for fast iteration/testing. Set to None for the full ~215k-patient cohort.
SAMPLE_SIZE = None

# >>> PUT A CASE ID HERE to get that single patient's SHAP explanation <<<
# This can be left as None and passed instead as a --caseid command-line
# argument to 06_shap_explain.py -- see that file's __main__ block.
TARGET_CASEID = None

# PhenoGraph clustering settings (Step 07)
PHENOGRAPH_K = 30                    # neighbors used to build the k-NN graph
# Most MBSAQIP rationale variables are binary/ordinal, so a lot of patients
# land on the EXACT same domain-score value (e.g. many patients share
# identical Behavioral scores because it's built from only 4 categorical
# variables). Left alone, that duplication fragments PhenoGraph into
# hundreds of tiny, near-identical clusters (we saw 200 clusters, many
# differing by <1 point, on the 2023 PUF) -- a real but useless partition.
# Two settings fix this:
PHENOGRAPH_JITTER_STD = 1.5          # tiny random noise added to each domain
                                      # score before clustering, only to break
                                      # exact ties -- 1.5 points on a 0-100
                                      # scale, small enough not to change
                                      # which patients are genuinely similar.
PHENOGRAPH_MIN_CLUSTER_SIZE = 500    # forces a coarser, more clinically
                                      # readable partition. Lower this back
                                      # toward 10 if you want PhenoGraph's
                                      # finest-grained subgroups instead.
# Clustering itself runs on the FULL cohort (~215k patients takes ~1.5 min).
# The UMAP picture drawn from those clusters is capped at this many patients
# -- both because 215k overlapping points isn't a readable scatterplot, and
# because UMAP with a fixed random seed runs single-threaded and gets slow
# well before 215k points. Raise this if you want a denser plot and don't
# mind a longer wait; it does NOT change which/how many patients get
# clustered, only how many are drawn.
PHENOGRAPH_PLOT_SAMPLE_SIZE = 15000
