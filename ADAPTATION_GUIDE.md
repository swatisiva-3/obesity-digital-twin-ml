# Adaptation Guide — switching this pipeline to a new dataset

Every script (`01` through `07`) reads its file paths, column names, clinical
weights, and thresholds from **`config.py`**. Nothing else in the codebase
should need to change for a new year of MBSAQIP, a different MBSAQIP-derived
extract, or NHANES. This guide walks through exactly what to edit in
`config.py`, in the order you'd naturally hit it.

Treat this as a checklist. Not every dataset switch needs every step below —
a new PUF *year* is mostly Step 1; a genuinely new source like NHANES is all
of them.

---

## Step 1 — Point at your new files

In `config.py`, Section 1:

```python
DATASET_LABEL = "MBSAQIP_2024_PUF"     # just a label, used in filenames/titles

FILE_PATHS = {
    "main": os.path.join(DATA_DIR, "PUF_MAINFINAL_2024.txt"),
    "intv": os.path.join(DATA_DIR, "PUF_INTVFINAL_2024.txt"),
    "reop": os.path.join(DATA_DIR, "PUF_REOPFINAL_2024.txt"),
    "read": os.path.join(DATA_DIR, "PUF_READFINAL_2024.txt"),
}
```

Also check `FILE_DELIMITER` (tab `"\t"` for MBSAQIP text exports; `","` for a
plain CSV) and `CASEID_COLUMN` (the patient-ID column name — this must be
spelled identically across every file you list).

## Step 2 — Decide `LINK_MODE`

Read the long comment above `LINK_MODE` in `config.py` Section 2 once. Short
version: `"main_only"` (default) uses every patient in your main file with a
valid outcome; `"strict_intersection"` restricts modeling to patients present
in literally every file you listed, which is usually a tiny, complication-
selected cohort. You'll almost always want `"main_only"`.

## Step 3 — Point `OUTCOME_VARS` at your new outcome columns

If you're still predicting BMI/weight at some follow-up point, just update the
column names. If you switch outcomes entirely (say, a readmission flag, or a
different follow-up window), update `column`, `unit_column` (or set it to
`None` if the new outcome has no unit conversion issue), and `label`.

## Step 4 — Re-map the variable rationale (the big one)

This is `config.VARIABLE_RATIONALE` in Section 4. For **every** variable, you
need to know: does a column with this exact meaning exist in the new file,
and if so, what is it called?

**How to find out**, for a new PUF year or similar file:

```bash
head -1 your_new_file.txt | tr '\t' '\n' | nl        # lists every column, numbered
```

Then for each rationale variable:

- **Same meaning, same or different name** → update `"column"` to the new
  name. (This is what happened going into this codebase: the rationale's
  `HTN_MEDS` is the file's `NBHTN_MEDS`, `RACE` is `RACE_PUF`,
  `CHRONIC_STEROIDS` is `IMMUNOSUPR_THER`. Renames like this are common
  between registry years — always re-check, don't assume last year's mapping
  still holds.)
- **Genuinely not present in the new file** → set `"column": None`. The
  pipeline detects this automatically (Step 1 prints a warning, Step 3/4
  redistribute that variable's weight to the rest of its domain). This is
  exactly what happened with `MOBILITY_DEVICE` and `PRIORITY` in the 2023
  PUF — neither exists in this cycle, so both are set to `None` right now.
  If a future file adds them back, just fill in the real column name and
  they'll start contributing again with no other code changes.
- **New variable you want to add that isn't in the current rationale** → add
  a new entry to the relevant domain's `"variables"` dict, following the
  same shape as the existing entries (rank, weight, column, var_type,
  direction, ideal, tolerance, severity_span or category_scores,
  contributes_to_severity_score, flag_direction_logic). Remember the
  within-domain weights should still make sense summing to ~100 (the code
  will renormalize automatically either way, but keeping it close to 100
  makes the raw numbers easier to sanity-check by eye).

**For each numeric variable**, also sanity-check `"ideal"`, `"tolerance"`,
and `"severity_span"` against the ACTUAL values in the new file — a
tolerance that made sense for one lab's reference range might not transfer.
A fast way to check:

```python
import pandas as pd
df = pd.read_csv("your_new_file.txt", sep="\t", usecols=["YOUR_COLUMN"])
print(df["YOUR_COLUMN"].astype(float).describe())
```

**For each categorical variable**, re-check `"category_scores"` against the
actual category strings in the new file — they must match EXACTLY
(case-sensitive), or that patient's value won't be recognized and will be
treated as missing. Check with:

```python
print(df["YOUR_COLUMN"].value_counts())
```

## Step 5 — Units

If the new file uses different unit strings than `"in"/"cm"` or `"lbs"/"kg"`,
extend `utils.height_to_cm()` / `utils.weight_to_kg()` to recognize them.

## Step 6 — Reset learned weights (usually)

`models/learned_weights.json` is where Optuna's continual-learning state
lives (see `config.LEARNED_WEIGHTS_PATH`). If you're switching to a
meaningfully different population or outcome, DELETE this file before
re-running Step 4, so Optuna starts fresh from the clinical baseline instead
of warm-starting from weights learned on the old dataset. If you're just
adding a new year of the SAME kind of data and want continuity, leave it —
that's the intended "learns the trend, then updates" behavior.

## Step 7 — Re-run

```bash
python run_pipeline.py                         # everything, in order
python run_pipeline.py --steps 1 2 3            # just re-check extraction/flagging/scoring
python run_pipeline.py --caseid <A_REAL_CASEID> # also produce that patient's SHAP explanation
```

---

## Worked example: adapting this to NHANES

NHANES is structurally different from MBSAQIP (many small component files
instead of 4 case files, linked by `SEQN` instead of `CASEID`, no bariatric
surgery outcome at all), so it touches every section above:

1. **`FILE_PATHS`**: one entry per NHANES component you need, e.g.
   `{"bmx": ".../BMX_J.XPT", "diq": ".../DIQ_J.XPT", "demo": ".../DEMO_J.XPT", ...}`.
   NHANES ships `.XPT` (SAS transport) files — read them with
   `pandas.read_sas(path, format="xport")` instead of `read_csv`; you'll need
   to adjust `read_one_file()` in `01_data_loader.py` to branch on file
   extension, since NHANES and MBSAQIP use different formats.
2. **`CASEID_COLUMN`**: `"SEQN"`.
3. **`VARIABLE_RATIONALE`**: this is where your existing
   `NHANES_Variable_Rationale.docx` (from earlier in this project) becomes
   directly useful — it already has the real NHANES variable codes (BMXBMI,
   BMXWT, DIQ010, LBXGH, BPQ050A, etc.) and which MBSAQIP variable each one
   maps to conceptually. Walk that document domain by domain and fill in
   `"column"` with the real NHANES code and `"source_file"` with whichever
   key you gave that component in `FILE_PATHS`.
4. **`OUTCOME_VARS`**: NHANES has no bariatric-surgery 30-day outcome. You'd
   redefine this entirely — e.g. against the NHANES Linked Mortality Files,
   or a weight-trajectory proxy, as discussed in your Week 4 deck's NHANES
   slide. This also means `01_data_loader.py`'s "all 4 files" linkage logic
   (Section 2/Step B) is MBSAQIP-specific bookkeeping you can simply skip for
   an NHANES run — it doesn't hurt anything to leave it, but it isn't
   meaningful there since NHANES has no intv/reop/read equivalent.
5. **Units**: NHANES anthropometrics are usually pre-standardized (cm, kg)
   already, so the unit-conversion step likely becomes a no-op — just
   confirm with the codebook before assuming that.

Everything downstream of Step 1 (preprocessing, scoring, Optuna, the MLP,
SHAP, PhenoGraph) needs zero changes to run against NHANES once
`VARIABLE_RATIONALE` and `OUTCOME_VARS` point at the right NHANES columns —
that's the entire point of keeping all of this in one config file.
