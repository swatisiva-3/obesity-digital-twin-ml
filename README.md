# Obesity Digital Twin -- Presurgical Phenotype-Scoring Pipeline

A full pipeline: load MBSAQIP PUF files -> extract only the rationale
variables -> standardize units / flag abnormal values / normalize ->
clinically-weighted domain + total phenotype scores -> Optuna-refined
weights -> a feed-forward MLP predicting 30-day BMI and weight -> SHAP
explainability (global + single-patient) -> PhenoGraph clustering.

This run was built and tested against your real 2023 MBSAQIP PUF files
(main/intv/reop/read). **Everything is driven by `config.py`** -- see
`ADAPTATION_GUIDE.md` for exactly what to change when you point this at a
new year, a different extract, or NHANES.

## Quickstart

```bash
python 00_setup_environment.py      # once: creates venv/, installs packages
source venv/bin/activate            # (Windows: venv\Scripts\activate)

# put your 4 raw files in data/ (see "Expected input files" below), then:
python run_pipeline.py                          # runs everything, ~6-8 min on 215k patients
python run_pipeline.py --caseid 1703142          # also explain one specific patient
```

Or run each numbered step by hand (each is independently runnable and
picks up where the last one left off):

```bash
python 01_data_loader.py
python 02_preprocessing.py
python 03_scoring.py
python 04_optuna_weight_learning.py
python 05_train_model.py
python 06_shap_explain.py --caseid 1703142
python 07_phenograph_visualization.py
```

## Expected input files

Drop these into `data/` (or update `config.FILE_PATHS` to point elsewhere):

| File | Expected default name |
|---|---|
| Main case file | `PUF_MAINFINAL_2023.txt` |
| Interventions | `PUF_INTVFINAL_2023.txt` |
| Reoperations  | `PUF_REOPFINAL_2023.txt` |
| Readmissions  | `PUF_READFINAL_2023.txt` |

Tab-separated, one CASEID column each. This delivery does NOT include your
actual patient data files (you already have them) -- just drop them in.

## What each file in this project is

| File | Role |
|---|---|
| `config.py` | **Edit this one file to adapt to a new dataset.** Every path, column mapping, clinical threshold, and weight lives here. |
| `utils.py` | Small shared helpers (unit conversion, save/load, logging). |
| `scoring_engine.py` | The domain/total scoring math -- used by both Step 3 and Step 4 (Optuna calls it hundreds of times). |
| `model_utils.py` | Builds the feature matrix the MLP trains on -- used by both Step 5 and Step 6, so they agree on what "feature #7" means. |
| `01_data_loader.py` | Loads the 4 PUF files, extracts only rationale-relevant columns, builds the "CASEID in all 4 files" Excel, assembles the analytic cohort. |
| `02_preprocessing.py` | Unit standardization, normalization, abnormal-value flagging with reasons -- writes the master flagged dataset. |
| `03_scoring.py` | Clinical weights (+ the ">3 flags" nudge) -> per-variable/domain/total phenotype scores. |
| `04_optuna_weight_learning.py` | TPE search over the weights, seeded from Step 3's clinical weights (or last run's learned weights), scored against real validation-set prediction error. Saves `models/learned_weights.json`. |
| `05_train_model.py` | Trains the two feed-forward MLPs (30-day BMI, 30-day weight). |
| `06_shap_explain.py` | Global + single-patient SHAP explanations, variable-wise and domain-wise. **`--caseid <ID>` is where you plug in a patient.** |
| `07_phenograph_visualization.py` | PhenoGraph clustering on the 4 domain scores + UMAP visualization. |
| `run_pipeline.py` | Runs any/all of the above in order. |
| `ADAPTATION_GUIDE.md` | Step-by-step: what to change in `config.py` for a new dataset, including a worked NHANES example. |

## Design decisions you should know about (not silent defaults)

These are judgment calls I made where your instructions didn't fully specify
an answer, or where the real data didn't have what the rationale expected.
Each is also commented at its exact location in `config.py` -- this is just
the one-page summary.

1. **`MOBILITY_DEVICE` and `PRIORITY` aren't in the 2023 MBSAQIP PUF.** I
   searched every column in the file you gave me; neither a pre-op
   ambulatory-aid field nor a general emergency-case flag exists this cycle
   (the closest field, `REV_CONV_EMERGCASE`, is blank for ~89% of cases --
   it only applies to revision/conversion procedures). Both variables are
   set to `column: None` in config.py, which the pipeline handles
   automatically (their domain weight is redistributed to the rest of that
   domain). If a future PUF year restores these fields, just fill in the
   real column name.

2. **"CASEID in all 4 files" is a ~240-patient cohort, not your model's
   training set.** intv/reop/read are event-detail files that only contain a
   patient who actually had that specific 30-day event. Patients in ALL
   THREE of those (before even joining main) number 240 out of 230,707 --
   the most complicated ~0.1% of cases. That literal list is still produced
   every run (`outputs/caseids_in_all_4_files.xlsx`), but the model trains
   on the ~215k patients in `main` with valid 30-day follow-up
   (`config.LINK_MODE = "main_only"`). Full reasoning is in `config.py`
   Section 2 -- change `LINK_MODE` there if you actually want the small
   cohort to drive the model.

3. **Domain-level weights (how the 4 domains combine into ONE total score)
   default to equal 25% each.** Your rationale document gives weights
   *within* each domain (each sums to 100) but doesn't specify how the 4
   domains combine. I used an equal-weight starting point --
   `config.DOMAIN_WEIGHTS_PRIOR` -- exactly the kind of "clinician-set
   starting point, Optuna free to move it" pattern from your Week 4 deck.
   Update this the moment Dr. Mancini gives real domain-level weights.

4. **RACE, SEX, HISPANIC, and SURGSPECIALTY_BAR don't get a hand-set
   severity score.** Assigning a numeric "how unhealthy is this race/sex"
   value isn't something the data justifies -- your rationale's own
   reasoning for including them is health-equity CONTEXT, not that any
   category is inherently pathological. So these four are excluded from the
   severity-score math (their domain weight is redistributed to AGE and
   ASACLASS), but they're still extracted, still in the master Excel, and
   still available to the ML model as raw one-hot features -- if they
   genuinely predict the 30-day outcome, the model and SHAP will show that
   honestly instead of the phenotype score asserting it up front. Full
   reasoning is on the `"RACE"` entry in `config.py`.

5. **"+/- 5 units" only applies literally to BMI and weight (in kg).** For
   lab values like Albumin (whole healthy range is 3.5-5.0 g/dL) or
   Creatinine (~0.6-1.2 mg/dL), a flat 5-unit tolerance would flag either
   almost nobody or almost everybody. Each numeric variable has its own
   `tolerance` in `config.py`, defaulting to 5 only where 5 is the
   clinically correct scale.

6. **Weight variables are flagged against a personalized "ideal weight"
   (Devine formula, using each patient's own height and sex), not one fixed
   number.** There's no single universal "ideal weight" the way there is an
   ideal BMI -- Devine ideal-body-weight is the standard clinical way to
   personalize that reference point.

7. **Almost every patient is flagged on BMI/weight/ASA class.** This is
   correct, not a bug: this is a bariatric surgery population by definition,
   so of course the overwhelming majority have clinically abnormal BMI and
   an elevated ASA classification going in. `outputs/flag_rate_summary.csv`
   shows the exact rate per variable.

8. **PhenoGraph clustering uses a small tie-breaking jitter and a larger
   minimum cluster size than PhenoGraph's own default.** Left at defaults,
   PhenoGraph fragmented into ~200 nearly-identical micro-clusters, because
   most rationale variables are binary/ordinal and produce lots of patients
   with the EXACT same domain-score vector. `config.PHENOGRAPH_JITTER_STD`
   (small random noise, just to break exact ties) and a
   `PHENOGRAPH_MIN_CLUSTER_SIZE` of 500 bring that down to a much more
   readable ~40 clusters, each labeled with its dominant phenotype domain
   in `outputs/phenograph_cluster_summary.csv`.

## What's included in this delivered package vs. what you regenerate

To keep this download small, the very large intermediate files
(`01_raw_extract.parquet` through `04_final_scored.parquet`, the full
`master_flagged_data.csv`, and the full per-patient `phenotype_scores.xlsx`
/ `phenograph_clusters.xlsx`) are **not** included here -- they're exactly
reproduced by running the pipeline on your machine (roughly 6-8 minutes end
to end on the full ~215k-patient cohort). What IS included under `outputs/`
is a real sample of what a full run produces: the flag-rate summary, the
Optuna optimization history, the SHAP plots and comparison table, one
worked single-patient SHAP example, and the PhenoGraph cluster summary +
visualizations -- all generated from an actual run of this exact code
against your real 2023 PUF files.
