"""
03_scoring.py
==============
STEP 3 of the pipeline. Run standalone with:   python 03_scoring.py
(Requires 02_preprocessing.py to have been run first.)

What this does:
  1. Starts from the clinical weights in config.VARIABLE_RATIONALE (with any
     dataset-unavailable variable's weight already redistributed within its
     domain -- see scoring_engine.default_variable_weights()).
  2. Applies your flag-count rule: any variable flagged on more than
     config.FLAG_THRESHOLD_COUNT patients gets its weight nudged up or down
     (config.FLAG_WEIGHT_STEP), in the direction YOU set per variable via
     "flag_direction_logic" in config.py.
  3. Computes, for every patient: each variable's 0-100 severity score
     (already computed in Step 2), each domain's 0-100 score, and one
     0-100 total score across all 4 domains.
  4. Saves this as the pipeline's "initial effective weights" -- the exact
     starting point 04_optuna_weight_learning.py will seed its search from.

This script does NOT run Optuna. It produces the clinically-grounded
baseline that Optuna (Step 4) is seeded from and compared against.
"""

import os
import pandas as pd

import config
import utils
import scoring_engine


def main():
    df = utils.load_dataframe("02_master_flagged")
    utils.log("03_scoring", f"Loaded {len(df):,} rows from Step 2.")

    clinical_weights = scoring_engine.default_variable_weights()
    effective_weights = scoring_engine.apply_flag_count_weight_nudge(df, clinical_weights)
    domain_weights = scoring_engine.default_domain_weights()

    utils.log("03_scoring", "Effective within-domain weights after the flag-count nudge:")
    for domain, vweights in effective_weights.items():
        utils.log("03_scoring", f"  {domain}: " + ", ".join(f"{v}={w:.1f}%" for v, w in
                                                              sorted(vweights.items(), key=lambda x: -x[1])))

    domain_scores_df, total_score = scoring_engine.full_scoring_pipeline(df, effective_weights, domain_weights)

    result = pd.concat([
        df[[config.CASEID_COLUMN, "OUTCOME_BMI_30D", "OUTCOME_WEIGHT_30D_KG"]],
        df[[c for c in df.columns if c.endswith("__SEVERITY")]],  # per-variable scores
        domain_scores_df,                                         # per-domain scores
    ], axis=1)
    result["TOTAL_PHENOTYPE_SCORE"] = total_score

    utils.save_dataframe(pd.concat([df, domain_scores_df, total_score.rename("TOTAL_PHENOTYPE_SCORE")], axis=1),
                          "03_scored")

    utils.ensure_output_dirs()
    out_path = os.path.join(config.OUTPUT_DIR, "phenotype_scores.xlsx")
    result.to_excel(out_path, index=False, engine="openpyxl")
    utils.log("03_scoring", f"Wrote per-patient variable/domain/total scores -> {out_path}")

    utils.save_json(
        {"variable_weights": effective_weights, "domain_weights": domain_weights},
        os.path.join(config.MODEL_DIR, "clinical_baseline_weights.json"),
    )

    utils.log("03_scoring", "Domain score summary (mean, std):")
    for d in config.VARIABLE_RATIONALE:
        col = f"{d}_SCORE"
        if col in domain_scores_df.columns:
            utils.log("03_scoring", f"  {d}: mean={domain_scores_df[col].mean():.1f}  std={domain_scores_df[col].std():.1f}")
    utils.log("03_scoring", f"Total phenotype score: mean={total_score.mean():.1f}  std={total_score.std():.1f}")

    return result


if __name__ == "__main__":
    main()
