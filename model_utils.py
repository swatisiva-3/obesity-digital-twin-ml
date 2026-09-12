"""
model_utils.py
===============
Shared feature-matrix construction, used by BOTH 05_train_model.py (to
train the MLP) and 06_shap_explain.py (to explain it) -- they must build
IDENTICAL features from the same dataframe, or SHAP values would be
explaining a different model than the one that's loaded. Lives outside the
numbered scripts for the same reason scoring_engine.py does (module names
can't start with a digit).

Feature-building rule, per variable in config.VARIABLE_RATIONALE:
  - numeric, contributes_to_severity_score=True  -> use its __SEVERITY column
    (0-100 severity, already computed in Step 2 -- this is what lets SHAP
    speak in the same "how severe/abnormal is this" language as the
    phenotype score itself).
  - numeric, contributes_to_severity_score=False (HGT, OPYEAR) -> use its
    __NORM column (z-scored raw value) -- still a real predictive feature,
    just not part of the hand-set severity scoring.
  - categorical, contributes_to_severity_score=True -> use its __SEVERITY
    column (already a meaningful 0-100 ordinal encoding, e.g. DIABETES:
    No=0, non-insulin=60, insulin=100).
  - categorical, contributes_to_severity_score=False (RACE, SEX, HISPANIC,
    SURGSPECIALTY_BAR) -> one-hot encoded from the raw category, so the
    model (and SHAP) can discover any real predictive relationship without
    a hand-set severity number asserting one in advance.

Domain scores / total score are DELIBERATELY NOT included as separate model
inputs, even though they're available -- they're just weighted sums of the
variable-level features already in the matrix, so including both would be
circular. Instead, "domain-wise SHAP" is computed by SUMMING the per-variable
SHAP values within each domain (SHAP values are additive, so this is valid
and is exactly what 06_shap_explain.py does).
"""

import numpy as np
import pandas as pd
import config


def build_feature_matrix(df):
    """
    Returns (X, feature_to_variable, feature_to_domain):
      X                   - DataFrame of numeric model-ready features
      feature_to_variable - {feature_column_name: variable_name}
      feature_to_domain   - {feature_column_name: domain_name}
    One variable can expand to several feature columns (one-hot categories),
    which is why this returns mapping dicts rather than assuming a 1:1 match.
    """
    feature_frames = []
    feature_to_variable = {}
    feature_to_domain = {}

    for domain_name, domain in config.VARIABLE_RATIONALE.items():
        for var_name, spec in domain["variables"].items():
            if spec["column"] is None or spec["column"] not in df.columns:
                continue  # not available in this dataset

            contributes = spec.get("contributes_to_severity_score", True)

            if spec["var_type"] == "numeric" and contributes:
                col = f"{var_name}__SEVERITY"
                series = df[col].fillna(df[col].median())
                feature_frames.append(series.rename(col))
                feature_to_variable[col] = var_name
                feature_to_domain[col] = domain_name

            elif spec["var_type"] == "numeric" and not contributes:
                col = f"{var_name}__NORM"
                if col not in df.columns:
                    continue
                series = df[col].fillna(0.0)
                feature_frames.append(series.rename(col))
                feature_to_variable[col] = var_name
                feature_to_domain[col] = domain_name

            elif spec["var_type"] in ("binary", "ordinal") and contributes:
                col = f"{var_name}__SEVERITY"
                series = df[col].fillna(df[col].median())
                feature_frames.append(series.rename(col))
                feature_to_variable[col] = var_name
                feature_to_domain[col] = domain_name

            else:  # categorical, context-only -> one-hot
                raw_col = f"{var_name}__STD"
                if raw_col not in df.columns:
                    continue
                dummies = pd.get_dummies(df[raw_col].fillna("Unknown"), prefix=var_name)
                for c in dummies.columns:
                    feature_to_variable[c] = var_name
                    feature_to_domain[c] = domain_name
                feature_frames.append(dummies.astype(float))

    X = pd.concat(feature_frames, axis=1)
    X.index = df.index
    return X, feature_to_variable, feature_to_domain
