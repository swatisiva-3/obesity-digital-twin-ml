"""
model_utils.py

Shared feature-matrix construction, used by BOTH 05_train_model.py and
06_shap_explain.py.

The feature definitions remain unchanged:

  - numeric + contributes=True  -> __SEVERITY
  - numeric + contributes=False -> __NORM
  - binary/ordinal + contributes=True -> __SEVERITY
  - categorical/context-only -> one-hot from __STD

Domain scores / total phenotype score are deliberately NOT included as
separate MLP inputs. They are weighted sums of the variable-level features.

Train-safe imputation:
  build_feature_matrix() can receive training-set medians. When supplied,
  those medians are used for missing severity features instead of calculating
  medians from the full dataset.

This prevents validation/test observations from influencing training-time
imputation statistics.
"""

import numpy as np
import pandas as pd
import config


def build_feature_matrix(df, imputation_medians=None):
    """
    Returns:

        X
            DataFrame of numeric model-ready features.

        feature_to_variable
            {feature_column_name: variable_name}

        feature_to_domain
            {feature_column_name: domain_name}

        imputation_medians
            {feature_column_name: median used for missing-value imputation}

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the preprocessed Step 4 data.

    imputation_medians : dict or None
        Training-set medians for severity features.

        If None:
            medians are calculated from df.

        If supplied:
            those values are used instead.

        This allows Step 5 to calculate imputation statistics from the
        training split only and then apply the exact same values to
        validation/test data.
    """

    feature_frames = []
    feature_to_variable = {}
    feature_to_domain = {}

    # Store the medians actually used by this feature matrix.
    used_imputation_medians = {}

    for domain_name, domain in config.VARIABLE_RATIONALE.items():

        for var_name, spec in domain["variables"].items():

            if spec["column"] is None or spec["column"] not in df.columns:
                continue

            contributes = spec.get(
                "contributes_to_severity_score",
                True
            )

            # ---------------------------------------------------------------
            # Numeric variable contributing to clinical severity
            # ---------------------------------------------------------------
            if spec["var_type"] == "numeric" and contributes:

                col = f"{var_name}__SEVERITY"

                if col not in df.columns:
                    continue

                series = df[col].copy()

                if imputation_medians is not None and col in imputation_medians:
                    median_value = imputation_medians[col]
                else:
                    median_value = series.median()

                used_imputation_medians[col] = median_value

                series = series.fillna(median_value)

                feature_frames.append(series.rename(col))

                feature_to_variable[col] = var_name
                feature_to_domain[col] = domain_name

            # ---------------------------------------------------------------
            # Numeric variable NOT contributing to clinical severity
            # ---------------------------------------------------------------
            elif spec["var_type"] == "numeric" and not contributes:

                col = f"{var_name}__NORM"

                if col not in df.columns:
                    continue

                series = df[col].fillna(0.0)

                feature_frames.append(series.rename(col))

                feature_to_variable[col] = var_name
                feature_to_domain[col] = domain_name

            # ---------------------------------------------------------------
            # Binary / ordinal variable contributing to severity
            # ---------------------------------------------------------------
            elif spec["var_type"] in ("binary", "ordinal") and contributes:

                col = f"{var_name}__SEVERITY"

                if col not in df.columns:
                    continue

                series = df[col].copy()

                if imputation_medians is not None and col in imputation_medians:
                    median_value = imputation_medians[col]
                else:
                    median_value = series.median()

                used_imputation_medians[col] = median_value

                series = series.fillna(median_value)

                feature_frames.append(series.rename(col))

                feature_to_variable[col] = var_name
                feature_to_domain[col] = domain_name

            # ---------------------------------------------------------------
            # Categorical context-only variable -> one-hot encoding
            # ---------------------------------------------------------------
            else:

                raw_col = f"{var_name}__STD"

                if raw_col not in df.columns:
                    continue

                dummies = pd.get_dummies(
                    df[raw_col].fillna("Unknown"),
                    prefix=var_name
                )

                for c in dummies.columns:
                    feature_to_variable[c] = var_name
                    feature_to_domain[c] = domain_name

                feature_frames.append(
                    dummies.astype(float)
                )

    if not feature_frames:
        raise ValueError(
            "No model features could be constructed from the dataframe."
        )

    X = pd.concat(feature_frames, axis=1)
    X.index = df.index

    return (
        X,
        feature_to_variable,
        feature_to_domain,
        used_imputation_medians
    )
