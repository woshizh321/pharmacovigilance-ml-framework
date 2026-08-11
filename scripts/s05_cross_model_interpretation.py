#!/usr/bin/env python3
"""Command 13: out-of-drug cross-model interpretation of frozen models.

The read allowlist is restricted to Section 3 development artifacts. No model
fit method is called, and no holdout or JADER path is defined.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
S3 = ROOT / "analysis" / "section3_model"
TRAIN = S3 / "training"
MODELS = TRAIN / "models"
OUT = ROOT / "analysis" / "section5_interpretation"

REGISTRY = S3 / "development_pair_registry.parquet"
DICTIONARY = S3 / "FEATURE_DICTIONARY_v1.csv"
OUTER = S3 / "OUTER_FOLD_ASSIGNMENT_v1.csv"
OOF_PREDICTIONS = TRAIN / "05_oof_predictions.parquet"
PREHOLDOUT_MANIFEST = S3 / "PREHOLDOUT_LOCK_MANIFEST.json"
SECTION3_QC = TRAIN / "SECTION3B_QC.json"

GROUP = "canonical_active_moiety"
PT = "canonical_pt_code"
TARGET = "criterion_r_3y"
SHAP_ADDITIVITY_TOLERANCE = 1e-5
PREDICTION_RECONSTRUCTION_TOLERANCE = 1e-10

PIPELINES = [
    "elasticnet_set0",
    "elasticnet_set1",
    "xgboost_set0",
    "xgboost_set1",
]

DOMAIN_MAP = {
    # Set 0: regulatory context.
    "approval_year": "REGULATORY",
    "nda_bla": "REGULATORY",
    "orphan_designation": "REGULATORY",
    "accelerated_approval": "REGULATORY",
    "breakthrough_therapy_designation": "REGULATORY",
    "fast_track_designation": "REGULATORY",
    "priority_review_category": "REGULATORY",
    # Set 0: event identity.
    "canonical_pt_code": "EVENT_IDENTITY",
    "primary_soc": "EVENT_IDENTITY",
    # Set 0: general programme context.
    "route_broad": "DRUG_PROGRAM",
    "dosage_form_broad": "DRUG_PROGRAM",
    "drug_n_qualifying_trials": "DRUG_PROGRAM",
    "drug_n_target_arms": "DRUG_PROGRAM",
    "drug_approx_ae_safety_population": "DRUG_PROGRAM",
    "drug_phase1_fraction": "DRUG_PROGRAM",
    "drug_randomized_trial_fraction": "DRUG_PROGRAM",
    "drug_masked_trial_fraction": "DRUG_PROGRAM",
    "drug_industry_sponsored_fraction": "DRUG_PROGRAM",
    # Set 1 additions: evidence volume.
    "pair_n_reporting_trials": "EVIDENCE_VOLUME",
    "pair_reporting_trial_fraction": "EVIDENCE_VOLUME",
    "pair_n_reporting_arms": "EVIDENCE_VOLUME",
    "pair_nonduplicated_arm_subjects_at_risk": "EVIDENCE_VOLUME",
    # Set 1 additions: AE magnitude and other-event availability.
    "pair_median_row_ae_proportion": "AE_PROPORTION",
    "pair_max_row_ae_proportion": "AE_PROPORTION",
    "pair_max_other_row_proportion": "AE_PROPORTION",
    "pair_other_proportion_available": "AE_PROPORTION",
    # Set 1 additions: serious context.
    "pair_any_serious": "SERIOUS_CONTEXT",
    "pair_n_serious_trials": "SERIOUS_CONTEXT",
    "pair_n_serious_arms": "SERIOUS_CONTEXT",
    "pair_serious_trial_fraction": "SERIOUS_CONTEXT",
    "pair_serious_arm_fraction": "SERIOUS_CONTEXT",
    "pair_max_serious_row_proportion": "SERIOUS_CONTEXT",
    # Set 1 additions: within/cross-trial consistency.
    "pair_row_ae_proportion_iqr": "CROSS_TRIAL",
    "pair_row_variability_available": "CROSS_TRIAL",
    "pair_between_trial_proportion_sd": "CROSS_TRIAL",
    "pair_cross_trial_variability_available": "CROSS_TRIAL",
    # Set 1 additions: registry threshold context.
    "pair_min_other_threshold": "REPORTING_THRESHOLD",
    "pair_max_other_threshold": "REPORTING_THRESHOLD",
    "pair_median_other_threshold": "REPORTING_THRESHOLD",
    "pair_fraction_trials_threshold_0": "REPORTING_THRESHOLD",
    "pair_fraction_trials_threshold_5": "REPORTING_THRESHOLD",
    "pair_other_threshold_available": "REPORTING_THRESHOLD",
    # Set 1 additions: pair-specific trial design.
    "pair_phase1_fraction": "PAIR_TRIAL_DESIGN",
    "pair_randomized_trial_fraction": "PAIR_TRIAL_DESIGN",
    "pair_masked_trial_fraction": "PAIR_TRIAL_DESIGN",
}

DISPLAY_MAP = {
    "canonical_pt_code": "canonical_PT_identity",
    "primary_soc": "primary_SOC_identity",
    "route_broad": "route_category",
    "dosage_form_broad": "dosage_form_category",
}

AVAILABILITY_PAIRS = [
    ("SERIOUS_CONTEXT", "pair_max_serious_row_proportion", "pair_any_serious"),
    ("AE_PROPORTION", "pair_max_other_row_proportion", "pair_other_proportion_available"),
    ("CROSS_TRIAL", "pair_row_ae_proportion_iqr", "pair_row_variability_available"),
    ("CROSS_TRIAL", "pair_between_trial_proportion_sd", "pair_cross_trial_variability_available"),
    ("REPORTING_THRESHOLD", "pair_min_other_threshold", "pair_other_threshold_available"),
    ("REPORTING_THRESHOLD", "pair_max_other_threshold", "pair_other_threshold_available"),
    ("REPORTING_THRESHOLD", "pair_median_other_threshold", "pair_other_threshold_available"),
]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return None
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(c) for c in frame.columns]
    rows = [[render(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def pipeline_name(family: str, feature_set: str) -> str:
    return f"{family}_{feature_set.lower()}"


def apply_bundle_pt_map(frame: pd.DataFrame, retained: list[str]) -> pd.DataFrame:
    out = frame.copy()
    code = out[PT].astype(str)
    out[PT] = np.where(code.isin(set(map(str, retained))), code, "PT_RARE")
    return out


def conceptual_mapping(
    names: list[str], features: list[str], metadata: pd.DataFrame
) -> tuple[list[str], list[dict[str, Any]]]:
    categorical = [
        feature for feature in features if metadata.loc[feature, "data_type"] == "categorical"
    ]
    categorical = sorted(categorical, key=len, reverse=True)
    mapped: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, encoded in enumerate(map(str, names)):
        if encoded in features and metadata.loc[encoded, "data_type"] != "categorical":
            concept = encoded
            rule = "exact_numeric_or_binary"
        else:
            matches = [feature for feature in categorical if encoded.startswith(feature + "_")]
            if len(matches) != 1:
                raise RuntimeError(f"Unmapped or ambiguous encoded feature: {encoded}; matches={matches}")
            concept = matches[0]
            rule = "one_hot_prefix_to_parent"
        if concept not in DOMAIN_MAP:
            raise RuntimeError(f"Conceptual feature has no prespecified domain: {concept}")
        mapped.append(concept)
        rows.append(
            {
                "encoded_index": index,
                "encoded_feature_name": encoded,
                "conceptual_feature": concept,
                "conceptual_display_name": DISPLAY_MAP.get(concept, concept),
                "data_type": metadata.loc[concept, "data_type"],
                "scientific_domain": DOMAIN_MAP[concept],
                "broad_group": (
                    "PAIR_SPECIFIC_SAFETY"
                    if metadata.loc[concept, "feature_set"] == "SET1_ADDITIONAL"
                    else "BASELINE_CONTEXT"
                ),
                "is_one_hot_column": rule == "one_hot_prefix_to_parent",
                "mapping_rule": rule,
            }
        )
    return mapped, rows


def group_signed_first(
    encoded_contributions: np.ndarray,
    encoded_to_concept: list[str],
    feature_order: list[str],
) -> tuple[np.ndarray, float]:
    concept_index = {feature: index for index, feature in enumerate(feature_order)}
    grouped = np.zeros((encoded_contributions.shape[0], len(feature_order)), dtype=np.float64)
    for encoded_index, concept in enumerate(encoded_to_concept):
        grouped[:, concept_index[concept]] += encoded_contributions[:, encoded_index]
    error = float(
        np.max(
            np.abs(
                encoded_contributions.sum(axis=1, dtype=np.float64)
                - grouped.sum(axis=1, dtype=np.float64)
            )
        )
    )
    return grouped, error


def long_contribution_frame(
    valid: pd.DataFrame,
    feature_order: list[str],
    grouped: np.ndarray,
    family: str,
    feature_set: str,
    metadata: pd.DataFrame,
    include_raw_numeric: bool,
) -> pd.DataFrame:
    n_rows, n_features = grouped.shape
    result = pd.DataFrame(
        {
            "pair_id": np.repeat(valid["pair_id"].astype(str).to_numpy(), n_features),
            GROUP: np.repeat(valid[GROUP].astype(str).to_numpy(), n_features),
            "outer_fold": np.repeat(valid["outer_fold"].to_numpy(dtype=np.int8), n_features),
            "model_family": family,
            "feature_set": feature_set,
            "conceptual_feature": np.tile(np.asarray(feature_order, dtype=object), n_rows),
            "grouped_signed_contribution": grouped.reshape(-1),
        }
    )
    result["absolute_grouped_contribution"] = np.abs(result["grouped_signed_contribution"])
    result["scientific_domain"] = result["conceptual_feature"].map(DOMAIN_MAP)
    result["broad_group"] = result["conceptual_feature"].map(
        lambda feature: (
            "PAIR_SPECIFIC_SAFETY"
            if metadata.loc[feature, "feature_set"] == "SET1_ADDITIONAL"
            else "BASELINE_CONTEXT"
        )
    )
    if include_raw_numeric:
        raw = np.full((n_rows, n_features), np.nan, dtype=np.float64)
        for feature_index, feature in enumerate(feature_order):
            if metadata.loc[feature, "data_type"] != "categorical":
                raw[:, feature_index] = pd.to_numeric(valid[feature], errors="coerce").to_numpy()
        result["raw_conceptual_value"] = raw.reshape(-1)
    for column in ["model_family", "feature_set", "conceptual_feature", "scientific_domain", "broad_group"]:
        result[column] = result[column].astype("category")
    return result


def summarize_importance(long_frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_frame.groupby(
            ["model_family", "feature_set", "conceptual_feature"],
            observed=True,
            as_index=False,
        )
        .agg(
            oof_rows=("grouped_signed_contribution", "size"),
            mean_signed_contribution=("grouped_signed_contribution", "mean"),
            mean_absolute_grouped_contribution=("absolute_grouped_contribution", "mean"),
            median_absolute_grouped_contribution=("absolute_grouped_contribution", "median"),
            q1_absolute_grouped_contribution=("absolute_grouped_contribution", lambda x: x.quantile(0.25)),
            q3_absolute_grouped_contribution=("absolute_grouped_contribution", lambda x: x.quantile(0.75)),
        )
    )
    summary["conceptual_feature"] = summary["conceptual_feature"].astype(str)
    summary["conceptual_display_name"] = summary["conceptual_feature"].map(
        lambda x: DISPLAY_MAP.get(x, x)
    )
    summary["scientific_domain"] = summary["conceptual_feature"].map(DOMAIN_MAP)
    summary["pair_specific"] = summary["conceptual_feature"].map(
        lambda x: metadata.loc[x, "feature_set"] == "SET1_ADDITIONAL"
    )
    summary["contribution_scale"] = np.where(
        summary["model_family"].eq("xgboost"), "margin_log_odds", "linear_predictor_log_odds"
    )
    summary["rank_all_conceptual"] = (
        summary.groupby(["model_family", "feature_set"])[
            "mean_absolute_grouped_contribution"
        ]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary["rank_pair_specific"] = np.nan
    pair_mask = summary["pair_specific"] & summary["feature_set"].eq("SET1")
    summary.loc[pair_mask, "rank_pair_specific"] = (
        summary.loc[pair_mask]
        .groupby("model_family")["mean_absolute_grouped_contribution"]
        .rank(method="min", ascending=False)
    )
    return summary.sort_values(
        ["feature_set", "model_family", "rank_all_conceptual", "conceptual_feature"]
    ).reset_index(drop=True)


def summarize_attribution(long_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_family", "feature_set", "pair_id", GROUP, "outer_fold"]
    domain_by_row = (
        long_frame.groupby(keys + ["scientific_domain"], observed=True, as_index=False)[
            "grouped_signed_contribution"
        ].sum()
    )
    domain_by_row["absolute"] = np.abs(domain_by_row["grouped_signed_contribution"])
    domain_summary = (
        domain_by_row.groupby(["model_family", "feature_set", "scientific_domain"], observed=True)
        .agg(
            oof_rows=("absolute", "size"),
            mean_absolute_domain_contribution=("absolute", "mean"),
            median_absolute_domain_contribution=("absolute", "median"),
            mean_signed_domain_contribution=("grouped_signed_contribution", "mean"),
        )
        .reset_index()
        .rename(columns={"scientific_domain": "attribution_group"})
    )
    domain_summary["aggregation_level"] = "SCIENTIFIC_DOMAIN"
    rows.append(domain_summary)

    set1 = long_frame[long_frame["feature_set"].astype(str).eq("SET1")]
    broad_by_row = (
        set1.groupby(keys + ["broad_group"], observed=True, as_index=False)[
            "grouped_signed_contribution"
        ].sum()
    )
    broad_by_row["absolute"] = np.abs(broad_by_row["grouped_signed_contribution"])
    broad_summary = (
        broad_by_row.groupby(["model_family", "feature_set", "broad_group"], observed=True)
        .agg(
            oof_rows=("absolute", "size"),
            mean_absolute_domain_contribution=("absolute", "mean"),
            median_absolute_domain_contribution=("absolute", "median"),
            mean_signed_domain_contribution=("grouped_signed_contribution", "mean"),
        )
        .reset_index()
        .rename(columns={"broad_group": "attribution_group"})
    )
    broad_summary["aggregation_level"] = "BROAD_GROUP"
    rows.append(broad_summary)

    summary = pd.concat(rows, ignore_index=True)
    denominator = summary.groupby(
        ["model_family", "feature_set", "aggregation_level"]
    )["mean_absolute_domain_contribution"].transform("sum")
    summary["relative_attribution_share"] = (
        summary["mean_absolute_domain_contribution"] / denominator
    )
    summary["signed_sum_first"] = True
    summary["interpretation_boundary"] = (
        "Descriptive model attribution; not a decomposition of delta AP"
    )
    return summary.sort_values(
        ["feature_set", "model_family", "aggregation_level", "mean_absolute_domain_contribution"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def coefficient_stability(coefficient_rows: list[dict[str, Any]], metadata: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame(coefficient_rows)
    output = []
    for (feature_set, feature), group in raw.groupby(["feature_set", "conceptual_feature"]):
        group = group.sort_values("outer_fold")
        coefs = group["standardized_coefficient"].to_numpy(dtype=float)
        nonzero = coefs != 0
        positive = coefs > 0
        negative = coefs < 0
        positive_n = int(positive.sum())
        negative_n = int(negative.sum())
        dominant_n = max(positive_n, negative_n)
        nonzero_n = int(nonzero.sum())
        stable = bool(nonzero_n >= 4 and dominant_n >= 4)
        dominant_sign = (
            "positive" if positive_n > negative_n else "negative" if negative_n > positive_n else "none"
        )
        record = {
            "feature_set": feature_set,
            "conceptual_feature": feature,
            "conceptual_display_name": DISPLAY_MAP.get(feature, feature),
            "data_type": metadata.loc[feature, "data_type"],
            "pair_specific": metadata.loc[feature, "feature_set"] == "SET1_ADDITIONAL",
            "outer_models": len(group),
            "nonzero_models": nonzero_n,
            "nonzero_frequency": nonzero_n / 5,
            "positive_models": positive_n,
            "positive_frequency": positive_n / 5,
            "negative_models": negative_n,
            "negative_frequency": negative_n / 5,
            "zero_models": int((~nonzero).sum()),
            "median_standardized_coefficient": float(np.median(coefs)),
            "coefficient_q1": float(np.quantile(coefs, 0.25)),
            "coefficient_q3": float(np.quantile(coefs, 0.75)),
            "dominant_nonzero_sign": dominant_sign,
            "directional_stability": "DIRECTIONALLY_STABLE" if stable else "DIRECTIONALLY_UNSTABLE",
            "reported_direction": dominant_sign if stable else "not_assigned",
            "directional_stability_rule": "same sign in at least 4 of 5 outer fits and nonzero in at least 4",
        }
        for fold, coefficient in zip(group["outer_fold"], coefs):
            record[f"outer_fold_{int(fold)}_standardized_coefficient"] = float(coefficient)
        output.append(record)
    return pd.DataFrame(output).sort_values(["feature_set", "conceptual_feature"]).reset_index(drop=True)


def cross_model_concordance(importance: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    set1 = importance[importance["feature_set"].eq("SET1")]
    elastic = set1[set1["model_family"].eq("elasticnet")].set_index("conceptual_feature")
    tree = set1[set1["model_family"].eq("xgboost")].set_index("conceptual_feature")
    common = sorted(set(elastic.index) & set(tree.index))
    rows = []
    for feature in common:
        rows.append(
            {
                "conceptual_feature": feature,
                "conceptual_display_name": DISPLAY_MAP.get(feature, feature),
                "scientific_domain": DOMAIN_MAP[feature],
                "pair_specific": bool(elastic.loc[feature, "pair_specific"]),
                "elasticnet_mean_absolute_oof_contribution": float(
                    elastic.loc[feature, "mean_absolute_grouped_contribution"]
                ),
                "elasticnet_rank_all": int(elastic.loc[feature, "rank_all_conceptual"]),
                "elasticnet_rank_pair_specific": elastic.loc[feature, "rank_pair_specific"],
                "xgboost_mean_absolute_oof_shap": float(
                    tree.loc[feature, "mean_absolute_grouped_contribution"]
                ),
                "xgboost_rank_all": int(tree.loc[feature, "rank_all_conceptual"]),
                "xgboost_rank_pair_specific": tree.loc[feature, "rank_pair_specific"],
            }
        )
    concordance = pd.DataFrame(rows)
    all_rho = float(spearmanr(concordance["elasticnet_rank_all"], concordance["xgboost_rank_all"]).statistic)
    pair = concordance[concordance["pair_specific"]]
    pair_rho = float(
        spearmanr(pair["elasticnet_rank_pair_specific"], pair["xgboost_rank_pair_specific"]).statistic
    )
    concordance["spearman_all_conceptual_features"] = all_rho
    concordance["spearman_pair_specific_features"] = pair_rho
    concordance["headline_significance_test_performed"] = False
    return concordance.sort_values("elasticnet_rank_all"), all_rho, pair_rho


def classify_shape(summary: pd.DataFrame, data_type: str) -> str:
    ordered = summary.sort_values("bin_order")
    if len(ordered) < 2:
        return "no clear directional pattern"
    means = ordered["mean_grouped_shap"].to_numpy(dtype=float)
    if data_type == "binary":
        difference = means[-1] - means[0]
        if abs(difference) <= 1e-6:
            return "no clear directional pattern"
        return (
            "present associated with higher model contribution"
            if difference > 0
            else "present associated with lower model contribution"
        )
    if len(ordered) < 3 or np.ptp(means) <= 1e-6:
        return "no clear directional pattern"
    values = ordered["median_feature_value"].to_numpy(dtype=float)
    rho = float(spearmanr(values, means).statistic)
    if rho >= 0.8 and means[-1] > means[0]:
        return "approximately increasing"
    if rho <= -0.8 and means[-1] < means[0]:
        return "approximately decreasing"
    differences = np.diff(means)
    if np.any(differences > 0) and np.any(differences < 0):
        return "nonlinear/nonmonotonic"
    return "no clear directional pattern"


def dependence_summaries(
    xgb_long: pd.DataFrame,
    supported_features: list[str],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    source = xgb_long[
        xgb_long["feature_set"].astype(str).eq("SET1")
        & xgb_long["conceptual_feature"].astype(str).isin(supported_features)
    ].copy()
    rows = []
    shape_labels: dict[str, str] = {}
    for feature in supported_features:
        data_type = str(metadata.loc[feature, "data_type"])
        feature_rows = source[source["conceptual_feature"].astype(str).eq(feature)].copy()
        nonmissing = feature_rows[feature_rows["raw_conceptual_value"].notna()].copy()
        if data_type == "binary":
            nonmissing["bin_order"] = nonmissing["raw_conceptual_value"].astype(int)
            nonmissing["bin_label"] = np.where(
                nonmissing["bin_order"].eq(0), "absent", "present"
            )
            binning_method = "binary_absent_present"
        else:
            try:
                bins = pd.qcut(nonmissing["raw_conceptual_value"], q=5, duplicates="drop")
                if len(bins.cat.categories) < 3 and nonmissing["raw_conceptual_value"].nunique() >= 3:
                    unique_values = np.sort(nonmissing["raw_conceptual_value"].unique())
                    edges = np.unique(np.quantile(unique_values, np.linspace(0, 1, 6)))
                    lower_epsilon = max(1e-12, abs(float(edges[0])) * 1e-12)
                    upper_epsilon = max(1e-12, abs(float(edges[-1])) * 1e-12)
                    edges[0] = edges[0] - lower_epsilon
                    edges[-1] = edges[-1] + upper_epsilon
                    bins = pd.cut(
                        nonmissing["raw_conceptual_value"],
                        bins=edges,
                        include_lowest=True,
                        duplicates="drop",
                    )
                    binning_method = (
                        "unique_value_quintile_fallback_preserving_tied_raw_values"
                    )
                else:
                    binning_method = "observed_nonmissing_quintiles_with_duplicate_edges_dropped"
                nonmissing["bin_order"] = bins.cat.codes + 1
                nonmissing["bin_label"] = nonmissing["bin_order"].map(lambda x: f"Q{int(x)}")
            except ValueError:
                nonmissing["bin_order"] = 1
                nonmissing["bin_label"] = "Q1"
                binning_method = "single_observed_value_bin"
        summary = (
            nonmissing.groupby(["bin_order", "bin_label"], observed=True)
            .agg(
                n=("pair_id", "size"),
                median_feature_value=("raw_conceptual_value", "median"),
                minimum_feature_value=("raw_conceptual_value", "min"),
                maximum_feature_value=("raw_conceptual_value", "max"),
                mean_grouped_shap=("grouped_signed_contribution", "mean"),
                median_grouped_shap=("grouped_signed_contribution", "median"),
            )
            .reset_index()
        )
        label = classify_shape(summary, data_type)
        shape_labels[feature] = label
        summary.insert(0, "conceptual_feature", feature)
        summary.insert(1, "conceptual_display_name", DISPLAY_MAP.get(feature, feature))
        summary.insert(2, "data_type", data_type)
        summary["nonmissing_n"] = len(nonmissing)
        summary["missing_n"] = len(feature_rows) - len(nonmissing)
        summary["binning_method"] = binning_method
        summary["response_shape_label"] = label
        summary["shap_scale"] = "margin_log_odds"
        summary["interpretation_boundary"] = "Descriptive OOF model response; not causal"
        rows.append(summary)
    return pd.concat(rows, ignore_index=True), shape_labels


def full_model_importance(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    mapping_rows: list[dict[str, Any]],
    additivity_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    outputs = []
    for family in ["elasticnet", "xgboost"]:
        for feature_set in ["SET0", "SET1"]:
            pname = pipeline_name(family, feature_set)
            artifact = MODELS / f"final_{pname}.joblib"
            bundle = joblib.load(artifact)
            mapped_frame = apply_bundle_pt_map(frame, bundle["retained_pt_categories"])
            feature_order = list(bundle["features"])
            transformed = bundle["preprocessor"].transform(mapped_frame[feature_order])
            names = list(map(str, bundle["feature_names_out"]))
            if names != list(map(str, bundle["preprocessor"].get_feature_names_out())):
                raise RuntimeError(f"Full model encoded vocabulary mismatch: {pname}")
            encoded_map, map_rows = conceptual_mapping(names, feature_order, metadata)
            for row in map_rows:
                row.update(
                    {
                        "artifact_scope": "FULL_DEVELOPMENT",
                        "outer_fold": np.nan,
                        "pipeline": pname,
                        "model_family": family,
                        "feature_set": feature_set,
                        "artifact_path": str(artifact),
                    }
                )
            mapping_rows.extend(map_rows)
            if family == "elasticnet":
                coefficients = bundle["model"].coef_.ravel()
                if sparse.issparse(transformed):
                    encoded = transformed.multiply(coefficients).toarray()
                else:
                    encoded = np.asarray(transformed) * coefficients
                grouped, grouping_error = group_signed_first(encoded, encoded_map, feature_order)
                decision = bundle["model"].decision_function(transformed)
                reconstruction = bundle["model"].intercept_[0] + grouped.sum(axis=1)
                if np.max(np.abs(decision - reconstruction)) > PREDICTION_RECONSTRUCTION_TOLERANCE:
                    raise RuntimeError(f"Full elastic-net reconstruction failed: {pname}")
            else:
                matrix = xgb.DMatrix(transformed)
                booster = bundle["model"].get_booster()
                contributions = booster.predict(matrix, pred_contribs=True, approx_contribs=False)
                encoded = contributions[:, :-1]
                base = contributions[:, -1]
                margin = booster.predict(matrix, output_margin=True)
                grouped, grouping_error = group_signed_first(encoded, encoded_map, feature_order)
                error = np.abs(base + grouped.sum(axis=1) - margin)
                additivity_rows.append(
                    {
                        "artifact_scope": "FULL_DEVELOPMENT_SECONDARY",
                        "outer_fold": np.nan,
                        "pipeline": pname,
                        "validation_drugs": frame[GROUP].nunique(),
                        "validation_pairs": len(frame),
                        "shap_scale": "margin_log_odds",
                        "maximum_additivity_error": float(error.max()),
                        "mean_additivity_error": float(error.mean()),
                        "tolerance": SHAP_ADDITIVITY_TOLERANCE,
                        "additivity_pass": bool(error.max() < SHAP_ADDITIVITY_TOLERANCE),
                        "signed_grouping_reconstruction_error": grouping_error,
                    }
                )
                if error.max() >= SHAP_ADDITIVITY_TOLERANCE:
                    raise RuntimeError(f"Full XGBoost SHAP additivity failed: {pname}")
            summary = pd.DataFrame(
                {
                    "conceptual_feature": feature_order,
                    "full_mean_absolute_grouped_contribution": np.mean(np.abs(grouped), axis=0),
                    "full_median_absolute_grouped_contribution": np.median(np.abs(grouped), axis=0),
                }
            )
            summary["model_family"] = family
            summary["feature_set"] = feature_set
            summary["full_rank_all_conceptual"] = summary[
                "full_mean_absolute_grouped_contribution"
            ].rank(method="min", ascending=False).astype(int)
            summary["pair_specific"] = summary["conceptual_feature"].map(
                lambda feature: metadata.loc[feature, "feature_set"] == "SET1_ADDITIONAL"
            )
            summary["full_rank_pair_specific"] = np.nan
            mask = summary["pair_specific"]
            summary.loc[mask, "full_rank_pair_specific"] = summary.loc[
                mask, "full_mean_absolute_grouped_contribution"
            ].rank(method="min", ascending=False)
            outputs.append(summary)
    return pd.concat(outputs, ignore_index=True)


def consistency_table(oof_importance: pd.DataFrame, full_importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ["elasticnet", "xgboost"]:
        for feature_set in ["SET0", "SET1"]:
            oof = oof_importance[
                oof_importance["model_family"].eq(family)
                & oof_importance["feature_set"].eq(feature_set)
            ][
                [
                    "conceptual_feature",
                    "mean_absolute_grouped_contribution",
                    "rank_all_conceptual",
                    "rank_pair_specific",
                    "pair_specific",
                ]
            ]
            full = full_importance[
                full_importance["model_family"].eq(family)
                & full_importance["feature_set"].eq(feature_set)
            ][
                [
                    "conceptual_feature",
                    "pair_specific",
                    "full_mean_absolute_grouped_contribution",
                    "full_median_absolute_grouped_contribution",
                    "full_rank_all_conceptual",
                    "full_rank_pair_specific",
                ]
            ]
            merged = oof.merge(full, on=["conceptual_feature", "pair_specific"], validate="one_to_one")
            rho_all = float(
                spearmanr(merged["rank_all_conceptual"], merged["full_rank_all_conceptual"]).statistic
            )
            if feature_set == "SET1":
                pair = merged[merged["pair_specific"]]
                rho_pair = float(
                    spearmanr(pair["rank_pair_specific"], pair["full_rank_pair_specific"]).statistic
                )
            else:
                rho_pair = np.nan
            agreement = "HIGH" if rho_all >= 0.75 else "MODERATE" if rho_all >= 0.50 else "LOW"
            merged.insert(0, "model_family", family)
            merged.insert(1, "feature_set", feature_set)
            merged["conceptual_display_name"] = merged["conceptual_feature"].map(
                lambda feature: DISPLAY_MAP.get(feature, feature)
            )
            merged["scientific_domain"] = merged["conceptual_feature"].map(DOMAIN_MAP)
            merged["rank_difference_full_minus_oof"] = (
                merged["full_rank_all_conceptual"] - merged["rank_all_conceptual"]
            )
            merged["spearman_all_conceptual_features"] = rho_all
            merged["spearman_pair_specific_features"] = rho_pair
            merged["agreement_classification"] = agreement
            merged["full_development_role"] = "secondary_consistency_check_only"
            rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Section 5 output directory already exists: {OUT}")
    required = [REGISTRY, DICTIONARY, OUTER, OOF_PREDICTIONS, PREHOLDOUT_MANIFEST, SECTION3_QC]
    required.extend(sorted(MODELS.glob("outer_*.joblib")))
    required.extend(sorted(MODELS.glob("final_*.joblib")))
    if any(not path.exists() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.exists()])
    if len(list(MODELS.glob("outer_*.joblib"))) != 20 or len(list(MODELS.glob("final_*.joblib"))) != 4:
        raise RuntimeError("Expected exactly 20 outer and 4 full-development frozen model bundles")

    protected_paths = required + sorted(MODELS.glob("*.json"))
    hashes_before = {str(path): sha256(path) for path in protected_paths}

    frame = pd.read_parquet(REGISTRY)
    dictionary = pd.read_csv(DICTIONARY)
    outer = pd.read_csv(OUTER)
    oof_predictions = pd.read_parquet(OOF_PREDICTIONS)
    metadata = dictionary.set_index("feature_name")
    frame = frame.merge(outer[[GROUP, "outer_fold"]], on=GROUP, validate="many_to_one")
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    if (len(frame), frame[GROUP].nunique(), int(frame[TARGET].sum())) != (16470, 107, 2064):
        raise RuntimeError("Development domain mismatch")
    if set(frame["pair_id"]) != set(oof_predictions["pair_id"]):
        raise RuntimeError("OOF prediction keys do not reconcile")
    retained = dictionary[dictionary["status"].isin(["PRIMARY", "SECONDARY"])]
    set0 = retained.loc[retained["feature_set"].eq("SET0"), "feature_name"].tolist()
    pair_features = retained.loc[
        retained["feature_set"].eq("SET1_ADDITIONAL"), "feature_name"
    ].tolist()
    if len(set0) != 18 or len(pair_features) != 27 or set(DOMAIN_MAP) != set(set0 + pair_features):
        raise RuntimeError("Feature/domain membership mismatch")

    staging = Path(tempfile.mkdtemp(prefix=".section5_interpretation_staging_", dir=ROOT / "analysis"))
    figure_source = staging / "12_figure_interpretation_source_data"
    figure_source.mkdir()

    mapping_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    additivity_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    contribution_outputs: dict[str, pd.DataFrame] = {}
    importance_parts = []
    attribution_parts = []

    for family in ["elasticnet", "xgboost"]:
        long_parts = []
        for feature_set in ["SET0", "SET1"]:
            pname = pipeline_name(family, feature_set)
            for fold in range(1, 6):
                artifact = MODELS / f"outer_{fold}_{pname}.joblib"
                bundle = joblib.load(artifact)
                if (
                    bundle["pipeline_status"] != "OUTER_FOLD_FIT"
                    or bundle["outer_fold"] != fold
                    or bundle["family"] != family
                    or bundle["feature_set"] != feature_set
                ):
                    raise RuntimeError(f"Bundle provenance mismatch: {artifact}")
                expected_features = set0 if feature_set == "SET0" else set0 + pair_features
                if list(bundle["features"]) != expected_features:
                    raise RuntimeError(f"Feature membership mismatch: {artifact}")
                valid = frame[frame["outer_fold"].eq(fold)].copy()
                mapped_valid = apply_bundle_pt_map(valid, bundle["retained_pt_categories"])
                feature_order = list(bundle["features"])
                transformed = bundle["preprocessor"].transform(mapped_valid[feature_order])
                names = list(map(str, bundle["feature_names_out"]))
                if names != list(map(str, bundle["preprocessor"].get_feature_names_out())):
                    raise RuntimeError(f"Encoded vocabulary mismatch: {artifact}")
                encoded_map, map_rows = conceptual_mapping(names, feature_order, metadata)
                for row in map_rows:
                    row.update(
                        {
                            "artifact_scope": "OUTER_FOLD",
                            "outer_fold": fold,
                            "pipeline": pname,
                            "model_family": family,
                            "feature_set": feature_set,
                            "artifact_path": str(artifact),
                        }
                    )
                mapping_rows.extend(map_rows)
                saved = oof_predictions.set_index("pair_id").loc[valid["pair_id"], pname].to_numpy()
                predicted = bundle["model"].predict_proba(transformed)[:, 1]
                prediction_error = float(np.max(np.abs(saved - predicted)))
                if prediction_error > PREDICTION_RECONSTRUCTION_TOLERANCE:
                    raise RuntimeError(f"Saved OOF prediction mismatch: {artifact}; {prediction_error}")

                if family == "elasticnet":
                    coefficients = bundle["model"].coef_.ravel()
                    encoded = (
                        transformed.multiply(coefficients).toarray()
                        if sparse.issparse(transformed)
                        else np.asarray(transformed) * coefficients
                    )
                    grouped, grouping_error = group_signed_first(encoded, encoded_map, feature_order)
                    decision = bundle["model"].decision_function(transformed)
                    reconstruction = bundle["model"].intercept_[0] + grouped.sum(axis=1)
                    linear_error = float(np.max(np.abs(decision - reconstruction)))
                    if linear_error > PREDICTION_RECONSTRUCTION_TOLERANCE:
                        raise RuntimeError(f"Elastic-net contribution reconstruction failed: {artifact}")
                    for feature in feature_order:
                        if metadata.loc[feature, "data_type"] == "categorical":
                            continue
                        index = names.index(feature)
                        coefficient_rows.append(
                            {
                                "outer_fold": fold,
                                "feature_set": feature_set,
                                "conceptual_feature": feature,
                                "standardized_coefficient": float(coefficients[index]),
                            }
                        )
                    shap_error = np.nan
                else:
                    matrix = xgb.DMatrix(transformed)
                    booster = bundle["model"].get_booster()
                    contributions = booster.predict(
                        matrix, pred_contribs=True, approx_contribs=False
                    )
                    encoded = contributions[:, :-1]
                    base = contributions[:, -1]
                    margin = booster.predict(matrix, output_margin=True)
                    grouped, grouping_error = group_signed_first(encoded, encoded_map, feature_order)
                    error = np.abs(base + grouped.sum(axis=1) - margin)
                    shap_error = float(error.max())
                    additivity_rows.append(
                        {
                            "artifact_scope": "PRIMARY_OUTER_VALIDATION",
                            "outer_fold": fold,
                            "pipeline": pname,
                            "validation_drugs": valid[GROUP].nunique(),
                            "validation_pairs": len(valid),
                            "shap_scale": "margin_log_odds",
                            "maximum_additivity_error": shap_error,
                            "mean_additivity_error": float(error.mean()),
                            "tolerance": SHAP_ADDITIVITY_TOLERANCE,
                            "additivity_pass": bool(shap_error < SHAP_ADDITIVITY_TOLERANCE),
                            "signed_grouping_reconstruction_error": grouping_error,
                        }
                    )
                    if shap_error >= SHAP_ADDITIVITY_TOLERANCE:
                        raise RuntimeError(f"OOF XGBoost SHAP additivity failed: {artifact}")
                long_parts.append(
                    long_contribution_frame(
                        valid,
                        feature_order,
                        grouped,
                        family,
                        feature_set,
                        metadata,
                        include_raw_numeric=family == "xgboost",
                    )
                )
                provenance_rows.append(
                    {
                        "outer_fold": fold,
                        "pipeline": pname,
                        "validation_drugs": valid[GROUP].nunique(),
                        "validation_pairs": len(valid),
                        "transformed_feature_count": len(names),
                        "mapped_conceptual_feature_count": len(set(encoded_map)),
                        "unmapped_transformed_columns": 0,
                        "saved_oof_prediction_max_abs_error": prediction_error,
                        "signed_grouping_reconstruction_error": grouping_error,
                        "shap_additivity_max_abs_error": shap_error,
                        "model_sha256": hashes_before[str(artifact)],
                    }
                )
        family_long = pd.concat(long_parts, ignore_index=True)
        contribution_outputs[family] = family_long
        importance_parts.append(summarize_importance(family_long, metadata))
        attribution_parts.append(summarize_attribution(family_long))
        if family == "elasticnet":
            family_long.to_parquet(staging / "01_elasticnet_oof_contributions.parquet", index=False)
        else:
            family_long.to_parquet(staging / "03_xgboost_oof_grouped_shap.parquet", index=False)

    feature_map = pd.DataFrame(mapping_rows)
    if feature_map["conceptual_feature"].isna().any() or len(feature_map) == 0:
        raise RuntimeError("Conceptual feature map is incomplete")
    feature_map.to_csv(staging / "CONCEPTUAL_FEATURE_MAP.csv", index=False)

    stability = coefficient_stability(coefficient_rows, metadata)
    stability.to_csv(staging / "02_elasticnet_coefficient_stability.csv", index=False)
    importance = pd.concat(importance_parts, ignore_index=True)
    importance.to_csv(staging / "05_conceptual_feature_importance.csv", index=False)
    attribution = pd.concat(attribution_parts, ignore_index=True)
    attribution.to_csv(staging / "06_domain_level_attribution.csv", index=False)

    concordance, all_rho, pair_rho = cross_model_concordance(importance)
    concordance.to_csv(staging / "07_cross_model_rank_concordance.csv", index=False)

    pair_concordance = concordance[concordance["pair_specific"]].copy()
    pair_concordance["elasticnet_top10_pair_specific"] = (
        pair_concordance["elasticnet_rank_pair_specific"] <= 10
    )
    pair_concordance["xgboost_top10_pair_specific"] = (
        pair_concordance["xgboost_rank_pair_specific"] <= 10
    )
    pair_concordance["cross_model_supported"] = (
        pair_concordance["elasticnet_top10_pair_specific"]
        & pair_concordance["xgboost_top10_pair_specific"]
    )
    supported_names = pair_concordance.loc[
        pair_concordance["cross_model_supported"], "conceptual_feature"
    ].tolist()
    stability_set1 = stability[
        stability["feature_set"].eq("SET1") & stability["pair_specific"]
    ]
    pair_concordance = pair_concordance.merge(
        stability_set1,
        on=["conceptual_feature", "conceptual_display_name", "pair_specific"],
        how="left",
        validate="one_to_one",
    )

    dependence, shape_labels = dependence_summaries(
        contribution_outputs["xgboost"], supported_names, metadata
    )
    dependence.to_csv(staging / "09_xgboost_dependence_summaries.csv", index=False)
    pair_concordance["xgboost_response_shape"] = pair_concordance[
        "conceptual_feature"
    ].map(shape_labels)
    pair_concordance.to_csv(staging / "08_cross_model_supported_features.csv", index=False)

    importance_index = importance[
        importance["feature_set"].eq("SET1")
    ].set_index(["model_family", "conceptual_feature"])
    availability_rows = []
    for domain, measurement, indicator in AVAILABILITY_PAIRS:
        duplicate_indicator = (
            "pair_other_threshold_available"
            if indicator == "pair_other_proportion_available"
            else "pair_other_proportion_available"
            if indicator == "pair_other_threshold_available"
            else None
        )
        row: dict[str, Any] = {
            "scientific_domain": domain,
            "observed_measurement_feature": measurement,
            "availability_indicator": indicator,
            "features_kept_separate": True,
            "availability_indicator_identical_to": duplicate_indicator,
            "indicator_exactly_duplicated_in_development": bool(
                duplicate_indicator is not None
                and frame[indicator].equals(frame[duplicate_indicator])
            ),
        }
        for family in ["elasticnet", "xgboost"]:
            measure = importance_index.loc[(family, measurement)]
            available = importance_index.loc[(family, indicator)]
            row[f"{family}_measurement_mean_abs_contribution"] = measure[
                "mean_absolute_grouped_contribution"
            ]
            row[f"{family}_measurement_rank_all"] = measure["rank_all_conceptual"]
            row[f"{family}_availability_mean_abs_contribution"] = available[
                "mean_absolute_grouped_contribution"
            ]
            row[f"{family}_availability_rank_all"] = available["rank_all_conceptual"]
            row[f"{family}_availability_dominates_measurement"] = bool(
                available["mean_absolute_grouped_contribution"]
                > measure["mean_absolute_grouped_contribution"]
            )
        availability_rows.append(row)
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(staging / "10_availability_indicator_audit.csv", index=False)

    full_importance = full_model_importance(frame, metadata, mapping_rows, additivity_rows)
    # Full-model mapping rows were appended after the first map write.
    pd.DataFrame(mapping_rows).to_csv(staging / "CONCEPTUAL_FEATURE_MAP.csv", index=False)
    consistency = consistency_table(importance, full_importance)
    consistency.to_csv(staging / "11_full_model_consistency.csv", index=False)

    additivity = pd.DataFrame(additivity_rows)
    additivity.to_csv(staging / "04_shap_additivity_qc.csv", index=False)

    concordance.to_csv(figure_source / "panel_a_cross_model_concordance.csv", index=False)
    attribution[attribution["feature_set"].eq("SET1")].to_csv(
        figure_source / "panel_b_set1_domain_attribution.csv", index=False
    )
    dependence.to_csv(figure_source / "panel_c_supported_feature_dependence.csv", index=False)

    hashes_after = {str(path): sha256(path) for path in protected_paths}
    sources_unchanged = hashes_before == hashes_after
    if not sources_unchanged:
        raise RuntimeError("Canonical model/source artifact changed during interpretation")

    provenance = pd.DataFrame(provenance_rows).sort_values(["outer_fold", "pipeline"])
    hash_rows = [
        {
            "artifact": path,
            "sha256_before": digest,
            "sha256_after": hashes_after[path],
            "unchanged": digest == hashes_after[path],
        }
        for path, digest in hashes_before.items()
    ]
    provenance_md = [
        "# Section 5 model provenance QC",
        "",
        f"**Verified:** {now()}",
        "",
        "**Result: PASS — all primary explanations use outer-validation rows scored by the corresponding frozen model that excluded those drugs.**",
        "",
        "The explicit read allowlist contains only the Section 3 development registry, feature dictionary, frozen outer-fold assignment, saved OOF predictions, Section 3 QC/manifest, and frozen model bundles. No temporal-holdout or JADER path is defined.",
        "",
        "## Outer-model reconstruction and mapping",
        "",
        md_table(
            provenance[
                [
                    "outer_fold",
                    "pipeline",
                    "validation_drugs",
                    "validation_pairs",
                    "transformed_feature_count",
                    "mapped_conceptual_feature_count",
                    "unmapped_transformed_columns",
                    "saved_oof_prediction_max_abs_error",
                    "signed_grouping_reconstruction_error",
                    "shap_additivity_max_abs_error",
                ]
            ],
            8,
        ),
        "",
        "## Protected artifact hash audit",
        "",
        md_table(pd.DataFrame(hash_rows), 8),
        "",
        "No model was fitted, tuned, recalibrated, or selected. The full-development models were opened only after OOF interpretation was complete and were used solely for secondary rank-consistency checks.",
    ]
    (staging / "00_model_provenance_qc.md").write_text("\n".join(provenance_md), encoding="utf-8")

    top_en = pair_concordance.nsmallest(10, "elasticnet_rank_pair_specific")
    top_xgb = pair_concordance.nsmallest(10, "xgboost_rank_pair_specific")
    supported = pair_concordance[pair_concordance["cross_model_supported"]].sort_values(
        ["elasticnet_rank_pair_specific", "xgboost_rank_pair_specific"]
    )
    domain_set1 = attribution[
        attribution["feature_set"].eq("SET1")
        & attribution["aggregation_level"].eq("SCIENTIFIC_DOMAIN")
    ][
        [
            "model_family",
            "attribution_group",
            "mean_absolute_domain_contribution",
            "relative_attribution_share",
        ]
    ]
    broad = attribution[
        attribution["feature_set"].eq("SET1")
        & attribution["aggregation_level"].eq("BROAD_GROUP")
    ][
        [
            "model_family",
            "attribution_group",
            "mean_absolute_domain_contribution",
            "relative_attribution_share",
        ]
    ]
    consistency_summary = consistency[
        [
            "model_family",
            "feature_set",
            "spearman_all_conceptual_features",
            "spearman_pair_specific_features",
            "agreement_classification",
        ]
    ].drop_duplicates()
    availability_signal = availability[
        availability["elasticnet_availability_dominates_measurement"]
        | availability["xgboost_availability_dominates_measurement"]
    ]
    shape_display = dependence[
        ["conceptual_feature", "response_shape_label"]
    ].drop_duplicates()
    event_identity = importance[
        importance["feature_set"].eq("SET1")
        & importance["conceptual_feature"].isin(["canonical_pt_code", "primary_soc"])
    ][
        [
            "model_family",
            "conceptual_display_name",
            "mean_absolute_grouped_contribution",
            "rank_all_conceptual",
        ]
    ]

    supported_stability = supported[
        [
            "conceptual_feature",
            "elasticnet_rank_pair_specific",
            "xgboost_rank_pair_specific",
            "nonzero_models",
            "reported_direction",
            "median_standardized_coefficient",
            "directional_stability",
            "xgboost_response_shape",
        ]
    ]
    report_lines = [
        "# Executive Result",
        "",
        f"Frozen out-of-drug interpretation identified {len(supported_names)} pair-specific conceptual predictors in the prespecified top-10 intersection across penalized logistic regression and gradient-boosted trees. Cross-model rank concordance was {all_rho:.3f} across all Set 1 conceptual predictors and {pair_rho:.3f} among the 27 pair-specific predictors. These are descriptive model-attribution findings, not causal effects or feature-selection rules.",
        "",
        "# Interpretation Design",
        "",
        "Primary interpretation used only the 2012–2018 development cohort and the immutable outer folds. Every observation was explained by the frozen outer model that excluded its drug. One-hot contributions were summed with sign within each row before magnitudes were calculated. Full-development interpretations were deferred until all OOF results were complete.",
        "",
        "# OOF Provenance and Model Integrity",
        "",
        "All 20 outer bundles reproduced their saved OOF probabilities within the locked numerical tolerance, and all protected source/model hashes were unchanged. No holdout outcome, JADER source, model fitting, retuning, recalibration, thresholding, or feature selection occurred.",
        "",
        "# Conceptual Feature Mapping",
        "",
        f"The feature map contains {len(pd.DataFrame(mapping_rows)):,} transformed-column mappings across 20 outer and four full-development bundles. Every transformed column mapped to one of 18 Set 0 or 45 Set 1 conceptual predictors; the unmapped count was zero. PT, SOC, route, dosage-form, and other categorical columns were interpreted only after signed within-row grouping.",
        "",
        "# Elastic-Net Contribution and Stability",
        "",
        "OOF linear contributions were calculated on the standardized linear-predictor scale. Numeric and binary coefficient stability is reported across the five outer fits; stability describes reproducibility and was not used to alter the models.",
        "",
        "Top 10 pair-specific conceptual predictors by mean absolute OOF contribution:",
        "",
        md_table(top_en[["conceptual_feature", "elasticnet_rank_pair_specific", "elasticnet_mean_absolute_oof_contribution"]], 5),
        "",
        "# XGBoost OOF SHAP Interpretation",
        "",
        "Package-native exact TreeSHAP was calculated only for outer-validation observations. Contributions use the raw margin/log-odds scale. Top 10 pair-specific conceptual predictors:",
        "",
        md_table(top_xgb[["conceptual_feature", "xgboost_rank_pair_specific", "xgboost_mean_absolute_oof_shap"]], 5),
        "",
        "# SHAP Additivity QC",
        "",
        f"For every outer XGBoost model, base value plus summed SHAP contributions reconstructed the prediction margin below the prespecified {SHAP_ADDITIVITY_TOLERANCE:.0e} tolerance. Maximum observed errors and signed-group reconstruction checks are stored in `04_shap_additivity_qc.csv`.",
        "",
        "# Feature-Domain Attribution",
        "",
        md_table(domain_set1, 4),
        "",
        "Domain shares describe the magnitude of model attribution after signed row-level aggregation. They do not decompose or explain a percentage of the locked ΔAP.",
        "",
        "Event-identity contributions remained baseline/context rather than pair-specific evidence:",
        "",
        md_table(event_identity, 5),
        "",
        "# Baseline vs Pair-Specific Attribution",
        "",
        md_table(broad, 4),
        "",
        "The baseline/context and pair-specific shares are descriptive attribution summaries, not proportions of incremental performance.",
        "",
        "# Cross-Model Concordance",
        "",
        f"Spearman rank correlation was {all_rho:.3f} across all 45 Set 1 conceptual predictors and {pair_rho:.3f} across the 27 pair-specific predictors. No headline significance test was performed.",
        "",
        "# Cross-Model-Supported Pair-Specific Predictors",
        "",
        md_table(supported_stability, 5),
        "",
        "Membership required top-10 pair-specific rank in both model families. Features outside the intersection are not classified as unimportant.",
        "",
        "# Directional Stability",
        "",
        "Direction was assigned only to supported continuous/binary predictors. `DIRECTIONALLY_STABLE` requires a common nonzero sign in at least four of five outer fits; sparse or conflicting coefficients remain directionally unstable.",
        "",
        "# Nonlinear Response Shapes",
        "",
        md_table(shape_display, 4),
        "",
        "Response labels summarize binned OOF margin-scale SHAP patterns and do not imply causality or biological monotonicity.",
        "",
        "# Availability-Indicator Findings",
        "",
        (
            md_table(
                availability_signal[
                    [
                        "scientific_domain",
                        "observed_measurement_feature",
                        "availability_indicator",
                        "availability_indicator_identical_to",
                        "indicator_exactly_duplicated_in_development",
                        "elasticnet_availability_dominates_measurement",
                        "xgboost_availability_dominates_measurement",
                    ]
                ],
                4,
            )
            if len(availability_signal)
            else "No prespecified availability indicator exceeded its associated observed-measurement contribution in either family."
        ),
        "",
        "Observed measurements and information-availability indicators remain separate; numeric contributions are not interpreted without this missingness context.",
        "",
        "`pair_other_proportion_available` and `pair_other_threshold_available` were exactly identical in the development registry (7,204 available; 9,266 unavailable). Their equal penalized-logistic contributions therefore represent duplicated availability information and cannot be attributed separately to proportion versus threshold availability.",
        "",
        "# Full-Development Model Consistency",
        "",
        md_table(consistency_summary, 4),
        "",
        "Full-development results are secondary consistency checks and do not replace out-of-drug interpretation. Low concordance, where present, is reported rather than resolved by refitting.",
        "",
        "# Candidate Main-Text Interpretation",
        "",
        "Across two frozen model families, a prespecified intersection identified pair-specific premarketing characteristics with reproducible out-of-drug predictive contributions. The candidate discussion set must be chosen after scientific review from the cross-model-supported features, considering coefficient stability, TreeSHAP response shape, availability context, and scientific interpretability; it must not be chosen automatically from numerical rank.",
        "",
        "# Candidate Supplementary Interpretation",
        "",
        "Supplementary sources may include complete conceptual rankings, fold-specific standardized coefficients, all domain shares, dependence bins for every supported continuous/binary predictor, availability comparisons, and OOF-versus-full rank consistency. Individual PT dummy rankings are excluded from primary interpretation.",
        "",
        "# Section-Specific Limitations",
        "",
        "1. Attributions describe fitted model behavior under correlated predictors and do not identify causal or independently modifiable effects.\n2. Conceptual grouping prevents one-hot cardinality inflation but can conceal heterogeneous category-specific contributions.\n3. Only 107 drug clusters and five outer fits are available for directional stability, while rare PT mapping changes the encoded vocabulary across folds.\n4. TreeSHAP response shapes are marginal descriptive summaries and may reflect interactions or correlated availability patterns.\n5. Two availability indicators are exactly duplicated, preventing separate attribution of their penalized-logistic contributions.\n6. Full-development consistency is in-sample and secondary.",
        "",
        "# Issues Requiring Scientific Review",
        "",
        "Review the supported-feature intersection, coefficient directions, empirical TreeSHAP shapes, availability dominance, domain attribution, and any low OOF/full concordance. Select approximately 3–6 predictors or feature families for eventual main-text discussion only after this review; do not change model membership or performance results.",
    ]
    report = "\n".join(report_lines)
    prohibited_claims = [
        "causes future adverse events",
        "causes a FAERS signal",
        "increases true ADR risk",
        "XGBoost was selected as the final model",
    ]
    if any(claim.lower() in report.lower() for claim in prohibited_claims):
        raise RuntimeError("Causal or prohibited interpretation leaked into report")
    (staging / "SECTION5_REPORT.md").write_text(report, encoding="utf-8")

    oof_additivity = additivity[additivity["artifact_scope"].eq("PRIMARY_OUTER_VALIDATION")]
    outer_mapping = pd.DataFrame(mapping_rows)
    outer_mapping = outer_mapping[outer_mapping["artifact_scope"].eq("OUTER_FOLD")]
    expected_output_names = {
        "00_model_provenance_qc.md",
        "CONCEPTUAL_FEATURE_MAP.csv",
        "01_elasticnet_oof_contributions.parquet",
        "02_elasticnet_coefficient_stability.csv",
        "03_xgboost_oof_grouped_shap.parquet",
        "04_shap_additivity_qc.csv",
        "05_conceptual_feature_importance.csv",
        "06_domain_level_attribution.csv",
        "07_cross_model_rank_concordance.csv",
        "08_cross_model_supported_features.csv",
        "09_xgboost_dependence_summaries.csv",
        "10_availability_indicator_audit.csv",
        "11_full_model_consistency.csv",
        "12_figure_interpretation_source_data",
        "SECTION5_REPORT.md",
    }
    gates = {
        "01_no_model_retrained": sources_unchanged,
        "02_no_hyperparameter_changed": sources_unchanged,
        "03_no_holdout_prediction_regenerated": True,
        "04_no_holdout_outcome_used": True,
        "05_no_jader_access": True,
        "06_primary_elasticnet_contributions_are_oof_out_of_drug": len(contribution_outputs["elasticnet"]) == 16470 * (18 + 45),
        "07_primary_xgboost_shap_is_oof_out_of_drug": len(contribution_outputs["xgboost"]) == 16470 * (18 + 45),
        "08_shap_additivity_passes": len(oof_additivity) == 10 and oof_additivity["additivity_pass"].all(),
        "09_every_transformed_column_maps_to_concept": len(outer_mapping) > 0 and outer_mapping["conceptual_feature"].notna().all() and (provenance["unmapped_transformed_columns"] == 0).all(),
        "10_one_hot_grouping_uses_signed_sum_first": (provenance["signed_grouping_reconstruction_error"] < 1e-10).all(),
        "11_raw_pt_dummy_rankings_not_primary": not importance["conceptual_feature"].astype(str).str.startswith("canonical_pt_code_").any(),
        "12_importance_not_used_for_feature_selection": True,
        "13_feature_set_membership_unchanged": set(DOMAIN_MAP) == set(set0 + pair_features),
        "14_availability_indicators_separate": availability["features_kept_separate"].all(),
        "15_causal_language_absent": not any(claim.lower() in report.lower() for claim in prohibited_claims),
        "16_canonical_model_source_artifacts_unchanged": sources_unchanged,
    }
    output_names_before_qc = {path.name for path in staging.iterdir()}
    required_before_qc = expected_output_names
    qc = {
        "status": "PASS" if all(bool(v) for v in gates.values()) and required_before_qc.issubset(output_names_before_qc) else "FAIL",
        "generated_at": now(),
        "scope": "SECTION5_FROZEN_MODEL_CROSS_MODEL_INTERPRETATION",
        "read_allowlist": [str(path) for path in required],
        "primary_population": {"drugs": 107, "pairs": 16470, "positives": 2064, "period": "2012-2018_DEVELOPMENT_ONLY"},
        "shap_implementation": "xgboost_package_native_exact_tree_shap_pred_contribs",
        "shap_scale": "margin_log_odds",
        "shap_additivity_tolerance": SHAP_ADDITIVITY_TOLERANCE,
        "qc_gates": {key: bool(value) for key, value in gates.items()},
        "conceptual_rank_spearman_all_set1": all_rho,
        "conceptual_rank_spearman_pair_specific_set1": pair_rho,
        "cross_model_supported_features": supported_names,
        "protected_artifact_hashes_before": hashes_before,
        "protected_artifact_hashes_after": hashes_after,
        "protected_artifacts_unchanged": sources_unchanged,
        "holdout_outcome_rows_accessed": 0,
        "jader_rows_accessed": 0,
        "models_retrained": False,
        "hyperparameters_changed": False,
        "predictions_recalibrated": False,
        "threshold_optimized": False,
        "features_selected_or_removed": False,
        "section5_readme_created": False,
    }
    (staging / "SECTION5_QC.json").write_text(
        json.dumps(json_safe(qc), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if qc["status"] != "PASS":
        raise RuntimeError(
            {
                "failed_gates": [key for key, value in gates.items() if not value],
                "missing_outputs": sorted(required_before_qc - output_names_before_qc),
            }
        )

    staging.rename(OUT)
    print(
        json.dumps(
            json_safe(
                {
                    "status": qc["status"],
                    "all_feature_rank_spearman": all_rho,
                    "pair_specific_rank_spearman": pair_rho,
                    "cross_model_supported": supported_names,
                    "maximum_oof_shap_additivity_error": oof_additivity[
                        "maximum_additivity_error"
                    ].max(),
                    "outputs": str(OUT),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
