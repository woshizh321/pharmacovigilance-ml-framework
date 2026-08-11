#!/usr/bin/env python3
"""Independent read-only audit of Command 13 interpretation outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "section5_interpretation"
S3 = ROOT / "analysis" / "section3_model"
GROUP = "canonical_active_moiety"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = {
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
        "SECTION5_QC.json",
    }
    present = {path.name for path in OUT.iterdir()}
    qc = json.loads((OUT / "SECTION5_QC.json").read_text())
    mapping = pd.read_csv(OUT / "CONCEPTUAL_FEATURE_MAP.csv")
    elastic = pd.read_parquet(OUT / "01_elasticnet_oof_contributions.parquet")
    tree = pd.read_parquet(OUT / "03_xgboost_oof_grouped_shap.parquet")
    additivity = pd.read_csv(OUT / "04_shap_additivity_qc.csv")
    importance = pd.read_csv(OUT / "05_conceptual_feature_importance.csv")
    attribution = pd.read_csv(OUT / "06_domain_level_attribution.csv")
    concordance = pd.read_csv(OUT / "07_cross_model_rank_concordance.csv")
    supported = pd.read_csv(OUT / "08_cross_model_supported_features.csv")
    dependence = pd.read_csv(OUT / "09_xgboost_dependence_summaries.csv")
    availability = pd.read_csv(OUT / "10_availability_indicator_audit.csv")
    consistency = pd.read_csv(OUT / "11_full_model_consistency.csv")
    report = (OUT / "SECTION5_REPORT.md").read_text()
    dictionary = pd.read_csv(S3 / "FEATURE_DICTIONARY_v1.csv")
    registry = pd.read_parquet(S3 / "development_pair_registry.parquet")

    expected_rows = 16470 * (18 + 45)
    recalculated_parts = []
    for family, long_frame in [("elasticnet", elastic), ("xgboost", tree)]:
        recalculated = (
            long_frame.groupby(["feature_set", "conceptual_feature"], observed=True)
            .agg(
                mean_abs=("absolute_grouped_contribution", "mean"),
                median_abs=("absolute_grouped_contribution", "median"),
            )
            .reset_index()
        )
        recalculated["model_family"] = family
        recalculated_parts.append(recalculated)
    recalculated = pd.concat(recalculated_parts, ignore_index=True)
    compare = importance.merge(
        recalculated,
        on=["model_family", "feature_set", "conceptual_feature"],
        validate="one_to_one",
    )
    importance_max_error = float(
        np.max(
            np.abs(
                compare["mean_absolute_grouped_contribution"] - compare["mean_abs"]
            )
        )
    )

    feature_rows = concordance.sort_values("elasticnet_rank_all")
    rho_all = float(
        spearmanr(feature_rows.elasticnet_rank_all, feature_rows.xgboost_rank_all).statistic
    )
    pair_rows = feature_rows[feature_rows.pair_specific]
    rho_pair = float(
        spearmanr(
            pair_rows.elasticnet_rank_pair_specific,
            pair_rows.xgboost_rank_pair_specific,
        ).statistic
    )
    expected_supported = set(
        pair_rows.loc[
            pair_rows.elasticnet_rank_pair_specific.le(10)
            & pair_rows.xgboost_rank_pair_specific.le(10),
            "conceptual_feature",
        ]
    )
    actual_supported = set(
        supported.loc[supported.cross_model_supported, "conceptual_feature"]
    )

    domain_errors = []
    for family, long_frame in [("elasticnet", elastic), ("xgboost", tree)]:
        set1 = long_frame[long_frame.feature_set.astype(str).eq("SET1")]
        by_row = (
            set1.groupby(["pair_id", GROUP, "outer_fold", "scientific_domain"], observed=True)[
                "grouped_signed_contribution"
            ]
            .sum()
            .abs()
            .groupby("scientific_domain", observed=True)
            .mean()
        )
        locked = attribution[
            attribution.model_family.eq(family)
            & attribution.feature_set.eq("SET1")
            & attribution.aggregation_level.eq("SCIENTIFIC_DOMAIN")
        ].set_index("attribution_group")["mean_absolute_domain_contribution"]
        domain_errors.extend(abs(by_row.loc[name] - locked.loc[name]) for name in by_row.index)

    full_summary = consistency[
        [
            "model_family",
            "feature_set",
            "spearman_all_conceptual_features",
            "spearman_pair_specific_features",
        ]
    ].drop_duplicates()
    full_correlation_errors = []
    for row in full_summary.itertuples(index=False):
        subset = consistency[
            consistency.model_family.eq(row.model_family)
            & consistency.feature_set.eq(row.feature_set)
        ]
        direct_all = float(
            spearmanr(subset.rank_all_conceptual, subset.full_rank_all_conceptual).statistic
        )
        full_correlation_errors.append(abs(direct_all - row.spearman_all_conceptual_features))
        if row.feature_set == "SET1":
            pair = subset[subset.pair_specific]
            direct_pair = float(
                spearmanr(pair.rank_pair_specific, pair.full_rank_pair_specific).statistic
            )
            full_correlation_errors.append(
                abs(direct_pair - row.spearman_pair_specific_features)
            )

    protected_match = all(
        sha256(Path(path)) == digest
        for path, digest in qc["protected_artifact_hashes_after"].items()
    )
    report_headings = [line[2:] for line in report.splitlines() if line.startswith("# ")]
    expected_headings = [
        "Executive Result",
        "Interpretation Design",
        "OOF Provenance and Model Integrity",
        "Conceptual Feature Mapping",
        "Elastic-Net Contribution and Stability",
        "XGBoost OOF SHAP Interpretation",
        "SHAP Additivity QC",
        "Feature-Domain Attribution",
        "Baseline vs Pair-Specific Attribution",
        "Cross-Model Concordance",
        "Cross-Model-Supported Pair-Specific Predictors",
        "Directional Stability",
        "Nonlinear Response Shapes",
        "Availability-Indicator Findings",
        "Full-Development Model Consistency",
        "Candidate Main-Text Interpretation",
        "Candidate Supplementary Interpretation",
        "Section-Specific Limitations",
        "Issues Requiring Scientific Review",
    ]
    duplicate = registry.pair_other_proportion_available.equals(
        registry.pair_other_threshold_available
    )
    unstable_direction_ok = (
        supported.loc[
            supported.directional_stability.eq("DIRECTIONALLY_UNSTABLE"),
            "reported_direction",
        ]
        .eq("not_assigned")
        .all()
    )
    stable_direction_ok = (
        supported.loc[
            supported.directional_stability.eq("DIRECTIONALLY_STABLE"),
            "reported_direction",
        ]
        .isin(["positive", "negative"])
        .all()
    )
    required_pair_families = set(
        dictionary.loc[dictionary.feature_set.eq("SET1_ADDITIONAL"), "feature_name"]
    )
    set1_importance_features = set(
        importance.loc[
            importance.feature_set.eq("SET1") & importance.pair_specific,
            "conceptual_feature",
        ]
    )
    read_allowlist_development_only = all(
        str(S3) in path for path in qc["read_allowlist"]
    )
    primary_additivity = additivity[
        additivity.artifact_scope.eq("PRIMARY_OUTER_VALIDATION")
    ]
    checks = {
        "required_outputs_present": required.issubset(present),
        "section5_readme_absent": not (OUT / "SECTION5_README.md").exists(),
        "qc_passes_16_of_16": qc["status"] == "PASS"
        and len(qc["qc_gates"]) == 16
        and all(qc["qc_gates"].values()),
        "development_read_allowlist_only": read_allowlist_development_only,
        "protected_hashes_still_match": protected_match,
        "elasticnet_oof_row_count_exact": len(elastic) == expected_rows,
        "xgboost_oof_row_count_exact": len(tree) == expected_rows,
        "outcomes_absent_from_contribution_files": "criterion_r_3y" not in elastic.columns
        and "criterion_r_3y" not in tree.columns,
        "conceptual_map_has_zero_unmapped": mapping.conceptual_feature.notna().all(),
        "one_hot_rows_map_to_parent": mapping.loc[
            mapping.is_one_hot_column, "mapping_rule"
        ].eq("one_hot_prefix_to_parent").all(),
        "direct_importance_recalculation_matches": importance_max_error < 1e-12,
        "direct_domain_recalculation_matches": max(domain_errors) < 1e-12,
        "rank_correlation_all_matches": abs(rho_all - qc["conceptual_rank_spearman_all_set1"]) < 1e-12,
        "rank_correlation_pair_matches": abs(rho_pair - qc["conceptual_rank_spearman_pair_specific_set1"]) < 1e-12,
        "top10_intersection_matches": expected_supported == actual_supported,
        "all_27_pair_features_interpreted": required_pair_families == set1_importance_features,
        "primary_shap_additivity_passes": len(primary_additivity) == 10
        and primary_additivity.additivity_pass.all()
        and primary_additivity.maximum_additivity_error.max() < 1e-5,
        "unstable_elasticnet_direction_not_assigned": unstable_direction_ok,
        "stable_elasticnet_direction_reported": stable_direction_ok,
        "duplicated_availability_indicators_disclosed": duplicate
        and availability.indicator_exactly_duplicated_in_development.any()
        and "exactly identical" in report,
        "full_model_correlations_recalculate": max(full_correlation_errors) < 1e-12,
        "report_headings_exact": report_headings == expected_headings,
        "figure_panel_a_matches": pd.read_csv(
            OUT / "12_figure_interpretation_source_data/panel_a_cross_model_concordance.csv"
        ).equals(concordance),
        "figure_panel_c_matches": pd.read_csv(
            OUT / "12_figure_interpretation_source_data/panel_c_supported_feature_dependence.csv"
        ).equals(dependence),
        "no_holdout_or_jader_rows": qc["holdout_outcome_rows_accessed"] == 0
        and qc["jader_rows_accessed"] == 0,
        "no_model_change_or_selection": qc["models_retrained"] is False
        and qc["hyperparameters_changed"] is False
        and qc["features_selected_or_removed"] is False,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_importance_recalculation_error": importance_max_error,
        "maximum_domain_recalculation_error": max(domain_errors),
        "maximum_full_correlation_recalculation_error": max(full_correlation_errors),
        "maximum_primary_shap_additivity_error": float(
            primary_additivity.maximum_additivity_error.max()
        ),
        "all_feature_spearman": rho_all,
        "pair_specific_spearman": rho_pair,
        "supported_features": sorted(actual_supported),
    }
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    main()
