#!/usr/bin/env python3
"""Independent, read-only numerical audit of the completed Section 4 outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "section4_holdout"
GROUP = "canonical_active_moiety"
PT = "canonical_pt_code"
TARGET = "criterion_r_3y"
PIPELINES = [
    "elasticnet_set0",
    "elasticnet_set1",
    "xgboost_set0",
    "xgboost_set1",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    intercept = sm.GLM(
        y,
        np.ones((len(y), 1)),
        family=sm.families.Binomial(),
        offset=logit,
    ).fit(disp=0).params[0]
    slope = sm.GLM(
        y,
        sm.add_constant(logit),
        family=sm.families.Binomial(),
    ).fit(disp=0).params[1]
    return float(intercept), float(slope)


def main() -> None:
    prediction_path = OUT / "03_holdout_predictions_PREOUTCOME.parquet"
    expected_hash = (OUT / "03_holdout_predictions_PREOUTCOME.sha256").read_text().split()[0]
    predictions = pd.read_parquet(prediction_path)
    outcomes = pd.read_parquet(OUT / "04_holdout_outcome_registry.parquet")
    performance = pd.read_csv(OUT / "05_holdout_performance.csv").set_index("pipeline")
    bootstrap = pd.read_csv(OUT / "06_holdout_bootstrap_ci.csv")
    incremental = pd.read_csv(OUT / "07_holdout_incremental_value.csv")
    calibration_bins = pd.read_csv(OUT / "08_holdout_calibration_source_data.csv")
    pt_transport = pd.read_csv(OUT / "09_pt_support_transportability.csv")
    coverage = pd.read_csv(OUT / "12_full_2012_2022_coverage.csv").set_index("scope")
    features = pd.read_parquet(OUT / "01_holdout_feature_registry.parquet")
    qc = json.loads((OUT / "SECTION4_QC.json").read_text())

    joined = predictions.merge(outcomes, on=[GROUP, PT], validate="one_to_one")
    y = joined[TARGET].to_numpy()
    deviations: dict[str, float] = {}
    for pipeline in PIPELINES:
        probability = joined[pipeline].to_numpy()
        intercept, slope = calibration(y, probability)
        recalculated = {
            "average_precision": average_precision_score(y, probability),
            "auroc": roc_auc_score(y, probability),
            "brier": brier_score_loss(y, probability),
            "log_loss": log_loss(y, probability, labels=[0, 1]),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
        for metric, value in recalculated.items():
            deviations[f"{pipeline}:{metric}"] = abs(value - performance.loc[pipeline, metric])

    inc = incremental.set_index(["model_family", "metric"])
    increment_deviations = {}
    for family in ["elasticnet", "xgboost"]:
        set0 = performance.loc[f"{family}_set0"]
        set1 = performance.loc[f"{family}_set1"]
        direct = {
            "delta_average_precision_set1_minus_set0": set1.average_precision - set0.average_precision,
            "brier_improvement_set0_minus_set1": set0.brier - set1.brier,
            "log_loss_improvement_set0_minus_set1": set0.log_loss - set1.log_loss,
        }
        for metric, value in direct.items():
            increment_deviations[f"{family}:{metric}"] = abs(value - inc.loc[(family, metric), "estimate"])

    forbidden = [
        column
        for column in features.columns
        if any(token in column.lower() for token in ["criterion", "ror", "faers", "consensus", "outcome", "signal", "jader"])
    ]
    supported = pt_transport[pt_transport.record_type.eq("PRIMARY_PT_SUPPORT")]
    coverage_arithmetic_ok = all(
        int(row.all_criterion_r_signals)
        == int(row.premarketing_observed) + int(row.postmarketing_only)
        for _, row in coverage.iterrows()
    )
    required_names = {
        "00_prehholdout_manifest_verification.md",
        "01_holdout_feature_registry.parquet",
        "02_holdout_pt_support_preoutcome.csv",
        "03_holdout_predictions_PREOUTCOME.parquet",
        "03_holdout_predictions_PREOUTCOME.sha256",
        "HOLDOUT_OPENING_LOG.md",
        "04_holdout_outcome_registry.parquet",
        "05_holdout_performance.csv",
        "06_holdout_bootstrap_ci.csv",
        "07_holdout_incremental_value.csv",
        "08_holdout_calibration_source_data.csv",
        "09_pt_support_transportability.csv",
        "10_development_vs_holdout_transport.csv",
        "11_holdout_coverage.csv",
        "12_full_2012_2022_coverage.csv",
        "SECTION4_REPORT.md",
        "SECTION4_QC.json",
    }
    checks = {
        "required_outputs_present": required_names.issubset({path.name for path in OUT.iterdir()}),
        "section4_readme_present_and_nonempty": (
            (OUT / "SECTION4_README.md").exists()
            and (OUT / "SECTION4_README.md").stat().st_size > 0
        ),
        "prediction_hash_matches": sha256(prediction_path) == expected_hash,
        "domain_exact": len(joined) == 9681 and joined[GROUP].nunique() == 59 and int(y.sum()) == 1110,
        "keys_unique": not joined.duplicated([GROUP, PT]).any(),
        "feature_registry_outcome_free": len(forbidden) == 0,
        "all_direct_metric_differences_below_1e_8": max(deviations.values()) < 1e-8,
        "all_increment_differences_below_1e_12": max(increment_deviations.values()) < 1e-12,
        "bootstrap_layout_valid": len(bootstrap) == 24 and set(bootstrap.bootstrap_replicates) == {5000} and set(bootstrap.resampling_unit) == {GROUP},
        "bootstrap_all_metrics_5000_successes": set(bootstrap.bootstrap_success_n) == {5000},
        "calibration_has_10_bins_per_pipeline": len(calibration_bins) == 40 and set(calibration_bins.groupby("pipeline").size()) == {10},
        "pt_strata_reconcile": int(supported.drop_duplicates("pt_support_stratum").pairs.sum()) == 9681 and int(supported.drop_duplicates("pt_support_stratum").positives.sum()) == 1110,
        "coverage_arithmetic_valid": coverage_arithmetic_ok,
        "qc_passes_20_of_20": qc["status"] == "PASS" and all(qc["qc_gates"].values()) and len(qc["qc_gates"]) == 20,
        "qc_prediction_hash_unchanged": qc["prediction_hash_unchanged"] is True,
        "qc_primary_hashes_match": qc["primary_hashes_still_match"] is True,
        "qc_zero_jader": qc["jader_rows_accessed"] == 0,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_direct_metric_absolute_difference": max(deviations.values()),
        "maximum_increment_absolute_difference": max(increment_deviations.values()),
    }
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError([name for name, passed in checks.items() if not passed])


if __name__ == "__main__":
    main()
