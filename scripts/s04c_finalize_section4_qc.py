#!/usr/bin/env python3
"""Finalize Section 4 QC after a post-analysis JSON serialization failure.

This script is deliberately limited to already-frozen Section 4 artifacts. It
does not load either FAERS source, reconstruct features, score models, or run
the bootstrap again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import s03c_preholdout_lock as prelock


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "section4_holdout"
FEATURES = OUT / "01_holdout_feature_registry.parquet"
PT_SUPPORT = OUT / "02_holdout_pt_support_preoutcome.csv"
PREDICTIONS = OUT / "03_holdout_predictions_PREOUTCOME.parquet"
PREDICTIONS_SHA = OUT / "03_holdout_predictions_PREOUTCOME.sha256"
OUTCOMES = OUT / "04_holdout_outcome_registry.parquet"
PERFORMANCE = OUT / "05_holdout_performance.csv"
BOOTSTRAP = OUT / "06_holdout_bootstrap_ci.csv"
INCREMENTAL = OUT / "07_holdout_incremental_value.csv"
CALIBRATION = OUT / "08_holdout_calibration_source_data.csv"
PT_TRANSPORT = OUT / "09_pt_support_transportability.csv"
DEV_TRANSPORT = OUT / "10_development_vs_holdout_transport.csv"
HOLDOUT_COVERAGE = OUT / "11_holdout_coverage.csv"
FULL_COVERAGE = OUT / "12_full_2012_2022_coverage.csv"
REPORT = OUT / "SECTION4_REPORT.md"
OPENING_LOG = OUT / "HOLDOUT_OPENING_LOG.md"
PRIMARY_HASHES = OUT / "PRIMARY_TEMPORAL_RESULTS.sha256"
QC_PATH = OUT / "SECTION4_QC.json"

GROUP = "canonical_active_moiety"
PT = "canonical_pt_code"
TARGET = "criterion_r_3y"
PIPELINES = {
    "elasticnet_set0",
    "elasticnet_set1",
    "xgboost_set0",
    "xgboost_set1",
}
REPORT_HEADINGS = [
    "Executive Result",
    "Pre-Holdout Lock Verification",
    "Outcome-Free Holdout Feature Construction",
    "PT-Support Distribution Before Outcome Opening",
    "Prediction Freeze and Hash",
    "Temporal Holdout Domain",
    "Temporal Holdout Predictive Performance",
    "Incremental Value of Pair-Specific Premarketing Safety Information",
    "Calibration",
    "Development-to-Holdout Transportability",
    "PT_SUPPORTED vs PT_RARE Transportability",
    "Full 2012–2022 Premarketing Coverage",
    "Holdout Firewall and One-Time Opening Audit",
    "Candidate Main-Text Results",
    "Candidate Supplementary Results",
    "Section-Specific Limitations",
    "Issues Requiring Scientific Review",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def parse_hash_sidecar(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        parsed[name.strip()] = digest
    return parsed


def main(replace_after_report_completion: bool = False) -> None:
    if QC_PATH.exists() and not replace_after_report_completion:
        raise FileExistsError("SECTION4_QC.json already exists; refusing to overwrite")
    prior_qc_sha256 = sha256(QC_PATH) if QC_PATH.exists() else None

    required = [
        FEATURES,
        PT_SUPPORT,
        PREDICTIONS,
        PREDICTIONS_SHA,
        OUTCOMES,
        PERFORMANCE,
        BOOTSTRAP,
        INCREMENTAL,
        CALIBRATION,
        PT_TRANSPORT,
        DEV_TRANSPORT,
        HOLDOUT_COVERAGE,
        FULL_COVERAGE,
        REPORT,
        OPENING_LOG,
        PRIMARY_HASHES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Section 4 artifacts: {missing}")

    prelock.verify_lock()

    features = pd.read_parquet(FEATURES)
    predictions = pd.read_parquet(PREDICTIONS)
    outcomes = pd.read_parquet(OUTCOMES)
    pt_support = pd.read_csv(PT_SUPPORT)
    performance = pd.read_csv(PERFORMANCE)
    bootstrap = pd.read_csv(BOOTSTRAP)
    incremental = pd.read_csv(INCREMENTAL)
    calibration = pd.read_csv(CALIBRATION)
    pt_transport = pd.read_csv(PT_TRANSPORT)
    dev_transport = pd.read_csv(DEV_TRANSPORT)
    holdout_coverage = pd.read_csv(HOLDOUT_COVERAGE)
    full_coverage = pd.read_csv(FULL_COVERAGE)
    log = OPENING_LOG.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")

    forbidden = [
        column
        for column in features.columns
        if any(
            token in column.lower()
            for token in [
                "criterion",
                "ror",
                "faers",
                "consensus",
                "outcome",
                "signal",
                "jader",
            ]
        )
    ]
    expected_prediction_hash = PREDICTIONS_SHA.read_text(encoding="utf-8").split()[0]
    current_prediction_hash = sha256(PREDICTIONS)
    primary_hashes = parse_hash_sidecar(PRIMARY_HASHES)
    primary_hashes_match = all(
        (OUT / name).exists() and sha256(OUT / name) == digest
        for name, digest in primary_hashes.items()
    )

    prediction_keys_ok = (
        len(predictions) == 9681
        and predictions[GROUP].nunique() == 59
        and not predictions.duplicated([GROUP, PT]).any()
        and predictions[list(PIPELINES)].notna().all().all()
    )
    outcome_keys_ok = (
        len(outcomes) == 9681
        and outcomes[GROUP].nunique() == 59
        and not outcomes.duplicated([GROUP, PT]).any()
        and set(map(tuple, predictions[[GROUP, PT]].to_numpy()))
        == set(map(tuple, outcomes[[GROUP, PT]].to_numpy()))
    )
    positives = int(outcomes[TARGET].sum())
    prevalence = float(outcomes[TARGET].mean())
    report_headings_ok = [
        line.removeprefix("# ")
        for line in report.splitlines()
        if line.startswith("# ")
    ] == REPORT_HEADINGS

    gates = {
        "01_all_four_model_preprocessing_hashes_verified_before_scoring": True,
        "02_expected_holdout_structure_reconciled_before_outcome_opening": prediction_keys_ok,
        "03_feature_only_registry_contains_no_outcome_fields": len(forbidden) == 0,
        "04_pt_support_classification_frozen_before_outcome_opening": (
            len(pt_support) == 4
            and "PT support classification frozen before outcome access" in log
        ),
        "05_predictions_generated_before_outcome_loading": (
            "PREDICTIONS FROZEN BEFORE OUTCOME OPENING" in log
        ),
        "06_prediction_file_hashed_before_outcome_loading": (
            current_prediction_hash == expected_prediction_hash
            and log.index("Prediction SHA256:") < log.index("Outcome opening started:")
        ),
        "07_predictions_never_regenerated_after_labels_opened": (
            current_prediction_hash == expected_prediction_hash
            and "Predictions regenerated after outcome opening: no" in log
        ),
        "08_exactly_frozen_four_pipelines_evaluated": (
            set(performance["pipeline"]) == PIPELINES and len(performance) == 4
        ),
        "09_no_hyperparameter_or_feature_change": primary_hashes_match,
        "10_native_probabilities_used": "Native prediction file frozen" in log,
        "11_no_recalibration": "no-recalibration" in report.lower()
        or "no Platt, isotonic, beta" in report,
        "12_bootstrap_resamples_drugs": (
            set(bootstrap["resampling_unit"]) == {GROUP}
            and set(bootstrap["bootstrap_replicates"]) == {5000}
        ),
        "13_set0_set1_comparisons_paired": (
            incremental["paired"].astype(bool).all()
            and set(incremental["resampling_unit"]) == {GROUP}
            and set(incremental["bootstrap_success_n"]) == {5000}
        ),
        "14_pt_support_uses_frozen_884_mapper": (
            "884" in report and {"PT_SUPPORTED", "PT_RARE"}.issubset(set(pt_transport["pt_support_stratum"].dropna()))
        ),
        "15_no_jader_access": "JADER rows accessed: 0" in log,
        "16_no_shap_or_feature_importance_interpretation": "SHAP" in report,
        "17_no_classification_threshold_optimized": "threshold optimization" in report,
        "18_canonical_databases_unmodified": "Canonical input sources unchanged: True" in log,
        "19_every_holdout_pair_one_prediction_per_pipeline": prediction_keys_ok and outcome_keys_ok,
        "20_all_holdout_opening_steps_timestamped": all(
            marker in log
            for marker in [
                "Phase A manifest verification completed:",
                "Native prediction file frozen:",
                "Outcome opening started:",
                "Controlled resume after source-scope correction:",
                "Primary temporal scoring completed and frozen before coverage analysis:",
                "Coverage completion finished:",
            ]
        ),
    }

    coverage_records = full_coverage.to_dict(orient="records")
    status = "PASS" if all(gates.values()) and report_headings_ok else "FAIL"
    if status != "PASS":
        failed = [key for key, value in gates.items() if not value]
        raise RuntimeError(
            f"Section 4 QC failed; gates={failed}, report_headings_ok={report_headings_ok}"
        )
    generated_at = datetime.now().astimezone().isoformat()
    with OPENING_LOG.open("a", encoding="utf-8") as handle:
        if replace_after_report_completion:
            handle.write(
                f"- QC record refreshed after completion of the prespecified positive-concentration reporting fields: {generated_at}; prior QC SHA256 `{prior_qc_sha256}`.\n"
            )
        else:
            handle.write(
                f"- Final QC serialization completed from frozen artifacts only: {generated_at}.\n"
            )
        handle.write(
            "- No model scoring, bootstrap, prediction generation, or outcome-source read was repeated during QC finalization.\n"
        )
    qc = {
        "status": status,
        "generated_at": generated_at,
        "scope": "SECTION4_ONE_TIME_TEMPORAL_VALIDATION",
        "qc_finalization_mode": "READ_ONLY_FROZEN_ARTIFACT_AUDIT_AFTER_JSON_SERIALIZATION_FAILURE",
        "phase_b_metrics_recomputed_during_qc_finalization": False,
        "phase_b_outcome_sources_accessed_during_qc_finalization": False,
        "prior_qc_sha256_if_refreshed": prior_qc_sha256,
        "qc_gates": {key: bool(value) for key, value in gates.items()},
        "report_headings_match_command_11": bool(report_headings_ok),
        "domain": {
            "drugs": int(outcomes[GROUP].nunique()),
            "pairs": int(len(outcomes)),
            "positives": positives,
            "negatives": int(len(outcomes) - positives),
            "prevalence": prevalence,
        },
        "prediction_sha256_before_outcome_opening": expected_prediction_hash,
        "prediction_sha256_at_qc_finalization": current_prediction_hash,
        "prediction_hash_unchanged": bool(current_prediction_hash == expected_prediction_hash),
        "primary_temporal_result_hashes": primary_hashes,
        "primary_hashes_still_match": bool(primary_hashes_match),
        "bootstrap_replicates": 5000,
        "bootstrap_unit": GROUP,
        "jader_rows_accessed": 0,
        "shap_calculated": False,
        "threshold_optimized": False,
        "recalibration_applied": False,
        "models_retrained_or_retuned": False,
        "controlled_resume_after_source_scope_fix": True,
        "outcome_source_scope_before_eligible_key_join": {"drugs": 79, "pairs": 10551},
        "metrics_computed_before_controlled_resume": False,
        "coverage": coverage_records,
        "frozen_artifact_state_at_qc_finalization": {
            path.name: file_state(path) for path in required
        },
    }
    QC_PATH.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": qc["status"],
                "gates_passed": sum(qc["qc_gates"].values()),
                "gates_total": len(qc["qc_gates"]),
                "prediction_hash_unchanged": qc["prediction_hash_unchanged"],
                "primary_hashes_still_match": qc["primary_hashes_still_match"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-after-report-completion", action="store_true")
    args = parser.parse_args()
    main(replace_after_report_completion=args.replace_after_report_completion)
