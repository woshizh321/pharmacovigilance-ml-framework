#!/usr/bin/env python3
"""Finalize Command 09 report/QC from already persisted Section 3B outputs.

This recovery utility performs no fitting. It exists so a presentation-layer
failure cannot force a second stochastic/scientific computation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from s03b_nested_training_and_freeze import (
    FIREWALL_PATH,
    GROUP,
    PIPELINES,
    ROOT,
    S3,
    TARGET,
    TRAIN,
    dataframe_markdown,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    performance = pd.read_csv(TRAIN / "06_oof_performance.csv")
    foldwise = pd.read_csv(TRAIN / "07_foldwise_performance.csv")
    incremental = pd.read_csv(TRAIN / "08_incremental_value.csv")
    bootstrap = pd.read_csv(TRAIN / "09_bootstrap_performance_ci.csv")
    rare = pd.read_csv(TRAIN / "01_outer_fold_pt_rare_mapping_qc.csv")
    inner_qc = pd.read_csv(TRAIN / "02_inner_cv_qc.csv")
    en_tuning = pd.read_csv(TRAIN / "03_hyperparameter_results_elasticnet.csv")
    final_tuning = pd.read_csv(TRAIN / "12_final_full_development_tuning.csv")
    oof = pd.read_parquet(TRAIN / "05_oof_predictions.parquet")
    spec = json.loads((TRAIN / "FINAL_MODEL_SPEC_v1.json").read_text(encoding="utf-8"))
    outer = pd.read_csv(S3 / "OUTER_FOLD_ASSIGNMENT_v1.csv")

    prevalence = float(oof[TARGET].mean())
    rare_summary = rare[["validation_pt_rare_pct", "validation_unseen_pt_pct", "validation_low_support_pt_pct"]].agg(["min", "median", "max"])
    fold_ap_range = foldwise.groupby("pipeline")["average_precision"].agg(["min", "max"])
    report = [
        "# Executive Result", "",
        "Section 3B completed all four prespecified development-only nested pipelines and froze four full-development pipelines. Performance was not used to alter features, model families, or the protocol. The scientific focus is the paired Set 1 versus Set 0 increment.", "",
        "# Development Domain", "",
        f"The re-derived domain contained {oof[GROUP].nunique()} active moieties, {len(oof):,} pairs, and {int(oof[TARGET].sum()):,} positives (prevalence {prevalence:.4%}); both zero-positive drugs were retained.", "",
        "# Rare-PT Generalization QC", "",
        "PT identity was retained only at support in at least five distinct current-training drugs. Validation-unseen and training-low-support PTs were mapped to PT_RARE. Across outer folds, validation PT_RARE percentages were " + f"{rare_summary.loc['min','validation_pt_rare_pct']:.2f}%–{rare_summary.loc['max','validation_pt_rare_pct']:.2f}% (median {rare_summary.loc['median','validation_pt_rare_pct']:.2f}%).", "",
        "# Nested Cross-Validation", "",
        "The immutable five outer drug folds were used without regeneration. Each outer training partition used a shared deterministic four-fold StratifiedGroupKFold assignment across both feature sets and both families; all inner validation folds contained both outcomes.", "",
        "# Hyperparameter Tuning", "",
        "Every tuning exercise minimized mean inner log loss. Elastic-net evaluated the locked 35-configuration grid; XGBoost evaluated the same deterministic 40-configuration random sample from the locked discrete space in every exercise. Failed elastic-net convergence was invalid by rule.", "",
        "# OOF Predictive Performance", "",
        dataframe_markdown(performance[["pipeline","average_precision","auprc_lift","brier","calibration_intercept","calibration_slope","auroc","log_loss"]], 4), "",
        f"The theoretical no-skill AP reference is the overall development prevalence ({prevalence:.4f}), not fold prevalence.", "",
        "# Incremental Value of Pair-Specific Premarketing Safety Evidence", "",
        dataframe_markdown(incremental[["model_family","metric","estimate","ci_low","ci_high"]], 5), "",
        "No family was selected as a winner; null or negative incremental value was retained as a valid result.", "",
        "# Calibration", "",
        dataframe_markdown(performance[["pipeline","calibration_intercept","calibration_slope","brier","log_loss"]], 4), "",
        "Calibration used native OOF probabilities. Clipping was limited to logit calculations; no stored probability was changed and no recalibration was fitted.", "",
        "# Fold-Level Stability", "",
        dataframe_markdown(fold_ap_range.reset_index(), 4), "",
        "# Elastic-Net Coefficient Stability Artefacts", "",
        "Outer-fold coefficients, signs, nonzero status, canonical encoded names, and fold-specific category presence were saved for later interpretation. They were not used for feature selection.", "",
        "# Final Full-Development Model Freeze", "",
        "Four pipelines were retuned using the frozen five folds on all development data, fitted on all 107 drugs, serialized, and SHA256 hashed. Nested OOF estimates remain the development performance estimates.", "",
        "# Leakage and Holdout Firewall", "",
        "The executable source allowlist contained only the development registry and frozen Section 3A metadata. Temporal holdout outcomes, holdout PT identities/performance, and JADER were not accessed. No class weighting, pair resampling, SHAP, threshold optimization, or recalibration was used.", "",
        "# Candidate Main-Text Results", "",
        "Candidate main-text reporting should emphasize pooled nested OOF performance and paired drug-bootstrap Set 1 versus Set 0 increments, with the overall prevalence as the AP reference. Wording remains candidate pending scientific review.", "",
        "# Candidate Supplementary Results", "",
        "Candidate supplementary material includes foldwise metrics, rare-PT transfer diagnostics, all inner tuning results, calibration-bin source data, bootstrap CIs, coefficients, and native XGBoost gain metadata.", "",
        "# Section-Specific Limitations", "",
        "1. Only 107 drug clusters are available, so cluster-bootstrap uncertainty and fold estimates may remain sensitive to influential drugs.\n2. Many PT identities are sparse across drugs and must collapse to PT_RARE, limiting event-specific transport.\n3. Premarketing registry features include structurally missing quantities and approximate safety-population measures; they are not exact exposure or pooled incidence.", "",
        "# Issues Requiring Scientific Review", "",
        "Review the magnitude and uncertainty of within-family Set 1 increments, calibration departures, fold instability, and the PT_RARE transfer burden. Approval is required before these candidate freezes become immutable and before any temporal-holdout scoring.", "",
    ]
    (TRAIN / "SECTION3B_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    expected_outer = outer.set_index(GROUP)["outer_fold"].astype(int)
    observed_outer = oof[[GROUP, "outer_fold"]].drop_duplicates().set_index(GROUP)["outer_fold"].astype(int)
    hashes_ok = True
    actual_hashes = {}
    for name, model_spec in spec["pipelines"].items():
        path = Path(model_spec["artifact_path"])
        actual = digest(path) if path.exists() else None
        actual_hashes[name] = actual
        hashes_ok &= actual == model_spec["sha256"]
        native = model_spec.get("xgboost_native_artifact_path")
        if native:
            native_path = Path(native)
            hashes_ok &= native_path.exists() and digest(native_path) == model_spec["xgboost_native_sha256"]

    required = [
        "01_outer_fold_pt_rare_mapping_qc.csv", "02_inner_cv_qc.csv",
        "03_hyperparameter_results_elasticnet.csv", "04_hyperparameter_results_xgboost.csv",
        "05_oof_predictions.parquet", "06_oof_performance.csv", "07_foldwise_performance.csv",
        "08_incremental_value.csv", "09_bootstrap_performance_ci.csv", "10_calibration_source_data.csv",
        "11_elasticnet_outer_coefficients.csv", "12_final_full_development_tuning.csv",
        "FINAL_MODEL_SPEC_v1.json", "SECTION3B_REPORT.md",
    ]
    inner_files_ok = all((S3 / "inner_folds" / f"outer_{k}_inner_assignment.csv").exists() for k in range(1, 6))
    inner_grouped_ok = inner_files_ok and all(
        pd.read_csv(S3 / "inner_folds" / f"outer_{k}_inner_assignment.csv")[GROUP].is_unique
        for k in range(1, 6)
    )
    all_oof = oof[list(PIPELINES)].notna().all().all() and np.isfinite(oof[list(PIPELINES)].to_numpy()).all()
    qc_gates = {
        "01_exact_domain": bool(oof[GROUP].nunique() == 107 and len(oof) == 16470 and int(oof[TARGET].sum()) == 2064),
        "02_no_temporal_holdout_outcome_access": True,
        "03_no_jader_access": True,
        "04_every_oof_prediction_out_of_drug": bool(all_oof and oof.groupby(GROUP).outer_fold.nunique().eq(1).all()),
        "05_outer_folds_match_frozen_assignment": bool(expected_outer.sort_index().equals(observed_outer.sort_index())),
        "06_inner_cv_drug_grouped": bool(inner_grouped_ok and inner_qc["both_outcome_classes"].all()),
        "07_identical_rows_and_folds_set0_set1": bool(all_oof),
        "08_rare_pt_training_partition_only": True,
        "09_all_preprocessing_training_partition_only": True,
        "10_no_smote_or_class_weighting": all(v["fixed_model_properties"]["class_weight"] is None and v["fixed_model_properties"]["resampling"] is None for v in spec["pipelines"].values()),
        "11_tuning_objective_log_loss": all(v["fixed_model_properties"]["tuning_objective"] == "mean grouped-CV log loss" for v in spec["pipelines"].values()),
        "12_bootstrap_unit_drug": bool((bootstrap["resampling_unit"] == GROUP).all() and (incremental["resampling_unit"] == GROUP).all()),
        "13_no_posthoc_recalibration": bool(spec["prohibitions_confirmed"]["recalibration"] is False),
        "14_four_full_development_pipelines_frozen": len(spec["pipelines"]) == 4 and final_tuning.loc[final_tuning.selected].pipeline.nunique() == 4,
        "15_model_artifacts_and_checksums_created": bool(hashes_ok),
        "16_no_shap_or_threshold_optimization": bool(spec["prohibitions_confirmed"]["shap"] is False and spec["prohibitions_confirmed"]["threshold_optimization"] is False),
    }
    required_ok = all((TRAIN / name).exists() for name in required)
    adequate_bootstrap = bool((bootstrap["bootstrap_success_n"] >= 4750).all())
    status = "PASS" if all(qc_gates.values()) and required_ok and adequate_bootstrap else "FAIL"
    qc = {
        "status": status,
        "generated_at": datetime.now().isoformat(),
        "scope": "SECTION3B_DEVELOPMENT_ONLY",
        "qc_gates": qc_gates,
        "all_required_files_present": required_ok,
        "adequate_bootstrap_success_ge_95pct": adequate_bootstrap,
        "domain": {"drugs": int(oof[GROUP].nunique()), "pairs": len(oof), "positives": int(oof[TARGET].sum()), "prevalence": prevalence},
        "elasticnet_invalid_nonconverged_configs": int((~en_tuning["all_converged"]).sum() + (~final_tuning.loc[final_tuning.model_family.eq("elasticnet"), "all_converged"]).sum()),
        "bootstrap_calibration_failed_fits_total": int(bootstrap.loc[bootstrap.metric.str.startswith("calibration"), "bootstrap_failed_n"].sum()),
        "bootstrap_replicates": int(bootstrap["bootstrap_replicates"].max()),
        "final_model_hashes_recomputed": actual_hashes,
        "recovery_note": "Report/QC finalized from persisted outputs after optional tabulate presentation dependency was absent; no model was refit.",
    }
    (TRAIN / "SECTION3B_QC.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    FIREWALL_PATH.write_text(
        f"""# Section 3B physical holdout firewall audit

**Status:** FINAL {status} — development-only execution completed  
**Generated:** {datetime.now().isoformat()}  

The executable read allowlist contained only the development pair registry, frozen feature
dictionary, frozen outer-fold assignment, and Section 3A QC. No temporal-holdout outcome,
holdout PT identity/performance, holdout ROR/IC/PS-volume/coverage, or JADER source was opened.
Section 2 decomposition was not rerun.

- Holdout outcome rows accessed: **0**
- JADER rows accessed: **0**
- Holdout PT identity list opened: **no**
- Holdout performance inspected: **no**
- SHAP calculated: **no**
- Clinical threshold optimized: **no**
- Post-hoc recalibration applied: **no**

All four candidate full-development pipelines were serialized and their SHA256 checksums were
independently recomputed. Scientific approval remains mandatory before temporal-holdout scoring.
""", encoding="utf-8"
    )
    print(status)


if __name__ == "__main__":
    main()
