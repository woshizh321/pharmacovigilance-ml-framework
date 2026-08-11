#!/usr/bin/env python3
"""Command 10: create and verify the immutable pre-holdout lock package.

This program never discovers, opens, or scores temporal-holdout/JADER data. Its
input allowlist is restricted to completed Section 3 development artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
S3 = ROOT / "analysis" / "section3_model"
TRAIN = S3 / "training"
README_PATH = S3 / "SECTION3_README.md"
MANIFEST_PATH = S3 / "PREHOLDOUT_LOCK_MANIFEST.json"
MANIFEST_HASH_PATH = S3 / "PREHOLDOUT_LOCK_MANIFEST.sha256"

# Explicit development-only source allowlist. Do not add Section 4 sources.
FEATURE_DICTIONARY = S3 / "FEATURE_DICTIONARY_v1.csv"
FINAL_SPEC = TRAIN / "FINAL_MODEL_SPEC_v1.json"
ROOT_FINAL_SPEC = S3 / "FINAL_MODEL_SPEC_v1.json"
S3B_QC = TRAIN / "SECTION3B_QC.json"
S3B_REPORT = TRAIN / "SECTION3B_REPORT.md"
OUTER_FOLDS = S3 / "OUTER_FOLD_ASSIGNMENT_v1.csv"
OOF_PREDICTIONS = TRAIN / "05_oof_predictions.parquet"
PERFORMANCE = TRAIN / "06_oof_performance.csv"
FOLDWISE = TRAIN / "07_foldwise_performance.csv"
INCREMENTAL = TRAIN / "08_incremental_value.csv"
BOOTSTRAP = TRAIN / "09_bootstrap_performance_ci.csv"
CALIBRATION_SOURCE = TRAIN / "10_calibration_source_data.csv"
ELASTICNET_COEFFICIENTS = TRAIN / "11_elasticnet_outer_coefficients.csv"
FINAL_TUNING = TRAIN / "12_final_full_development_tuning.csv"
XGBOOST_GAIN = TRAIN / "13_xgboost_outer_gain.csv"
RARE_QC = TRAIN / "01_outer_fold_pt_rare_mapping_qc.csv"

LOCKED_SOURCE_PATHS = [
    FEATURE_DICTIONARY, FINAL_SPEC, ROOT_FINAL_SPEC, S3B_QC, S3B_REPORT, OUTER_FOLDS,
    OOF_PREDICTIONS, PERFORMANCE, FOLDWISE, INCREMENTAL, BOOTSTRAP, CALIBRATION_SOURCE,
    ELASTICNET_COEFFICIENTS, FINAL_TUNING, XGBOOST_GAIN, RARE_QC,
]

PIPELINE_ORDER = ["elasticnet_set0", "elasticnet_set1", "xgboost_set0", "xgboost_set1"]
DISPLAY = {
    "elasticnet_set0": "Elastic-net Set 0",
    "elasticnet_set1": "Elastic-net Set 1",
    "xgboost_set0": "XGBoost Set 0",
    "xgboost_set1": "XGBoost Set 1",
}
LOCKED_PERFORMANCE = {
    "elasticnet_set0": {"ap": 0.2396, "lift": 1.912, "brier": 0.1033, "auroc": 0.7263, "log_loss": 0.3443, "calibration_intercept": -0.0777, "calibration_slope": 0.9567},
    "elasticnet_set1": {"ap": 0.3257, "lift": 2.599, "brier": 0.0976, "auroc": 0.7609, "log_loss": 0.3291, "calibration_intercept": -0.1075, "calibration_slope": 1.0911},
    "xgboost_set0": {"ap": 0.2596, "lift": 2.071, "brier": 0.1027, "auroc": 0.7256, "log_loss": 0.3440, "calibration_intercept": -0.1816, "calibration_slope": 0.9375},
    # Command 10 explicitly approves 0.7921 as the locked reporting value.
    "xgboost_set1": {"ap": 0.3429, "lift": 2.736, "brier": 0.0966, "auroc": 0.7921, "log_loss": 0.3174, "calibration_intercept": -0.1311, "calibration_slope": 0.8839},
}
LOCKED_INCREMENTAL = {
    ("elasticnet", "delta_average_precision_set1_minus_set0"): {"estimate": 0.08604, "ci_low": 0.04626, "ci_high": 0.11908},
    ("elasticnet", "brier_improvement_set0_minus_set1"): {"estimate": 0.00567, "ci_low": 0.00211, "ci_high": 0.01034},
    ("xgboost", "delta_average_precision_set1_minus_set0"): {"estimate": 0.08333, "ci_low": 0.05364, "ci_high": 0.11633},
    ("xgboost", "brier_improvement_set0_minus_set1"): {"estimate": 0.00612, "ci_low": 0.00244, "ci_high": 0.01030},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def preprocessing_checksum(preprocessor: Any) -> str:
    """Hash the exact fitted preprocessing object embedded in the model bundle."""
    return hashlib.sha256(pickle.dumps(preprocessor, protocol=5)).hexdigest()


def pt_mapper_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule": "retain canonical PT only when present in >=5 distinct development drugs",
        "support_unit": "distinct canonical active moieties",
        "minimum_support": 5,
        "retained_pt_categories": sorted(str(x) for x in bundle["retained_pt_categories"]),
        "rare_label": "PT_RARE",
        "unseen_mapping": "PT_RARE",
    }


def load_sources() -> dict[str, Any]:
    for path in LOCKED_SOURCE_PATHS:
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        "spec": json.loads(FINAL_SPEC.read_text(encoding="utf-8")),
        "qc": json.loads(S3B_QC.read_text(encoding="utf-8")),
        "performance": pd.read_csv(PERFORMANCE),
        "foldwise": pd.read_csv(FOLDWISE),
        "incremental": pd.read_csv(INCREMENTAL),
        "bootstrap": pd.read_csv(BOOTSTRAP),
        "rare": pd.read_csv(RARE_QC),
    }


def validate_locked_science(data: dict[str, Any]) -> None:
    spec, qc = data["spec"], data["qc"]
    domain = spec["development_domain"]
    assert qc["status"] == "PASS"
    assert sha256_file(FINAL_SPEC) == sha256_file(ROOT_FINAL_SPEC)
    assert (domain["drugs"], domain["pairs"], domain["positives"]) == (107, 16470, 2064)
    assert round(domain["prevalence"], 6) == 0.125319
    assert list(spec["pipelines"]) == PIPELINE_ORDER

    perf = data["performance"].set_index("pipeline")
    for name, expected in LOCKED_PERFORMANCE.items():
        observed = perf.loc[name]
        pairs = [
            (observed["average_precision"], expected["ap"], 0.00011),
            (observed["auprc_lift"], expected["lift"], 0.00051),
            (observed["brier"], expected["brier"], 0.00011),
            (observed["auroc"], expected["auroc"], 0.00011),
            (observed["log_loss"], expected["log_loss"], 0.00011),
            (observed["calibration_intercept"], expected["calibration_intercept"], 0.00011),
            (observed["calibration_slope"], expected["calibration_slope"], 0.00011),
        ]
        assert all(abs(float(a) - float(b)) <= tolerance for a, b, tolerance in pairs), (name, pairs)
    assert round(float(perf["theoretical_no_skill_ap"].iloc[0]), 6) == 0.125319

    inc = data["incremental"].set_index(["model_family", "metric"])
    for key, expected in LOCKED_INCREMENTAL.items():
        row = inc.loc[key]
        assert all(abs(float(row[x]) - expected[x]) <= 0.000011 for x in ["estimate", "ci_low", "ci_high"]), (key, row, expected)

    rare = data["rare"].set_index("outer_fold")
    expected_rare = {
        1: (680, 54.38, 18.73, 35.65), 2: (677, 53.35, 16.36, 36.99),
        3: (715, 35.61, 7.78, 27.83), 4: (715, 32.35, 8.95, 23.40),
        5: (728, 28.29, 5.35, 22.94),
    }
    for fold, expected in expected_rare.items():
        row = rare.loc[fold]
        observed = (
            int(row["retained_pt_categories"]), round(row["validation_pt_rare_pct"], 2),
            round(row["validation_unseen_pt_pct"], 2), round(row["validation_low_support_pt_pct"], 2),
        )
        assert observed == expected, (fold, observed, expected)

    expected_params = {
        "elasticnet_set0": {"C": 0.3, "l1_ratio": 0.0},
        "elasticnet_set1": {"C": 0.01, "l1_ratio": 0.5},
        "xgboost_set0": {"subsample": 0.85, "reg_lambda": 5.0, "reg_alpha": 1.0, "n_estimators": 800, "min_child_weight": 5, "max_depth": 5, "learning_rate": 0.02, "gamma": 0.1, "colsample_bytree": 0.5},
        "xgboost_set1": {"subsample": 0.85, "reg_lambda": 5.0, "reg_alpha": 1.0, "n_estimators": 800, "min_child_weight": 5, "max_depth": 5, "learning_rate": 0.02, "gamma": 0.1, "colsample_bytree": 0.5},
    }
    for name in PIPELINE_ORDER:
        assert spec["pipelines"][name]["hyperparameters"] == expected_params[name]
        assert spec["pipelines"][name]["final_retained_pt_category_count"] == 884


def build_pipeline_locks(data: dict[str, Any]) -> dict[str, Any]:
    spec = data["spec"]
    feature_hash = sha256_file(FEATURE_DICTIONARY)
    locks = {}
    for name in PIPELINE_ORDER:
        model_spec = spec["pipelines"][name]
        model_path = Path(model_spec["artifact_path"])
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        actual_hash = sha256_file(model_path)
        if actual_hash != model_spec["sha256"]:
            raise RuntimeError(f"Model hash mismatch: {name}")
        bundle = joblib.load(model_path)
        if bundle["pipeline_status"] != "FROZEN_FULL_DEVELOPMENT":
            raise RuntimeError(f"Unexpected pipeline status: {name}")
        if bundle["hyperparameters"] != model_spec["hyperparameters"]:
            raise RuntimeError(f"Bundle/spec parameter mismatch: {name}")
        if len(bundle["retained_pt_categories"]) != 884:
            raise RuntimeError(f"PT mapper count mismatch: {name}")
        mapper = pt_mapper_payload(bundle)
        prep_hash = preprocessing_checksum(bundle["preprocessor"])
        native = model_spec.get("xgboost_native_artifact_path")
        native_lock = None
        if native:
            native_path = Path(native)
            native_hash = sha256_file(native_path)
            if native_hash != model_spec["xgboost_native_sha256"]:
                raise RuntimeError(f"Native XGBoost hash mismatch: {name}")
            native_lock = {"path": str(native_path), "sha256": native_hash, "size_bytes": native_path.stat().st_size}
        locks[name] = {
            "display_name": DISPLAY[name],
            "status": "IMMUTABLE_PREHOLDOUT_FREEZE",
            "model_path": str(model_path),
            "model_sha256": actual_hash,
            "model_size_bytes": model_path.stat().st_size,
            "xgboost_native_model": native_lock,
            "feature_set_identifier": model_spec["feature_set"],
            "conceptual_feature_count": model_spec["conceptual_feature_count"],
            "feature_dictionary_path": str(FEATURE_DICTIONARY),
            "feature_dictionary_sha256": feature_hash,
            "pt_mapper": {
                "checksum_method": "SHA256 of canonical sorted JSON payload",
                "sha256": sha256_json(mapper),
                "retained_individual_pt_categories": 884,
                "payload": mapper,
            },
            "preprocessing": {
                "checksum_method": "SHA256 of pickle protocol 5 serialization of fitted preprocessor embedded in frozen joblib model bundle",
                "sha256": prep_hash,
                "encoded_feature_count": len(bundle["feature_names_out"]),
                "specification": model_spec["preprocessing"],
            },
            "package_versions": model_spec["software_versions"],
            "random_seeds": model_spec["random_seeds"],
            "final_hyperparameters": model_spec["hyperparameters"],
            "native_probabilities_frozen": True,
            "recalibration": None,
            "training_drug_count": model_spec["training_drugs"],
            "training_pair_count": model_spec["training_pairs"],
            "training_positive_count": model_spec["training_positives"],
        }
    return locks


def build_readme(data: dict[str, Any], locks: dict[str, Any]) -> str:
    perf = data["performance"].set_index("pipeline")
    boot = data["bootstrap"].set_index(["pipeline", "metric"])
    inc = data["incremental"].set_index(["model_family", "metric"])
    rare = data["rare"].set_index("outer_fold")
    lines = [
        "# Section Purpose", "",
        "Section 3 estimates development-only predictive performance and the incremental value of pair-specific premarketing clinical-trial safety information. Section 3 is scientifically **PASS** and development modelling is **CLOSED**. All four pipelines proceed unchanged to temporal validation; no model family is designated the winner. Temporal outcomes were not scored or opened for this lock.", "",
        "# Development Domain", "",
        "The locked development domain contains 107 canonical active moieties, 16,470 active-moiety–MedDRA PT pairs, and 2,064 Criterion-R positives (prevalence 12.5319%). Two zero-positive drugs remain included. The theoretical no-skill AP reference is 0.125319.", "",
        "# Frozen Feature Architecture", "",
        "Feature Set 0 contains the 18 frozen regulatory, event-identity, and general trial-programme features. Feature Set 1 contains exactly Set 0 plus 27 pair-specific premarketing-safety features (45 conceptual features total). Exact drug identity remains a grouping key, never a predictor. The approximate AE safety-population feature remains qualified and must not be described as exact cumulative exposure.", "",
        "# PT Identity and Rare-Category Rule", "",
        "Canonical PT identity is categorical. In every training partition, an individual PT is retained only when it occurs in at least five distinct training active moieties; lower-support and validation-unseen PTs map to `PT_RARE`. The full-development mapper retains exactly 884 individual PT categories. This target-free rule is immutable.", "",
        "For temporal validation, `PT_SUPPORTED` means that the holdout PT belongs to these 884 retained development categories. `PT_RARE` means that the holdout PT was absent from development or occurred in fewer than five development drugs. This is a preregistered transportability diagnostic, not a new primary endpoint; no stratum-specific model, retuning, or recalibration is permitted.", "",
        "# Nested Drug-Grouped Validation", "",
        "Five immutable outer folds grouped by active moiety generated one out-of-drug OOF probability per pair and pipeline. Within each outer-training partition, the same deterministic four-fold StratifiedGroupKFold assignments were used for both feature sets and model families. Fold-local fitting covered PT mapping, one-hot vocabularies, imputation, transformations, standardization where applicable, hyperparameter selection, and model fitting. Mean inner-CV log loss was the only tuning objective.", "",
        "# Elastic-Net Results", "",
        "| Pipeline | AP | AP 95% CI | Lift | Brier | AUROC | Log loss | Calibration intercept | Calibration slope |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["elasticnet_set0", "elasticnet_set1"]:
        r, b = LOCKED_PERFORMANCE[name], boot.loc[(name, "average_precision")]
        lines.append(f"| {DISPLAY[name]} | {r['ap']:.4f} | {b.ci_low:.4f}–{b.ci_high:.4f} | {r['lift']:.3f} | {r['brier']:.4f} | {r['auroc']:.4f} | {r['log_loss']:.4f} | {r['calibration_intercept']:.4f} | {r['calibration_slope']:.4f} |")
    lines += ["", "# XGBoost Results", "", "| Pipeline | AP | AP 95% CI | Lift | Brier | AUROC | Log loss | Calibration intercept | Calibration slope |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["xgboost_set0", "xgboost_set1"]:
        r, b = LOCKED_PERFORMANCE[name], boot.loc[(name, "average_precision")]
        lines.append(f"| {DISPLAY[name]} | {r['ap']:.4f} | {b.ci_low:.4f}–{b.ci_high:.4f} | {r['lift']:.3f} | {r['brier']:.4f} | {r['auroc']:.4f} | {r['log_loss']:.4f} | {r['calibration_intercept']:.4f} | {r['calibration_slope']:.4f} |")
    lines += ["", "XGBoost is not designated the final or winning model; all four pipelines remain co-primary for temporal validation.", "",
        "# Incremental Value of Pair-Specific Premarketing Safety Information", "",
        "Pair-specific premarketing clinical-trial safety information provided incremental predictive value beyond regulatory, event-identity, and general premarketing trial-program context in drug-grouped nested validation.", "",
        "| Family | ΔAP, Set 1−Set 0 (95% CI) | Brier improvement, Set 0−Set 1 (95% CI) | Log-loss improvement, Set 0−Set 1 (95% CI) |\n|---|---:|---:|---:|"]
    for family, label in [("elasticnet", "Elastic-net"), ("xgboost", "XGBoost")]:
        ap = LOCKED_INCREMENTAL[(family, "delta_average_precision_set1_minus_set0")]
        br = LOCKED_INCREMENTAL[(family, "brier_improvement_set0_minus_set1")]
        ll = inc.loc[(family, "log_loss_improvement_set0_minus_set1")]
        lines.append(f"| {label} | {ap['estimate']:.5f} ({ap['ci_low']:.5f}–{ap['ci_high']:.5f}) | {br['estimate']:.5f} ({br['ci_low']:.5f}–{br['ci_high']:.5f}) | {ll.estimate:.5f} ({ll.ci_low:.5f}–{ll.ci_high:.5f}) |")
    lines += ["", "# Calibration", "",
        "Native OOF probabilities were evaluated without Platt, isotonic, beta, or intercept-only recalibration. Probability clipping was used only for numerical logit calculations. Calibration will be evaluated—not repaired—in temporal validation.", "",
        "# Fold-Level Stability", "",
        "Development outer-fold AP ranged from 0.204–0.297 for Elastic-net Set 0, 0.327–0.398 for Elastic-net Set 1, 0.210–0.342 for XGBoost Set 0, and 0.300–0.465 for XGBoost Set 1. Performance therefore varied across held-out drug groups, particularly for nonlinear Set 1 modelling. No fold or drug may be removed.", "",
        "# Rare-PT Generalizability Findings", "",
        "| Fold | Retained PT categories | PT_RARE % | Unseen % | Low-support % |\n|---:|---:|---:|---:|---:|"]
    for fold in range(1, 6):
        r = rare.loc[fold]
        lines.append(f"| {fold} | {int(r.retained_pt_categories)} | {r.validation_pt_rare_pct:.2f} | {r.validation_unseen_pt_pct:.2f} | {r.validation_low_support_pt_pct:.2f} |")
    lines += ["", "The 28.29%–54.38% PT_RARE burden is a locked development generalizability limitation.", "",
        "# Final Full-Development Pipelines", "",
        "| Pipeline | Final hyperparameters | Encoded features | Model SHA256 |\n|---|---|---:|---|"]
    for name in PIPELINE_ORDER:
        lock = locks[name]
        params = json.dumps(lock["final_hyperparameters"], sort_keys=True, separators=(",", ":"))
        lines.append(f"| {DISPLAY[name]} | `{params}` | {lock['preprocessing']['encoded_feature_count']} | `{lock['model_sha256']}` |")
    lines += ["", "All preprocessing, the 884-category development PT mapper, vocabularies, imputation/scaling parameters, hyperparameters, coefficients/trees, native probability outputs, software versions, and random seeds are frozen in the model artefacts and pre-holdout manifest.", "",
        "# Holdout Firewall", "",
        "Section 3 used development data only. No 2019–2022 outcome, holdout ROR/IC, exact holdout FAERS volume, holdout coverage, holdout PT identity list, holdout prediction, or JADER outcome was opened. This command did not score the temporal holdout. Section 4 can begin only after the lock manifest verifies and the PI provides separate scientific authorization.", "",
        "# Figure 3 Source Specification", "",
        "No final artwork is generated here. Provisional Panel A uses nested OOF precision–recall curves for Set 0 versus Set 1, separated by family if four curves are visually confusing. Panel B shows paired drug-bootstrap ΔAP and 95% CI for both families. Panel C may show Brier improvement and/or calibration curves; final layout is deferred to visualization review. Sources: `training/05_oof_predictions.parquet`, `training/08_incremental_value.csv`, `training/09_bootstrap_performance_ci.csv`, and `training/10_calibration_source_data.csv`.", "",
        "# Table 2 Source Specification", "",
        "Candidate title: **Drug-grouped development performance of baseline/context and pair-specific premarketing safety models**. The four model rows report AP, drug-bootstrap AP CI, AUPRC lift, AUROC, Brier, log loss, calibration intercept, and calibration slope. A separate paired block reports ΔAP, Brier improvement, log-loss improvement, and their drug-cluster bootstrap CIs. Authoritative sources are `training/06_oof_performance.csv`, `training/08_incremental_value.csv`, and `training/09_bootstrap_performance_ci.csv`.", "",
        "# Candidate Main-Text Results", "",
        "In drug-grouped nested development validation, adding pair-specific premarketing clinical-trial safety information improved AP by 0.08604 (95% drug-bootstrap CI 0.04626–0.11908) for elastic-net and 0.08333 (0.05364–0.11633) for XGBoost. Corresponding Brier improvements were 0.00567 (0.00211–0.01034) and 0.00612 (0.00244–0.01030). These results establish incremental development predictive value, not causality, clinical utility, or temporal generalizability.", "",
        "# Candidate Supplementary Results", "",
        "Supplementary reporting may include the complete tuning grids, inner-fold QC, outer-fold performance, PT_RARE transfer diagnostics, 10-bin calibration source data, bootstrap intervals, outer elastic-net coefficient-presence artefacts, and native XGBoost split/gain metadata. These are descriptive or stability artefacts and must not be used for post-hoc model modification.", "",
        "# Section-Specific Limitations", "",
        "1. The effective sample size is 107 active moieties despite 16,470 pair-level observations.\n2. PT identity is sparse across drugs; 28.29%–54.38% of outer-validation pairs map to PT_RARE.\n3. Outer-fold AP shows meaningful variability across held-out drug groups.\n4. Structurally missing pair-level trial features require fold-local imputation and prespecified availability indicators.\n5. Approximate AE safety-population size is not exact exposure.", "",
        "# Files and Provenance", "",
        "- Development registry: `development_pair_registry.parquet`\n- Feature lock: `FEATURE_DICTIONARY_v1.csv`\n- Outer folds: `OUTER_FOLD_ASSIGNMENT_v1.csv`\n- OOF results: `training/05_oof_predictions.parquet` through `training/10_calibration_source_data.csv`\n- Stability artefacts: `training/11_elasticnet_outer_coefficients.csv` and `training/13_xgboost_outer_gain.csv`\n- Final tuning: `training/12_final_full_development_tuning.csv`\n- Final specification: `training/FINAL_MODEL_SPEC_v1.json`\n- Section 3B QC: `training/SECTION3B_QC.json`\n- Pre-holdout manifest: `PREHOLDOUT_LOCK_MANIFEST.json`\n- Reproducible lock generator/verifier: `scripts/s03c_preholdout_lock.py`.", "",
        "# Locked Numbers", "",
        "The authoritative locked numbers are the four-row performance table, the paired incremental-value table, the five-fold rare-PT table, the 107/16,470/2,064 domain, prevalence 0.125319, the 884-category final mapper, the final hyperparameters, and the checksums recorded above and in `PREHOLDOUT_LOCK_MANIFEST.json`. More precise machine-readable values remain in the cited CSV/JSON files; rounded values in this README are the approved reporting values.", "",
        "# Prohibited Interpretations", "",
        "Do not claim that XGBoost was selected as the final model; that predictions represent causal ADR risk; that AP establishes clinical utility; or that internal validation proves temporal generalizability. Do not retune, alter features or eligibility, modify PT mapping, recalibrate, select a probability threshold, remove folds/drugs, or change model families before temporal validation.", "",
    ]
    return "\n".join(lines)


def create_lock() -> None:
    if MANIFEST_PATH.exists() or MANIFEST_HASH_PATH.exists():
        raise FileExistsError("Pre-holdout manifest already exists; immutable lock will not be overwritten")
    data = load_sources()
    validate_locked_science(data)
    locks = build_pipeline_locks(data)
    README_PATH.write_text(build_readme(data, locks), encoding="utf-8")
    section_sources = {}
    for path in [*LOCKED_SOURCE_PATHS, README_PATH]:
        section_sources[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "manifest": "PREHOLDOUT_LOCK_MANIFEST",
        "version": 1,
        "status": "IMMUTABLE_PREHOLDOUT_LOCK_VERIFIED",
        "created_at": datetime.now().astimezone().isoformat(),
        "section3_status": "PASS_DEVELOPMENT_MODELLING_CLOSED",
        "temporal_holdout_scored": False,
        "jader_outcome_accessed": False,
        "development_domain": data["spec"]["development_domain"],
        "scientific_interpretation_lock": "Pair-specific premarketing clinical-trial safety information provided incremental predictive value beyond regulatory, event-identity, and general premarketing trial-program context in drug-grouped nested validation.",
        "native_probability_lock": {"frozen": True, "recalibration": None, "prohibited": ["Platt", "isotonic", "beta", "intercept-only recalibration"]},
        "pipelines": locks,
        "section4_preregistration": {
            "status": "PREREGISTERED_BEFORE_OUTCOME_OPENING",
            "PT_SUPPORTED": "holdout PT is one of the 884 individually retained full-development PT categories",
            "PT_RARE": "holdout PT is absent from full development or present in fewer than five distinct development drugs and maps to PT_RARE",
            "report_per_stratum_after_outcome_opening": ["pair_count", "positive_prevalence", "average_precision", "AUROC", "Brier", "calibration summaries where stable"],
            "role": "transportability diagnostic; not a new primary endpoint",
            "separate_models_by_stratum": False,
            "retuning": False,
            "recalibration": False,
        },
        "prohibited_interpretations": [
            "XGBoost is the final or winning model", "the model predicts causal ADR risk",
            "AP demonstrates clinical utility", "internal validation proves temporal generalizability",
        ],
        "input_allowlist_and_provenance": section_sources,
        "runtime_used_for_lock": {"python": platform.python_version(), "joblib": joblib.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "release_condition": {
            "FINAL_MODEL_SPEC_v1_exists_and_verified": True,
            "four_frozen_model_artifacts_exist_and_verify": True,
            "PREHOLDOUT_LOCK_MANIFEST_exists": True,
            "SECTION3_README_exists": True,
            "separate_scientific_authorization_for_section4_required": True,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_hash = sha256_file(MANIFEST_PATH)
    MANIFEST_HASH_PATH.write_text(f"{manifest_hash}  {MANIFEST_PATH.name}\n", encoding="utf-8")
    verify_lock()


def verify_lock() -> None:
    if not MANIFEST_PATH.exists() or not README_PATH.exists() or not MANIFEST_HASH_PATH.exists():
        raise FileNotFoundError("Incomplete pre-holdout lock package")
    expected_manifest_hash = MANIFEST_HASH_PATH.read_text(encoding="utf-8").split()[0]
    if sha256_file(MANIFEST_PATH) != expected_manifest_hash:
        raise RuntimeError("Manifest sidecar checksum mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = load_sources()
    validate_locked_science(data)
    current_locks = build_pipeline_locks(data)
    for name in PIPELINE_ORDER:
        frozen = manifest["pipelines"][name]
        current = current_locks[name]
        for field in ["model_sha256", "feature_dictionary_sha256"]:
            if frozen[field] != current[field]:
                raise RuntimeError(f"{name} {field} changed after lock")
        if frozen["pt_mapper"]["sha256"] != current["pt_mapper"]["sha256"]:
            raise RuntimeError(f"{name} PT mapper changed after lock")
        if frozen["preprocessing"]["sha256"] != current["preprocessing"]["sha256"]:
            raise RuntimeError(f"{name} preprocessing changed after lock")
    for path_str, recorded in manifest["input_allowlist_and_provenance"].items():
        path = Path(path_str)
        if not path.exists() or sha256_file(path) != recorded["sha256"]:
            raise RuntimeError(f"Locked source changed: {path}")
    if manifest["temporal_holdout_scored"] or manifest["jader_outcome_accessed"]:
        raise RuntimeError("Holdout/JADER firewall violation in manifest")
    print("PASS: immutable pre-holdout lock package verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify_lock() if args.verify_only else create_lock()
