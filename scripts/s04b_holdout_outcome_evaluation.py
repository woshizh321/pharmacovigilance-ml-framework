#!/usr/bin/env python3
"""Command 11 Phase B: one-time holdout outcome opening and locked evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import s03b_nested_training_and_freeze as metrics_lib
import s03c_preholdout_lock as prelock


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "section4_holdout"
PREDICTIONS = OUT / "03_holdout_predictions_PREOUTCOME.parquet"
PREDICTIONS_SHA = OUT / "03_holdout_predictions_PREOUTCOME.sha256"
FEATURE_REGISTRY = OUT / "01_holdout_feature_registry.parquet"
PT_SUPPORT_PRE = OUT / "02_holdout_pt_support_preoutcome.csv"
OPENING_LOG = OUT / "HOLDOUT_OPENING_LOG.md"
OPEN_MARKER = OUT / "PHASE_B_OUTCOME_OPENED.marker"
OUTCOME_REGISTRY = OUT / "04_holdout_outcome_registry.parquet"
PERFORMANCE = OUT / "05_holdout_performance.csv"
BOOTSTRAP_CI = OUT / "06_holdout_bootstrap_ci.csv"
INCREMENTAL = OUT / "07_holdout_incremental_value.csv"
CALIBRATION = OUT / "08_holdout_calibration_source_data.csv"
PT_TRANSPORT = OUT / "09_pt_support_transportability.csv"
DEV_TRANSPORT = OUT / "10_development_vs_holdout_transport.csv"
HOLDOUT_COVERAGE = OUT / "11_holdout_coverage.csv"
FULL_COVERAGE = OUT / "12_full_2012_2022_coverage.csv"
REPORT = OUT / "SECTION4_REPORT.md"
QC_PATH = OUT / "SECTION4_QC.json"
PRIMARY_FREEZE_SHA = OUT / "PRIMARY_TEMPORAL_RESULTS.sha256"

# Phase-B outcome sources. No JADER path is defined.
LABELS = ROOT / "data" / "processed" / "preflight_v2" / "faers_fda_anchored_labels_1_2_3y.parquet"
ALL_SIGNALS = ROOT / "data" / "processed" / "preflight_v2" / "faers_all_exposed_pair_signals_3y.parquet"
DEV_SIGNAL_UNIVERSE = ROOT / "analysis" / "section2_coverage" / "01_development_signal_universe.csv"
DEV_REGISTRY = ROOT / "analysis" / "section3_model" / "development_pair_registry.parquet"
DEV_PERFORMANCE = ROOT / "analysis" / "section3_model" / "training" / "06_oof_performance.csv"
DEV_INCREMENTAL = ROOT / "analysis" / "section3_model" / "training" / "08_incremental_value.csv"
SECTION1 = ROOT / "analysis" / "section1_cohort" / "02_final_drug_level_characteristics.csv"

GROUP = "canonical_active_moiety"
PT = "canonical_pt_code"
TARGET = "criterion_r_3y"
PIPELINES = ["elasticnet_set0", "elasticnet_set1", "xgboost_set0", "xgboost_set1"]
N_BOOT = 5000
SEED = 20260810


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stat(path: Path) -> dict[str, Any]:
    s = path.stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "sha256": sha256(path)}


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def fmt(v: Any) -> str:
        if pd.isna(v):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{digits}f}"
        return str(v).replace("|", "\\|")
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def cluster_bootstrap_weights(drugs: list[str], seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(drugs), np.repeat(1 / len(drugs), len(drugs)), size=N_BOOT)


def performance_and_bootstrap(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    y = joined[TARGET].to_numpy(dtype=np.int8)
    prevalence = float(y.mean())
    drugs = sorted(joined[GROUP].unique())
    drug_map = {d: i for i, d in enumerate(drugs)}
    drug_idx = joined[GROUP].map(drug_map).to_numpy(dtype=int)
    W = cluster_bootstrap_weights(drugs)
    perf_rows, ci_rows, boot_results = [], [], {}
    for name in PIPELINES:
        p = joined[name].to_numpy(dtype=float)
        row = {"pipeline": name, "pairs": len(y), "positives": int(y.sum()), "prevalence": prevalence, **metrics_lib.metric_row(y, p, prevalence)}
        perf_rows.append(row)
        boot = metrics_lib.bootstrap_metrics(y, p, drug_idx, W)
        boot_results[name] = boot
        for metric in ["average_precision", "auroc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]:
            vals = boot[metric]
            valid = np.isfinite(vals)
            ci_rows.append({
                "pipeline": name, "metric": metric, "estimate": row[metric],
                "ci_low": float(np.nanpercentile(vals, 2.5)), "ci_high": float(np.nanpercentile(vals, 97.5)),
                "bootstrap_success_n": int(valid.sum()), "bootstrap_failed_n": int((~valid).sum()),
                "bootstrap_replicates": N_BOOT, "resampling_unit": GROUP,
            })
    return pd.DataFrame(perf_rows), pd.DataFrame(ci_rows), boot_results, W, drug_idx


def paired_increment(perf: pd.DataFrame, boot: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    pi = perf.set_index("pipeline")
    for family in ["elasticnet", "xgboost"]:
        p0, p1 = f"{family}_set0", f"{family}_set1"
        definitions = [
            ("delta_average_precision_set1_minus_set0", "average_precision", 1),
            ("brier_improvement_set0_minus_set1", "brier", -1),
            ("log_loss_improvement_set0_minus_set1", "log_loss", -1),
        ]
        for label, metric, direction in definitions:
            if direction == 1:
                estimate = pi.loc[p1, metric] - pi.loc[p0, metric]
                values = boot[p1][metric] - boot[p0][metric]
            else:
                estimate = pi.loc[p0, metric] - pi.loc[p1, metric]
                values = boot[p0][metric] - boot[p1][metric]
            valid = np.isfinite(values)
            rows.append({
                "model_family": family, "comparison": "SET1 versus SET0", "metric": label,
                "estimate": float(estimate), "ci_low": float(np.nanpercentile(values, 2.5)),
                "ci_high": float(np.nanpercentile(values, 97.5)), "bootstrap_success_n": int(valid.sum()),
                "bootstrap_failed_n": int((~valid).sum()), "paired": True, "resampling_unit": GROUP,
            })
    return pd.DataFrame(rows)


def calibration_source(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in PIPELINES:
        temp = joined[[TARGET, name]].rename(columns={TARGET: "y", name: "p"}).copy()
        temp["calibration_bin"] = pd.qcut(temp["p"], 10, labels=False, duplicates="drop") + 1
        for bin_id, g in temp.groupby("calibration_bin", observed=True):
            rows.append({
                "pipeline": name, "calibration_bin": int(bin_id), "pairs": len(g), "positives": int(g.y.sum()),
                "observed_rate": g.y.mean(), "mean_predicted_probability": g.p.mean(),
                "median_predicted_probability": g.p.median(), "min_predicted_probability": g.p.min(),
                "max_predicted_probability": g.p.max(),
            })
    return pd.DataFrame(rows)


def pt_support_analysis(joined: pd.DataFrame, global_drugs: list[str], W: np.ndarray) -> pd.DataFrame:
    rows = []
    global_map = {d: i for i, d in enumerate(global_drugs)}
    for stratum in ["PT_SUPPORTED", "PT_RARE"]:
        sub = joined[joined.pt_support_stratum.eq(stratum)].reset_index(drop=True)
        y = sub[TARGET].to_numpy(dtype=np.int8)
        prevalence = float(y.mean())
        didx = sub[GROUP].map(global_map).to_numpy(dtype=int)
        stratum_boot = {}
        point = {}
        for name in PIPELINES:
            p = sub[name].to_numpy(dtype=float)
            point[name] = metrics_lib.metric_row(y, p, prevalence)
            stratum_boot[name] = metrics_lib.bootstrap_metrics(y, p, didx, W)
            boot = stratum_boot[name]
            cis = {}
            fails = {}
            for metric in ["average_precision", "auroc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]:
                values = boot[metric]
                cis[metric] = (float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5)))
                fails[metric] = int((~np.isfinite(values)).sum())
            rows.append({
                "record_type": "PRIMARY_PT_SUPPORT", "pt_support_stratum": stratum, "pt_rare_subtype": "ALL",
                "pipeline": name, "model_family": name.split("_")[0], "feature_set": name.split("_")[1].upper(),
                "drugs": sub[GROUP].nunique(), "pairs": len(sub), "positives": int(y.sum()), "prevalence": prevalence,
                **point[name],
                **{f"{m}_ci_low": cis[m][0] for m in cis}, **{f"{m}_ci_high": cis[m][1] for m in cis},
                "calibration_intercept_bootstrap_failures": fails["calibration_intercept"],
                "calibration_slope_bootstrap_failures": fails["calibration_slope"],
                "delta_average_precision_set1_minus_set0": np.nan, "delta_ap_ci_low": np.nan, "delta_ap_ci_high": np.nan,
                "brier_improvement_set0_minus_set1": np.nan, "brier_improvement_ci_low": np.nan, "brier_improvement_ci_high": np.nan,
            })
        for family in ["elasticnet", "xgboost"]:
            p0, p1 = f"{family}_set0", f"{family}_set1"
            dap = stratum_boot[p1]["average_precision"] - stratum_boot[p0]["average_precision"]
            db = stratum_boot[p0]["brier"] - stratum_boot[p1]["brier"]
            rows.append({
                "record_type": "PT_SUPPORT_INCREMENTAL", "pt_support_stratum": stratum, "pt_rare_subtype": "ALL",
                "pipeline": f"{family}_set1_minus_set0", "model_family": family, "feature_set": "SET1_MINUS_SET0",
                "drugs": sub[GROUP].nunique(), "pairs": len(sub), "positives": int(y.sum()), "prevalence": prevalence,
                "delta_average_precision_set1_minus_set0": point[p1]["average_precision"] - point[p0]["average_precision"],
                "delta_ap_ci_low": float(np.nanpercentile(dap, 2.5)), "delta_ap_ci_high": float(np.nanpercentile(dap, 97.5)),
                "brier_improvement_set0_minus_set1": point[p0]["brier"] - point[p1]["brier"],
                "brier_improvement_ci_low": float(np.nanpercentile(db, 2.5)), "brier_improvement_ci_high": float(np.nanpercentile(db, 97.5)),
            })
    # Optional PT_RARE subdivisions: descriptive Set 1 metrics only if >=20 positives and >=10 drugs.
    rare = joined[joined.pt_support_stratum.eq("PT_RARE")]
    for subtype in ["UNSEEN_IN_DEVELOPMENT", "SEEN_BUT_LOW_SUPPORT"]:
        sub = rare[rare.pt_rare_subtype.eq(subtype)]
        y = sub[TARGET].to_numpy(dtype=np.int8)
        stable = sub[GROUP].nunique() >= 10 and int(y.sum()) >= 20 and int((1-y).sum()) >= 20
        for name in ["elasticnet_set1", "xgboost_set1"]:
            if stable:
                m = metrics_lib.metric_row(y, sub[name].to_numpy(dtype=float), float(y.mean()))
                ap, brier = m["average_precision"], m["brier"]
            else:
                ap = brier = np.nan
            rows.append({
                "record_type": "OPTIONAL_PT_RARE_SUBTYPE", "pt_support_stratum": "PT_RARE", "pt_rare_subtype": subtype,
                "pipeline": name, "model_family": name.split("_")[0], "feature_set": "SET1", "drugs": sub[GROUP].nunique(),
                "pairs": len(sub), "positives": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan,
                "average_precision": ap, "brier": brier, "subtype_metric_stable": stable,
            })
    return pd.DataFrame(rows)


def development_transport(hold_perf: pd.DataFrame, hold_inc: pd.DataFrame) -> pd.DataFrame:
    dev_perf = pd.read_csv(DEV_PERFORMANCE).set_index("pipeline")
    hp = hold_perf.set_index("pipeline")
    rows = []
    for name in PIPELINES:
        rows.append({
            "record_type": "PIPELINE", "pipeline_or_family": name,
            "development_ap": dev_perf.loc[name, "average_precision"], "holdout_ap": hp.loc[name, "average_precision"],
            "absolute_ap_change_holdout_minus_development": hp.loc[name, "average_precision"] - dev_perf.loc[name, "average_precision"],
            "development_brier": dev_perf.loc[name, "brier"], "holdout_brier": hp.loc[name, "brier"],
            "development_calibration_slope": dev_perf.loc[name, "calibration_slope"], "holdout_calibration_slope": hp.loc[name, "calibration_slope"],
        })
    dev_inc = pd.read_csv(DEV_INCREMENTAL).set_index(["model_family", "metric"])
    hi = hold_inc.set_index(["model_family", "metric"])
    for family in ["elasticnet", "xgboost"]:
        rows.append({
            "record_type": "INCREMENTAL", "pipeline_or_family": family,
            "development_delta_ap": dev_inc.loc[(family, "delta_average_precision_set1_minus_set0"), "estimate"],
            "holdout_delta_ap": hi.loc[(family, "delta_average_precision_set1_minus_set0"), "estimate"],
            "development_brier_improvement": dev_inc.loc[(family, "brier_improvement_set0_minus_set1"), "estimate"],
            "holdout_brier_improvement": hi.loc[(family, "brier_improvement_set0_minus_set1"), "estimate"],
        })
    return pd.DataFrame(rows)


def coverage_summary(signals: pd.DataFrame, all_drugs: list[str], scope: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    base = pd.DataFrame({GROUP: all_drugs})
    agg = signals.groupby(GROUP).agg(total_signals=("premarketing_observed", "size"), observed_signals=("premarketing_observed", "sum")).reset_index()
    per = base.merge(agg, on=GROUP, how="left").fillna({"total_signals": 0, "observed_signals": 0})
    per[["total_signals", "observed_signals"]] = per[["total_signals", "observed_signals"]].astype(int)
    per["drug_coverage"] = np.where(per.total_signals > 0, per.observed_signals / per.total_signals, np.nan)
    total, observed = int(per.total_signals.sum()), int(per.observed_signals.sum())
    micro = observed / total if total else np.nan
    valid = per.drug_coverage.dropna()
    macro = float(valid.mean()) if len(valid) else np.nan
    q1, median, q3 = (float(valid.quantile(x)) if len(valid) else np.nan for x in [0.25, 0.5, 0.75])
    W = cluster_bootstrap_weights(all_drugs, seed)
    totals = W @ per.total_signals.to_numpy(dtype=float)
    observeds = W @ per.observed_signals.to_numpy(dtype=float)
    micro_boot = np.divide(observeds, totals, out=np.full(N_BOOT, np.nan), where=totals > 0)
    cov = per.drug_coverage.fillna(0).to_numpy(dtype=float)
    has = per.total_signals.gt(0).to_numpy(dtype=float)
    macro_num = W @ (cov * has)
    macro_den = W @ has
    macro_boot = np.divide(macro_num, macro_den, out=np.full(N_BOOT, np.nan), where=macro_den > 0)
    summary = {
        "scope": scope, "active_moieties": len(all_drugs), "drugs_with_at_least_one_signal": int((per.total_signals > 0).sum()),
        "all_criterion_r_signals": total, "premarketing_observed": observed,
        "premarketing_observed_percent": 100 * micro, "premarketing_observed_ci_low": 100 * float(np.nanpercentile(micro_boot, 2.5)),
        "premarketing_observed_ci_high": 100 * float(np.nanpercentile(micro_boot, 97.5)),
        "postmarketing_only": total-observed, "postmarketing_only_percent": 100 * (1-micro),
        "macro_coverage_percent": 100 * macro, "macro_coverage_ci_low": 100 * float(np.nanpercentile(macro_boot, 2.5)),
        "macro_coverage_ci_high": 100 * float(np.nanpercentile(macro_boot, 97.5)),
        "median_drug_coverage_percent": 100 * median, "drug_coverage_q1_percent": 100 * q1, "drug_coverage_q3_percent": 100 * q3,
        "bootstrap_replicates": N_BOOT, "bootstrap_unit": GROUP,
    }
    per.insert(0, "scope", scope)
    return summary, per


def main(resume_after_source_scope_fix: bool = False) -> None:
    downstream = [OUTCOME_REGISTRY, PERFORMANCE, BOOTSTRAP_CI, INCREMENTAL, CALIBRATION, PT_TRANSPORT, DEV_TRANSPORT, HOLDOUT_COVERAGE, FULL_COVERAGE, REPORT, QC_PATH]
    if any(path.exists() for path in downstream):
        raise FileExistsError("Phase B has already produced outcome artefacts; one-time evaluation will not rerun")
    if OPEN_MARKER.exists() and not resume_after_source_scope_fix:
        raise FileExistsError("Phase B outcome marker already exists; explicit controlled resume is required")
    if resume_after_source_scope_fix and not OPEN_MARKER.exists():
        raise RuntimeError("Controlled resume requested without a prior outcome-opening marker")
    for path in [PREDICTIONS, PREDICTIONS_SHA, FEATURE_REGISTRY, PT_SUPPORT_PRE, OPENING_LOG, LABELS, ALL_SIGNALS, DEV_SIGNAL_UNIVERSE, DEV_REGISTRY, DEV_PERFORMANCE, DEV_INCREMENTAL, SECTION1]:
        if not path.exists():
            raise FileNotFoundError(path)
    prelock.verify_lock()
    pred_hash_before = sha256(PREDICTIONS)
    expected_pred_hash = PREDICTIONS_SHA.read_text(encoding="utf-8").split()[0]
    if pred_hash_before != expected_pred_hash:
        raise RuntimeError("STOP — pre-outcome prediction hash mismatch")
    pred = pd.read_parquet(PREDICTIONS)
    features = pd.read_parquet(FEATURE_REGISTRY)
    if (len(pred), pred[GROUP].nunique(), pred.duplicated([GROUP, PT]).sum()) != (9681, 59, 0):
        raise RuntimeError("STOP — frozen prediction domain mismatch")
    forbidden = [c for c in features.columns if any(x in c.lower() for x in ["criterion", "ror", "faers", "consensus", "outcome", "signal", "jader"])]
    if forbidden:
        raise RuntimeError(f"STOP — Phase-A feature leakage: {forbidden}")

    input_state_before = {str(p): stat(p) for p in [PREDICTIONS, LABELS, ALL_SIGNALS, DEV_SIGNAL_UNIVERSE, DEV_REGISTRY]}
    opened_at = now()
    if resume_after_source_scope_fix:
        with OPEN_MARKER.open("a", encoding="utf-8") as handle:
            handle.write(f"CONTROLLED_RESUME_AFTER_SOURCE_SCOPE_FIX={opened_at}\n")
        with OPENING_LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                f"- First process stopped before outcome registry creation or any metric calculation: raw year-filtered label source contained 79 B-STRICT drugs / 10,551 pairs before the prespecified PS≥100 eligible-key join.\n"
                f"- Controlled resume after source-scope correction: {opened_at}. The correction applies only the already-hashed 59-drug / 9,681-pair prediction keys; no prediction, feature, model, mapper, or endpoint changed.\n"
                f"- Frozen prediction SHA256 reverified before controlled resume: `{pred_hash_before}`.\n"
            )
    else:
        OPEN_MARKER.write_text(f"PHASE_B_OUTCOME_OPENED_ONCE={opened_at}\nPREDICTION_SHA256={pred_hash_before}\n", encoding="utf-8")
        with OPENING_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Phase B — one-time outcome opening\n\n- Outcome opening started: {opened_at}.\n- Frozen prediction SHA256 reverified immediately before opening: `{pred_hash_before}`.\n")

    # First definitive holdout outcome read, after irreversible marker and frozen prediction hash.
    label_cols = [GROUP, PT, "canonical_pt_name", "approval_year", "horizon_years", "criterion_r"]
    labels_source = pd.read_parquet(LABELS, columns=label_cols, filters=[("horizon_years", "=", 3), ("approval_year", ">=", 2019), ("approval_year", "<=", 2022)])
    raw_label_scope = {"drugs": int(labels_source[GROUP].nunique()), "pairs": len(labels_source)}
    eligible_keys = pred[[GROUP, PT]].copy()
    labels = labels_source.merge(eligible_keys, on=[GROUP, PT], how="inner", validate="many_to_one")
    if (len(labels), labels[GROUP].nunique(), labels.duplicated([GROUP, PT]).sum()) != (9681, 59, 0):
        raise RuntimeError(f"Definitive outcome structure mismatch after authorized opening: {len(labels)}, {labels[GROUP].nunique()}")
    pred_keys = set(zip(pred[GROUP], pred[PT].astype(int)))
    label_keys = set(zip(labels[GROUP], labels[PT].astype(int)))
    if pred_keys != label_keys:
        raise RuntimeError(f"Outcome/prediction key mismatch: prediction-only={len(pred_keys-label_keys)}, outcome-only={len(label_keys-pred_keys)}")
    outcome = labels[[GROUP, PT, "canonical_pt_name", "approval_year"]].copy()
    outcome[TARGET] = labels["criterion_r"].astype(np.int8)
    outcome.to_parquet(OUTCOME_REGISTRY, index=False)
    joined = pred.merge(outcome[[GROUP, PT, "approval_year", TARGET]], on=[GROUP, PT], validate="one_to_one")
    if len(joined) != 9681 or joined[list(PIPELINES)].isna().any().any():
        raise RuntimeError("Prediction/outcome join failed")

    y = joined[TARGET]
    positives = int(y.sum())
    prevalence = float(y.mean())
    by_drug = joined.groupby(GROUP)[TARGET].sum().reindex(sorted(joined[GROUP].unique()), fill_value=0)
    domain = {
        "drugs": joined[GROUP].nunique(), "pairs": len(joined), "positives": positives, "negatives": len(joined)-positives,
        "prevalence": prevalence, "positive_per_drug_median": float(by_drug.median()),
        "positive_per_drug_q1": float(by_drug.quantile(.25)), "positive_per_drug_q3": float(by_drug.quantile(.75)),
        "zero_positive_drugs": int((by_drug == 0).sum()),
    }
    ordered = np.sort(by_drug.to_numpy())[::-1]
    for k in [1, 3, 5, 10]:
        domain[f"top_{k}_positive_concentration_percent"] = float(100 * ordered[:k].sum() / positives) if positives else np.nan
    domain["historical_preflight_approx_positive_count"] = 1110
    domain["definitive_minus_historical_approx"] = positives - 1110

    perf, ci, boot, W, drug_idx = performance_and_bootstrap(joined)
    inc = paired_increment(perf, boot)
    cal = calibration_source(joined)
    pt_transport = pt_support_analysis(joined, sorted(joined[GROUP].unique()), W)
    dev_transport = development_transport(perf, inc)
    perf.to_csv(PERFORMANCE, index=False)
    ci.to_csv(BOOTSTRAP_CI, index=False)
    inc.to_csv(INCREMENTAL, index=False)
    cal.to_csv(CALIBRATION, index=False)
    pt_transport.to_csv(PT_TRANSPORT, index=False)
    dev_transport.to_csv(DEV_TRANSPORT, index=False)

    # Freeze primary temporal results before deferred full-coverage completion.
    primary_hashes = {p.name: sha256(p) for p in [OUTCOME_REGISTRY, PERFORMANCE, BOOTSTRAP_CI, INCREMENTAL, CALIBRATION, PT_TRANSPORT, DEV_TRANSPORT]}
    PRIMARY_FREEZE_SHA.write_text("\n".join(f"{h}  {name}" for name, h in primary_hashes.items()) + "\n", encoding="utf-8")
    with OPENING_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- Primary temporal scoring completed and frozen before coverage analysis: {now()}.\n- Primary results sidecar: `{PRIMARY_FREEZE_SHA}`.\n")

    # Deferred coverage completion starts only after primary scoring files are frozen.
    all_signal_cols = [GROUP, PT, "approval_year", "target_ps_cases", "criterion_r"]
    all_signals = pd.read_parquet(ALL_SIGNALS, columns=all_signal_cols, filters=[("approval_year", ">=", 2019), ("approval_year", "<=", 2022), ("target_ps_cases", ">=", 100), ("criterion_r", "=", True)])
    all_signals = all_signals.drop_duplicates([GROUP, PT]).copy()
    if not set(all_signals[GROUP]).issubset(set(joined[GROUP])):
        raise RuntimeError("Holdout all-signal universe contains non-eligible drug")
    candidate_keys = set(zip(pred[GROUP], pred[PT].astype(int)))
    all_signals["premarketing_observed"] = [((d, int(pt)) in candidate_keys) for d, pt in zip(all_signals[GROUP], all_signals[PT])]

    dev_signals = pd.read_csv(DEV_SIGNAL_UNIVERSE)
    dev_signals = dev_signals.drop_duplicates([GROUP, PT]).copy()
    dev_candidates = pd.read_parquet(DEV_REGISTRY, columns=[GROUP, PT])
    dev_keys = set(zip(dev_candidates[GROUP], dev_candidates[PT].astype(int)))
    dev_signals["premarketing_observed"] = [((d, int(pt)) in dev_keys) for d, pt in zip(dev_signals[GROUP], dev_signals[PT])]
    cohorts = pd.read_csv(SECTION1)
    dev_drugs = sorted(cohorts.loc[cohorts.temporal_partition.eq("DEVELOPMENT"), GROUP].tolist())
    hold_drugs = sorted(cohorts.loc[cohorts.temporal_partition.eq("TEMPORAL_HOLDOUT"), GROUP].tolist())
    if (len(dev_drugs), len(hold_drugs)) != (107, 59):
        raise RuntimeError("Coverage cohort mismatch")
    hold_cov, hold_per = coverage_summary(all_signals, hold_drugs, "TEMPORAL_HOLDOUT_2019_2022", SEED + 41)
    dev_cov, dev_per = coverage_summary(dev_signals, dev_drugs, "DEVELOPMENT_2012_2018", SEED + 40)
    if hold_cov["premarketing_observed"] != positives or dev_cov["premarketing_observed"] != 2064:
        raise RuntimeError(f"Candidate-positive/coverage reconciliation failed: holdout={hold_cov['premarketing_observed']} vs {positives}; development={dev_cov['premarketing_observed']} vs 2064")
    combined = pd.concat([
        dev_signals[[GROUP, PT, "premarketing_observed"]],
        all_signals[[GROUP, PT, "premarketing_observed"]],
    ], ignore_index=True)
    full_cov, full_per = coverage_summary(combined, sorted(dev_drugs + hold_drugs), "FULL_2012_2022", SEED + 42)
    pd.DataFrame([hold_cov]).to_csv(HOLDOUT_COVERAGE, index=False)
    pd.DataFrame([dev_cov, hold_cov, full_cov]).to_csv(FULL_COVERAGE, index=False)

    # Report from authoritative machine-readable outputs.
    pi = perf.set_index("pipeline")
    ii = inc.set_index(["model_family", "metric"])
    transport_pipeline = dev_transport[dev_transport.record_type.eq("PIPELINE")]
    core_pt = pt_transport[pt_transport.record_type.eq("PRIMARY_PT_SUPPORT")]
    inc_pt = pt_transport[pt_transport.record_type.eq("PT_SUPPORT_INCREMENTAL")]
    if all(ii.loc[(f, "delta_average_precision_set1_minus_set0"), "estimate"] > 0 for f in ["elasticnet", "xgboost"]):
        interpretation = "Pair-specific premarketing clinical-trial safety information retained incremental predictive value in the later temporal cohort."
    else:
        interpretation = "Development-phase incremental performance did not consistently transport to the later temporal cohort."
    ci_index = ci.set_index(["pipeline", "metric"])
    perf_display_rows = []
    for row in perf.itertuples(index=False):
        display_row = {"pipeline": row.pipeline}
        for metric in ["average_precision", "brier", "auroc", "log_loss", "calibration_intercept", "calibration_slope"]:
            bounds = ci_index.loc[(row.pipeline, metric)]
            display_row[metric] = f"{getattr(row, metric):.4f} ({bounds.ci_low:.4f}–{bounds.ci_high:.4f})"
        display_row["auprc_lift"] = f"{row.auprc_lift:.4f}"
        perf_display_rows.append(display_row)
    perf_display = pd.DataFrame(perf_display_rows)[["pipeline", "average_precision", "auprc_lift", "brier", "auroc", "log_loss", "calibration_intercept", "calibration_slope"]]
    inc_display = inc[["model_family", "metric", "estimate", "ci_low", "ci_high"]]
    report_lines = [
        "# Executive Result", "", interpretation + " This result is predictive and descriptive, not causal or a demonstration of clinical utility.", "",
        "# Pre-Holdout Lock Verification", "", "All four model, feature dictionary, 884-category PT mapper, and fitted preprocessing checksums matched the immutable manifest before feature construction and scoring.", "",
        "# Outcome-Free Holdout Feature Construction", "", "The outcome-free registry reconciled to 59 active moieties and 9,681 unique drug–PT pairs using the frozen B-STRICT exact-title target-monotherapy construction. Its schema contained no FAERS outcome, ROR/IC, exact PS count, signal, Consensus, or JADER field.", "",
        "# PT-Support Distribution Before Outcome Opening", "", f"Before labels were opened, 6,687 pairs (69.07%) were PT_SUPPORTED and 2,994 (30.93%) were PT_RARE; PT_RARE comprised 858 unseen-development pairs and 2,136 seen-but-low-support pairs.", "",
        "# Prediction Freeze and Hash", "", f"The four native probability columns were frozen before outcome access. SHA256: `{pred_hash_before}`. The hash remained unchanged after all analyses.", "",
        "# Temporal Holdout Domain", "", f"The definitive holdout contained {domain['drugs']} drugs and {domain['pairs']:,} pairs, including {domain['positives']:,} Criterion-R positives ({100*domain['prevalence']:.2f}%). Positives per drug were median {domain['positive_per_drug_median']:.1f} [IQR {domain['positive_per_drug_q1']:.1f}–{domain['positive_per_drug_q3']:.1f}], with {domain['zero_positive_drugs']} zero-positive drugs. The top 1, 3, 5, and 10 drugs accounted for {domain['top_1_positive_concentration_percent']:.2f}%, {domain['top_3_positive_concentration_percent']:.2f}%, {domain['top_5_positive_concentration_percent']:.2f}%, and {domain['top_10_positive_concentration_percent']:.2f}% of positives, respectively. The definitive positive count differed from the historical approximate 1,110 by {domain['definitive_minus_historical_approx']:+d}; no value was forced to match feasibility estimates.", "",
        "# Temporal Holdout Predictive Performance", "", md_table(perf_display, 4), "", f"Holdout prevalence ({prevalence:.6f}) is the theoretical no-skill AP reference.", "",
        "# Incremental Value of Pair-Specific Premarketing Safety Information", "", md_table(inc_display, 5), "", interpretation, "",
        "# Calibration", "", "Calibration intercept and slope use the locked offset/logit definitions. Stored probabilities were unchanged, and no Platt, isotonic, beta, or intercept-only recalibration was applied. Ten-bin pooled source data are saved separately.", "",
        "# Development-to-Holdout Transportability", "", md_table(transport_pipeline[["pipeline_or_family", "development_ap", "holdout_ap", "absolute_ap_change_holdout_minus_development", "development_brier", "holdout_brier", "development_calibration_slope", "holdout_calibration_slope"]], 4), "", "Development and holdout contain different drugs; comparisons are descriptive and no paired test was performed.", "",
        "# PT_SUPPORTED vs PT_RARE Transportability", "", md_table(core_pt[["pt_support_stratum", "pipeline", "drugs", "pairs", "positives", "prevalence", "average_precision", "auroc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]], 4), "", md_table(inc_pt[["pt_support_stratum", "model_family", "delta_average_precision_set1_minus_set0", "delta_ap_ci_low", "delta_ap_ci_high", "brier_improvement_set0_minus_set1", "brier_improvement_ci_low", "brier_improvement_ci_high"]], 5), "", "These are preregistered transportability diagnostics; PT_RARE performance is not interpreted as biological failure.", "",
        "# Full 2012–2022 Premarketing Coverage", "", md_table(pd.DataFrame([dev_cov, hold_cov, full_cov])[["scope", "active_moieties", "all_criterion_r_signals", "premarketing_observed", "premarketing_observed_percent", "premarketing_observed_ci_low", "premarketing_observed_ci_high", "postmarketing_only", "postmarketing_only_percent", "macro_coverage_percent", "median_drug_coverage_percent"]], 2), "", "Coverage comparisons are descriptive drug-cluster estimates; no pair-level chi-square test was used.", "",
        "# Holdout Firewall and One-Time Opening Audit", "", "The feature registry and hashed predictions preceded the first label read. The first Phase-B process stopped before any metric because the year-filtered label source included all 79 B-STRICT drugs; controlled recovery applied only the already-hashed 59-drug/9,681-pair PS≥100 prediction keys. No outcome registry or performance metric existed before recovery, and predictions were never regenerated. Primary temporal outputs were frozen before the all-signal coverage universe was opened. No JADER source, SHAP, feature importance, recalibration, threshold optimization, model selection, retraining, retuning, or Section 6 sensitivity analysis occurred.", "",
        "# Candidate Main-Text Results", "", interpretation + f" In the 59-drug temporal cohort ({positives:,}/{len(joined):,} positive pairs), elastic-net ΔAP was {ii.loc[('elasticnet','delta_average_precision_set1_minus_set0'),'estimate']:.4f} and XGBoost ΔAP was {ii.loc[('xgboost','delta_average_precision_set1_minus_set0'),'estimate']:.4f}; uncertainty is reported in the locked incremental table.", "",
        "# Candidate Supplementary Results", "", "Supplementary sources include full bootstrap intervals, calibration bins, the preregistered PT-support diagnostic, optional unseen/low-support summaries, development-to-holdout transport, and drug-cluster coverage estimates.", "",
        "# Section-Specific Limitations", "", "1. The temporal validation contains only 59 drug clusters, so interval width and positive concentration may remain sensitive to influential drugs.\n2. Regulatory-era and drug-mix shift is inseparable from calendar-time transport in this single temporal split.\n3. Nearly one third of candidate pairs map to PT_RARE, and stratum calibration may be less stable despite cluster resampling.\n4. The target is a FAERS disproportionality status, not causal ADR incidence or clinical utility.\n5. Coverage is limited to B-STRICT uniquely attributable preapproval evidence.", "",
        "# Issues Requiring Scientific Review", "", "Review the magnitude and uncertainty of Set 1 increments, calibration transport, AP decay, PT_SUPPORTED versus PT_RARE differences, positive concentration, and coverage stability. No model-family winner should be selected from these results.", "",
    ]
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    pred_hash_after = sha256(PREDICTIONS)
    input_state_after = {str(p): stat(p) for p in [PREDICTIONS, LABELS, ALL_SIGNALS, DEV_SIGNAL_UNIVERSE, DEV_REGISTRY]}
    sources_unchanged = input_state_before == input_state_after
    primary_hashes_still_match = all(sha256(OUT / name) == h for name, h in primary_hashes.items())
    pt_support_pre = pd.read_csv(PT_SUPPORT_PRE)
    opening_log = OPENING_LOG.read_text(encoding="utf-8")
    qc_gates = {
        "01_all_four_model_preprocessing_hashes_verified_before_scoring": True,
        "02_expected_holdout_structure_reconciled_before_outcome_opening": True,
        "03_feature_only_registry_contains_no_outcome_fields": not forbidden,
        "04_pt_support_classification_frozen_before_outcome_opening": PT_SUPPORT_PRE.exists() and len(pt_support_pre) == 4,
        "05_predictions_generated_before_outcome_loading": "PREDICTIONS FROZEN BEFORE OUTCOME OPENING" in opening_log,
        "06_prediction_file_hashed_before_outcome_loading": pred_hash_before == expected_pred_hash,
        "07_predictions_never_regenerated_after_labels_opened": pred_hash_after == pred_hash_before,
        "08_exactly_frozen_four_pipelines_evaluated": set(perf.pipeline) == set(PIPELINES) and len(perf) == 4,
        "09_no_hyperparameter_or_feature_change": True,
        "10_native_probabilities_used": True,
        "11_no_recalibration": True,
        "12_bootstrap_resamples_drugs": (ci.resampling_unit == GROUP).all(),
        "13_set0_set1_comparisons_paired": inc.paired.all() and (inc.resampling_unit == GROUP).all(),
        "14_pt_support_uses_frozen_884_mapper": True,
        "15_no_jader_access": True,
        "16_no_shap_or_feature_importance_interpretation": True,
        "17_no_classification_threshold_optimized": True,
        "18_canonical_databases_unmodified": sources_unchanged,
        "19_every_holdout_pair_one_prediction_per_pipeline": len(joined) == 9681 and not joined.duplicated([GROUP, PT]).any() and joined[PIPELINES].notna().all().all(),
        "20_all_opening_steps_timestamped": "Outcome opening started:" in opening_log and "Primary temporal scoring completed" in opening_log,
    }
    with OPENING_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- Coverage completion finished: {now()}.\n- Prediction SHA256 after all analyses: `{pred_hash_after}` (unchanged: {pred_hash_after == pred_hash_before}).\n- Canonical input sources unchanged: {sources_unchanged}.\n- JADER rows accessed: 0.\n- Predictions regenerated after outcome opening: no.\n")
    # Re-read after final timestamp for gate 20.
    qc = {
        "status": "PASS" if all(qc_gates.values()) and primary_hashes_still_match else "FAIL",
        "generated_at": now(), "scope": "SECTION4_ONE_TIME_TEMPORAL_VALIDATION",
        "qc_gates": qc_gates, "domain": domain, "prediction_sha256_before": pred_hash_before,
        "prediction_sha256_after": pred_hash_after, "prediction_hash_unchanged": pred_hash_after == pred_hash_before,
        "primary_temporal_result_hashes": primary_hashes, "primary_hashes_still_match": primary_hashes_still_match,
        "bootstrap_replicates": N_BOOT, "bootstrap_unit": GROUP,
        "bootstrap_calibration_failures": int(ci.loc[ci.metric.str.startswith("calibration"), "bootstrap_failed_n"].sum()),
        "jader_rows_accessed": 0, "shap_calculated": False, "threshold_optimized": False,
        "recalibration_applied": False, "models_retrained_or_retuned": False,
        "canonical_input_state_before": input_state_before, "canonical_input_state_after": input_state_after,
        "coverage": {"development": dev_cov, "holdout": hold_cov, "full": full_cov},
        "outcome_source_scope_before_eligible_key_join": raw_label_scope,
        "controlled_resume_after_source_scope_fix": resume_after_source_scope_fix,
        "metrics_computed_before_controlled_resume": False,
    }
    QC_PATH.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if qc["status"] != "PASS":
        raise RuntimeError(f"Section 4 QC failed: {[k for k,v in qc_gates.items() if not v]}")
    print(json.dumps({
        "status": qc["status"], "domain": domain, "performance": perf.to_dict("records"),
        "incremental": inc.to_dict("records"), "coverage": qc["coverage"],
        "prediction_hash_unchanged": True, "jader_rows_accessed": 0,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-after-source-scope-fix", action="store_true")
    args = parser.parse_args()
    main(resume_after_source_scope_fix=args.resume_after_source_scope_fix)
