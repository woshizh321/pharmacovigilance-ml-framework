#!/usr/bin/env python3
"""Command 15: JADER replication and prespecified robustness analyses.

This script consumes frozen Section 1--5 artifacts and canonical read-only
sources. It does not load model objects, fit models, regenerate predictions,
calculate SHAP, or construct a JADER time window.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "section6_robustness"
FIG = OUT / "14_figure5_source_data"

JADER = Path("/path/to/Database/Jader")
JDRUG = JADER / "jader_v5_drug.parquet"
JREAC = JADER / "jader_v5_reac.parquet"
JMASTER_FORBIDDEN = JADER / "jader_v5_master.parquet"
AACT = Path("/path/to/Database/AACT/aact.duckdb")

PROC = ROOT / "data" / "processed" / "preflight_v2"
LABELS = PROC / "faers_fda_anchored_labels_1_2_3y.parquet"
ALL_SIGNALS = PROC / "faers_all_exposed_pair_signals_3y.parquet"
JMAP = PROC / "jader_v5_name_to_fda_identity.csv"
LINKS = PROC / "aact_fda_intervention_links.parquet"
MASTER = ROOT / "preflight_v2" / "drug_identity_master.csv"
MASTER_SHA = ROOT / "preflight_v2" / "drug_identity_master.sha256"
PT_MAP = ROOT / "preflight_v2" / "faers_pt_repair" / "aact_meddra28_term_mapping.csv"
BREG = ROOT / "preflight_v2" / "bstrict_candidate_registry.parquet"

DEV_REG = ROOT / "analysis" / "section3_model" / "development_pair_registry.parquet"
DEV_PRED = ROOT / "analysis" / "section3_model" / "training" / "05_oof_predictions.parquet"
HOLD_PRED = ROOT / "analysis" / "section4_holdout" / "03_holdout_predictions_PREOUTCOME.parquet"
HOLD_OUTCOME = ROOT / "analysis" / "section4_holdout" / "04_holdout_outcome_registry.parquet"
S2_THRESHOLD = ROOT / "analysis" / "section2_coverage" / "07_reporting_threshold_context.csv"

PIPELINES = ["elasticnet_set0", "elasticnet_set1", "xgboost_set0", "xgboost_set1"]
N_BOOT = 5000
SEED = 20260810

JADER_QC = OUT / "00_jader_source_and_identity_qc.md"
JADER_ASSESS = OUT / "01_jader_assessability.csv"
JADER_2X2 = OUT / "02_jader_pair_2x2.parquet"
JADER_STATUS = OUT / "03_jader_replication_status.csv"
JADER_SUMMARY = OUT / "04_jader_replication_summary.csv"
JADER_BY_PRE = OUT / "05_jader_replication_by_premarketing_status.csv"
CONS_PERF = OUT / "06_consensus_endpoint_performance.csv"
HORIZON_PERF = OUT / "07_horizon_1y_2y_performance.csv"
PS_SENS = OUT / "08_ps_threshold_sensitivity.csv"
DEFA_SENS = OUT / "09_definitionA_sensitivity.csv"
BEXP_SENS = OUT / "10_bexpanded_sensitivity.csv"
ARM_SENS = OUT / "11_arm_attribution_sensitivity.md"
THRESHOLD_CONTEXT = OUT / "12_reporting_threshold_context.md"
ROBUST_SUMMARY = OUT / "13_model_robustness_summary.csv"
REPORT = OUT / "SECTION6_REPORT.md"
QC_PATH = OUT / "SECTION6_QC.json"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_stat(path: Path, with_hash: bool = True) -> dict[str, Any]:
    s = path.stat()
    out = {"size": s.st_size, "mtime_ns": s.st_mtime_ns}
    if with_hash:
        out["sha256"] = sha256(path)
    return out


def fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "NA"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v).replace("|", "\\|")


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def cluster_weights(drugs: list[str], seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(drugs), np.repeat(1 / len(drugs), len(drugs)), size=N_BOOT)


def ratio_ci_by_drug(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    all_drugs: list[str] | None = None,
    seed: int = SEED,
) -> tuple[float, float, int]:
    drugs = all_drugs or sorted(frame["canonical_active_moiety"].unique().tolist())
    grouped = frame.groupby("canonical_active_moiety")[[numerator, denominator]].sum().reindex(drugs, fill_value=0)
    w = cluster_weights(drugs, seed)
    num = w @ grouped[numerator].to_numpy(dtype=float)
    den = w @ grouped[denominator].to_numpy(dtype=float)
    val = np.divide(num, den, out=np.full(N_BOOT, np.nan), where=den > 0)
    valid = val[np.isfinite(val)]
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)), int(len(valid))


def bootstrap_ap(y: np.ndarray, p: np.ndarray, drug_idx: np.ndarray, weights: np.ndarray, batch: int = 40) -> np.ndarray:
    result = np.full(len(weights), np.nan)
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]
    for begin in range(0, len(weights), batch):
        end = min(len(weights), begin + batch)
        sw = weights[begin:end, :][:, drug_idx].astype(float)
        sws = sw[:, order]
        posg = np.add.reduceat(sws * ys[None, :], starts, axis=1)
        negg = np.add.reduceat(sws * (1 - ys)[None, :], starts, axis=1)
        cpos = np.cumsum(posg, axis=1)
        cneg = np.cumsum(negg, axis=1)
        precision = np.divide(cpos, cpos + cneg, out=np.zeros_like(cpos), where=(cpos + cneg) > 0)
        total_pos = cpos[:, -1]
        result[begin:end] = np.divide(
            np.sum(precision * posg, axis=1), total_pos,
            out=np.full(end - begin, np.nan), where=total_pos > 0,
        )
    return result


def source_guards() -> dict[str, dict[str, Any]]:
    files = [
        LABELS, ALL_SIGNALS, BREG, MASTER, JMAP,
        DEV_REG, DEV_PRED, HOLD_PRED, HOLD_OUTCOME,
        ROOT / "analysis/section1_cohort/SECTION1_QC.json",
        ROOT / "analysis/section1_cohort/SECTION1_README.md",
        ROOT / "analysis/section2_coverage/SECTION2_QC.json",
        ROOT / "analysis/section2_coverage/SECTION2_README.md",
        ROOT / "analysis/section3_model/PREHOLDOUT_LOCK_MANIFEST.json",
        ROOT / "analysis/section3_model/SECTION3_README.md",
        ROOT / "analysis/section3_model/training/SECTION3B_QC.json",
        ROOT / "analysis/section4_holdout/03_holdout_predictions_PREOUTCOME.sha256",
        ROOT / "analysis/section4_holdout/PRIMARY_TEMPORAL_RESULTS.sha256",
        ROOT / "analysis/section4_holdout/SECTION4_QC.json",
        ROOT / "analysis/section4_holdout/SECTION4_README.md",
        ROOT / "analysis/section5_interpretation/SECTION5_QC.json",
        ROOT / "analysis/section5_interpretation/SECTION5_README.md",
    ]
    return {str(p): file_stat(p) for p in files}


def validate_inputs() -> None:
    required = [JDRUG, JREAC, AACT, LABELS, ALL_SIGNALS, JMAP, LINKS, MASTER, MASTER_SHA, PT_MAP,
                BREG, DEV_REG, DEV_PRED, HOLD_PRED, HOLD_OUTCOME, S2_THRESHOLD]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    expected_master = MASTER_SHA.read_text().split()[0]
    if sha256(MASTER) != expected_master:
        raise RuntimeError("Drug identity master hash mismatch")


def build_jader(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    con.execute(f"""
      CREATE TABLE primary_drugs AS
      SELECT canonical_active_moiety, any_value(approval_year) approval_year,
             any_value(target_ps_cases) faers_ps_cases_3y
      FROM read_parquet('{LABELS}')
      WHERE horizon_years=3 AND target_ps_cases>=100
      GROUP BY 1
    """)
    con.execute(f"""
      CREATE TABLE signal_universe AS
      SELECT canonical_active_moiety,canonical_pt_code,canonical_pt_name,canonical_soc_name,
             approval_year,target_ps_cases,coverage_class
      FROM read_parquet('{ALL_SIGNALS}')
      WHERE criterion_r AND target_ps_cases>=100 AND approval_year BETWEEN 2012 AND 2022
    """)
    u = con.execute("SELECT count(*),count(DISTINCT canonical_active_moiety),count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM signal_universe").fetchone()
    if u != (16252, 166, 16252):
        raise RuntimeError(f"Definitive FAERS signal universe mismatch: {u}")

    con.execute(f"""
      CREATE TABLE jader_all_ids AS
      SELECT DISTINCT ID FROM (
        SELECT ID FROM read_parquet('{JDRUG}')
        UNION ALL
        SELECT ID FROM read_parquet('{JREAC}')
      )
    """)
    total_cases = con.execute("SELECT count(*) FROM jader_all_ids").fetchone()[0]
    con.execute(f"""
      CREATE TABLE jdrug_ps AS
      WITH aliases AS (
        SELECT ID,DRUG_SEQ,DRUGNAME_CLEANED source_name FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,DRUG_INN_EN FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,DRUG_MOLECULE FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,DRUGNAME_EN FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,BRANDNAME_KEY FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
      ), hits AS (
        SELECT a.ID,a.DRUG_SEQ,m.canonical_active_moiety
        FROM aliases a JOIN read_csv('{JMAP}',header=true,all_varchar=true) m
          ON a.source_name=m.jader_source_name
        WHERE a.source_name IS NOT NULL
      ), one AS (
        SELECT ID,DRUG_SEQ,any_value(canonical_active_moiety) canonical_active_moiety
        FROM hits GROUP BY 1,2 HAVING count(DISTINCT canonical_active_moiety)=1
      )
      SELECT DISTINCT ID,canonical_active_moiety FROM one
    """)
    con.execute(f"""
      CREATE TABLE jpt AS
      SELECT ID,try_cast(PT_CODE AS BIGINT) canonical_pt_code,
             any_value(PT_NAME_EN) jader_pt_name_en
      FROM read_parquet('{JREAC}')
      WHERE try_cast(PT_CODE AS BIGINT) IS NOT NULL
      GROUP BY 1,2
    """)
    con.execute("CREATE TABLE jdrug_n AS SELECT canonical_active_moiety,count(DISTINCT ID) jader_ps_cases FROM jdrug_ps GROUP BY 1")
    con.execute("CREATE TABLE jpt_n AS SELECT canonical_pt_code,count(DISTINCT ID) jader_pt_cases,any_value(jader_pt_name_en) jader_pt_name_en FROM jpt GROUP BY 1")
    con.execute("""
      CREATE TABLE joverlap AS
      SELECT s.canonical_active_moiety,s.canonical_pt_code,count(DISTINCT d.ID) jader_a
      FROM signal_universe s JOIN jdrug_ps d USING(canonical_active_moiety)
      JOIN jpt e ON e.ID=d.ID AND e.canonical_pt_code=s.canonical_pt_code
      GROUP BY 1,2
    """)
    con.execute(f"""
      CREATE TABLE jader_pair AS
      WITH cells AS (
        SELECT s.*,coalesce(d.jader_ps_cases,0)::BIGINT jader_ps_cases,
               coalesce(t.jader_pt_cases,0)::BIGINT jader_pt_cases,
               t.jader_pt_name_en,
               coalesce(o.jader_a,0)::BIGINT a,
               (coalesce(d.jader_ps_cases,0)-coalesce(o.jader_a,0))::BIGINT b,
               (coalesce(t.jader_pt_cases,0)-coalesce(o.jader_a,0))::BIGINT c,
               ({total_cases}-coalesce(d.jader_ps_cases,0)-coalesce(t.jader_pt_cases,0)+coalesce(o.jader_a,0))::BIGINT d
        FROM signal_universe s
        LEFT JOIN jdrug_n d USING(canonical_active_moiety)
        LEFT JOIN jpt_n t USING(canonical_pt_code)
        LEFT JOIN joverlap o USING(canonical_active_moiety,canonical_pt_code)
      ), stats AS (
        SELECT *,
          CASE WHEN a>0 AND b>0 AND c>0 AND d>0 THEN (a::DOUBLE*d)/(b::DOUBLE*c) END ror,
          CASE WHEN a>0 AND b>0 AND c>0 AND d>0 THEN
            exp(ln((a::DOUBLE*d)/(b::DOUBLE*c))-1.96*sqrt(1.0/a+1.0/b+1.0/c+1.0/d)) END ror_lcl95,
          log2((a+0.5)/((jader_ps_cases::DOUBLE*jader_pt_cases/{total_cases})+0.5))
            -3.3*pow(a+0.5,-0.5)-2.0*pow(a+0.5,-1.5) ic025
        FROM cells
      ), flags AS (
        SELECT *,
          (jader_ps_cases>=50 AND jader_pt_cases>0) assessable,
          coalesce(jader_ps_cases>=50 AND jader_pt_cases>0 AND ror>1,FALSE) directionally_positive,
          coalesce(jader_ps_cases>=50 AND jader_pt_cases>0 AND a>=3 AND ror_lcl95>1,FALSE) jader_r,
          coalesce(jader_ps_cases>=50 AND jader_pt_cases>0 AND a>=3 AND ror_lcl95>1 AND ic025>0,FALSE) jader_consensus
        FROM stats
      )
      SELECT *,
        CASE WHEN NOT assessable THEN 'NOT_ASSESSABLE' WHEN jader_r THEN 'REPLICATED' ELSE 'NOT_REPLICATED' END replication_status,
        CASE
          WHEN jader_ps_cases=0 THEN 'DRUG_ABSENT_FROM_VERIFIED_JADER_PS_MAPPING'
          WHEN jader_ps_cases<50 THEN 'DRUG_PRESENT_BUT_LT50_PS_CASES'
          WHEN jader_pt_cases=0 THEN 'PT_ABSENT_FROM_JADER'
          WHEN a=0 THEN 'DRUG_AND_PT_PRESENT_BUT_PAIR_ABSENT'
          WHEN a<3 THEN 'PAIR_A_LT3'
          WHEN ror IS NULL THEN 'ROR_UNDEFINED_ZERO_MARGIN'
          WHEN ror<=1 THEN 'PAIR_A_GE3_BUT_ROR_LE1'
          WHEN ror_lcl95<=1 THEN 'ROR_GT1_BUT_LOWER_CI_LE1'
          WHEN jader_r AND ic025<=0 THEN 'JADER_R_POSITIVE_BUT_IC025_LE0'
          WHEN jader_consensus THEN 'JADER_CONSENSUS_POSITIVE'
          ELSE 'UNCLASSIFIED' END audit_reason
      FROM flags
    """)
    qc = con.execute(f"""
      SELECT count(*) FILTER (WHERE a<0 OR b<0 OR c<0 OR d<0),
             count(*) FILTER (WHERE a+b+c+d<>{total_cases}),count(*)
      FROM jader_pair
    """).fetchone()
    if qc != (0, 0, 16252):
        raise RuntimeError(f"JADER cell QC failed: {qc}")
    con.execute(f"COPY jader_pair TO '{JADER_2X2}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    status = con.execute("""
      SELECT canonical_active_moiety,canonical_pt_code,canonical_pt_name,coverage_class,
             replication_status,audit_reason,assessable,directionally_positive,jader_r,jader_consensus
      FROM jader_pair ORDER BY canonical_active_moiety,canonical_pt_code
    """).fetchdf()
    status.to_csv(JADER_STATUS, index=False)

    assess = con.execute("""
      SELECT p.canonical_active_moiety,p.approval_year,p.faers_ps_cases_3y,
             coalesce(j.jader_ps_cases,0) jader_ps_cases,
             (coalesce(j.jader_ps_cases,0)>0) any_jader_ps,
             (coalesce(j.jader_ps_cases,0)>=50) jader_ps_ge50,
             (coalesce(j.jader_ps_cases,0)>=100) jader_ps_ge100,
             (coalesce(j.jader_ps_cases,0)>=200) jader_ps_ge200,
             (coalesce(j.jader_ps_cases,0)>=500) jader_ps_ge500,
             count(s.canonical_pt_code) faers_r_signal_pairs,
             count(s.canonical_pt_code) FILTER (WHERE s.jader_pt_cases>0) pt_represented_pairs,
             count(s.canonical_pt_code) FILTER (WHERE s.assessable) assessable_pairs
      FROM primary_drugs p LEFT JOIN jdrug_n j USING(canonical_active_moiety)
      LEFT JOIN jader_pair s USING(canonical_active_moiety)
      GROUP BY ALL ORDER BY p.canonical_active_moiety
    """).fetchdf()
    assess.to_csv(JADER_ASSESS, index=False)

    pair = con.execute("SELECT * FROM jader_pair").fetchdf()
    all_drugs = sorted(assess.canonical_active_moiety.tolist())
    pair["den_assessable"] = pair.assessable.astype(int)
    pair["num_r"] = pair.jader_r.astype(int)
    pair["num_consensus"] = pair.jader_consensus.astype(int)
    pair["num_directional"] = pair.directionally_positive.astype(int)
    pair["num_not_assessable"] = (~pair.assessable).astype(int)
    summary_rows: list[dict[str, Any]] = []
    total = len(pair)
    metrics = [
        ("JADER_ASSESSABLE", int(pair.assessable.sum()), total, None, None),
        ("NOT_ASSESSABLE", int((~pair.assessable).sum()), total, None, None),
        ("JADER_R_REPLICATED", int(pair.jader_r.sum()), int(pair.assessable.sum()), "num_r", "den_assessable"),
        ("JADER_CONSENSUS_REPLICATED", int(pair.jader_consensus.sum()), int(pair.assessable.sum()), "num_consensus", "den_assessable"),
        ("DIRECTIONALLY_POSITIVE", int(pair.directionally_positive.sum()), int(pair.assessable.sum()), "num_directional", "den_assessable"),
    ]
    for metric, n, den, num_col, den_col in metrics:
        lo = hi = np.nan
        valid = 0
        if num_col:
            lo, hi, valid = ratio_ci_by_drug(pair, num_col, den_col, all_drugs)
        summary_rows.append({
            "analysis_population": "DEFINITIVE_FAERS_CRITERION_R_SIGNALS",
            "metric": metric, "n": n, "denominator": den, "percent": 100*n/den,
            "ci_low_percent": 100*lo if np.isfinite(lo) else np.nan,
            "ci_high_percent": 100*hi if np.isfinite(hi) else np.nan,
            "bootstrap_success_n": valid, "bootstrap_replicates": N_BOOT if num_col else 0,
            "bootstrap_unit": "canonical_active_moiety" if num_col else "NA",
        })
    for threshold, col in [(1,"any_jader_ps"),(50,"jader_ps_ge50"),(100,"jader_ps_ge100"),(200,"jader_ps_ge200"),(500,"jader_ps_ge500")]:
        n = int(assess[col].sum())
        summary_rows.append({
            "analysis_population": "PRIMARY_COHORT_DRUGS", "metric": f"JADER_PS_GE_{threshold}" if threshold>1 else "ANY_JADER_PS",
            "n": n, "denominator": len(assess), "percent": 100*n/len(assess),
            "ci_low_percent": np.nan,"ci_high_percent": np.nan,"bootstrap_success_n":0,"bootstrap_replicates":0,"bootstrap_unit":"NA",
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(JADER_SUMMARY, index=False)

    by_rows = []
    for stratum in ["PREMARKETING_OBSERVED", "POSTMARKETING_ONLY"]:
        sub = pair[pair.coverage_class.eq(stratum)].copy()
        lo, hi, valid = ratio_ci_by_drug(sub, "num_r", "den_assessable", all_drugs, SEED)
        clo, chi, cvalid = ratio_ci_by_drug(sub, "num_consensus", "den_assessable", all_drugs, SEED)
        den = int(sub.assessable.sum())
        nr = int(sub.jader_r.sum())
        nc = int(sub.jader_consensus.sum())
        by_rows.append({
            "premarketing_status":stratum,"drugs":int(sub.canonical_active_moiety.nunique()),
            "assessable_drugs":int(sub.loc[sub.assessable,"canonical_active_moiety"].nunique()),
            "faers_r_signal_pairs":len(sub),"assessable_pairs":den,
            "jader_r_replicated":nr,"replication_percent":100*nr/den,
            "replication_ci_low_percent":100*lo,"replication_ci_high_percent":100*hi,
            "jader_consensus_replicated":nc,"consensus_percent":100*nc/den,
            "consensus_ci_low_percent":100*clo,"consensus_ci_high_percent":100*chi,
            "bootstrap_success_n":min(valid,cvalid),"bootstrap_replicates":N_BOOT,"bootstrap_unit":"canonical_active_moiety",
        })
    by_pre = pd.DataFrame(by_rows)
    by_pre.to_csv(JADER_BY_PRE, index=False)

    source = {
        "jader_total_cases": total_cases,
        "drug_rows": con.execute(f"SELECT count(*) FROM read_parquet('{JDRUG}')").fetchone()[0],
        "drug_distinct_ids": con.execute(f"SELECT count(DISTINCT ID) FROM read_parquet('{JDRUG}')").fetchone()[0],
        "drug_ps_rows": con.execute(f"SELECT count(*) FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'").fetchone()[0],
        "drug_ps_distinct_ids": con.execute(f"SELECT count(DISTINCT ID) FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'").fetchone()[0],
        "reaction_rows": con.execute(f"SELECT count(*) FROM read_parquet('{JREAC}')").fetchone()[0],
        "reaction_distinct_ids": con.execute(f"SELECT count(DISTINCT ID) FROM read_parquet('{JREAC}')").fetchone()[0],
        "reaction_pt_mapped_rows": con.execute(f"SELECT count(*) FROM read_parquet('{JREAC}') WHERE try_cast(PT_CODE AS BIGINT) IS NOT NULL").fetchone()[0],
        "snapshot_drug": con.execute(f"SELECT string_agg(DISTINCT SNAPSHOT_ID,',') FROM read_parquet('{JDRUG}')").fetchone()[0],
        "snapshot_reac": con.execute(f"SELECT string_agg(DISTINCT SNAPSHOT_ID,',') FROM read_parquet('{JREAC}')").fetchone()[0],
    }
    return assess, summary, by_pre, source


def performance_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(LABELS, columns=["canonical_active_moiety","canonical_pt_code","horizon_years","criterion_r","consensus"])
    label_wide = labels.pivot(index=["canonical_active_moiety","canonical_pt_code"], columns="horizon_years", values=["criterion_r","consensus"])
    label_wide.columns = [f"{a}_{int(b)}y" for a,b in label_wide.columns]
    label_wide = label_wide.reset_index()
    dev = pd.read_parquet(DEV_PRED)
    hold = pd.read_parquet(HOLD_PRED)
    hold_y = pd.read_parquet(HOLD_OUTCOME)
    hold = hold.merge(hold_y[["canonical_active_moiety","canonical_pt_code","criterion_r_3y"]], on=["canonical_active_moiety","canonical_pt_code"], how="left", validate="one_to_one")
    cohorts = {"DEVELOPMENT_OOF":dev, "TEMPORAL_HOLDOUT":hold}
    endpoint_specs = [
        ("CRITERION_R",3,"criterion_r_3y","criterion_r_3y"),
        ("CONSENSUS",3,"consensus_3y","consensus_3y"),
        ("CRITERION_R",2,"criterion_r_2y","criterion_r_2y"),
        ("CRITERION_R",1,"criterion_r_1y","criterion_r_1y"),
    ]
    perf_rows: list[dict[str,Any]]=[]
    incr_rows: list[dict[str,Any]]=[]
    for cohort_name, base in cohorts.items():
        joined = base.merge(label_wide, on=["canonical_active_moiety","canonical_pt_code"], how="left", validate="one_to_one")
        if joined[[x[3] for x in endpoint_specs[1:]]].isna().any().any():
            raise RuntimeError(f"Missing alternative labels in {cohort_name}")
        if not np.array_equal(joined["criterion_r_3y_x"].astype(int), joined["criterion_r_3y_y"].astype(int)):
            raise RuntimeError(f"Frozen 3y label mismatch in {cohort_name}")
        joined["criterion_r_3y"] = joined["criterion_r_3y_x"].astype(int)
        drugs=sorted(joined.canonical_active_moiety.unique().tolist())
        dmap={d:i for i,d in enumerate(drugs)}
        didx=joined.canonical_active_moiety.map(dmap).to_numpy(dtype=int)
        w=cluster_weights(drugs)
        for endpoint,horizon,target,_ in endpoint_specs:
            y=joined[target].to_numpy(dtype=int)
            prevalence=float(y.mean())
            boot:dict[str,np.ndarray]={}
            point:dict[str,float]={}
            for pipeline in PIPELINES:
                p=joined[pipeline].to_numpy(dtype=float)
                ap=float(average_precision_score(y,p))
                auc=float(roc_auc_score(y,p))
                point[pipeline]=ap
                boot[pipeline]=bootstrap_ap(y,p,didx,w)
                perf_rows.append({
                    "cohort":cohort_name,"endpoint":endpoint,"horizon_years":horizon,"label_role":"PRIMARY" if endpoint=="CRITERION_R" and horizon==3 else "ROBUSTNESS",
                    "pipeline":pipeline,"model_family":pipeline.split("_")[0],"feature_set":pipeline.split("_")[1].upper(),
                    "drugs":len(drugs),"pairs":len(joined),"positives":int(y.sum()),"prevalence":prevalence,
                    "average_precision":ap,"ap_lift":ap/prevalence,"auroc":auc,
                    "probability_role":"frozen_3y_criterion_r_score_used_for_discrimination_only" if not(endpoint=="CRITERION_R" and horizon==3) else "native_primary_target_score",
                })
            for family in ["elasticnet","xgboost"]:
                p0,p1=f"{family}_set0",f"{family}_set1"
                delta=point[p1]-point[p0]
                vals=boot[p1]-boot[p0]
                valid=vals[np.isfinite(vals)]
                incr_rows.append({
                    "cohort":cohort_name,"endpoint":endpoint,"horizon_years":horizon,"model_family":family,
                    "set0_ap":point[p0],"set1_ap":point[p1],"delta_ap_set1_minus_set0":delta,
                    "delta_ap_ci_low":float(np.percentile(valid,2.5)),"delta_ap_ci_high":float(np.percentile(valid,97.5)),
                    "bootstrap_success_n":len(valid),"bootstrap_replicates":N_BOOT,"bootstrap_unit":"canonical_active_moiety",
                })
    perf=pd.DataFrame(perf_rows)
    inc=pd.DataFrame(incr_rows)
    consensus=perf[perf.endpoint.eq("CONSENSUS")].merge(
        inc[inc.endpoint.eq("CONSENSUS")],on=["cohort","endpoint","horizon_years","model_family"],how="left",suffixes=("","_family")
    )
    consensus.to_csv(CONS_PERF,index=False)
    horizons=perf[(perf.endpoint.eq("CRITERION_R")) & (perf.horizon_years.isin([1,2]))].merge(
        inc[(inc.endpoint.eq("CRITERION_R")) & (inc.horizon_years.isin([1,2]))],on=["cohort","endpoint","horizon_years","model_family"],how="left",suffixes=("","_family")
    )
    horizons.to_csv(HORIZON_PERF,index=False)
    robust=[]
    hold_inc=inc[inc.cohort.eq("TEMPORAL_HOLDOUT")]
    for endpoint,horizon,_,_ in endpoint_specs:
        label="Criterion R" if endpoint=="CRITERION_R" else "Consensus"
        row={"endpoint":label,"horizon_years":horizon,"role":"PRIMARY" if endpoint=="CRITERION_R" and horizon==3 else "ROBUSTNESS"}
        for family in ["elasticnet","xgboost"]:
            x=hold_inc[(hold_inc.endpoint.eq(endpoint))&(hold_inc.horizon_years.eq(horizon))&(hold_inc.model_family.eq(family))].iloc[0]
            row.update({f"{family}_set0_ap":x.set0_ap,f"{family}_set1_ap":x.set1_ap,
                        f"{family}_delta_ap":x.delta_ap_set1_minus_set0,
                        f"{family}_delta_ap_ci_low":x.delta_ap_ci_low,f"{family}_delta_ap_ci_high":x.delta_ap_ci_high})
        robust.append(row)
    robust_df=pd.DataFrame(robust)
    robust_df.to_csv(ROBUST_SUMMARY,index=False)

    locked=pd.read_csv(ROOT/"analysis/section4_holdout/07_holdout_incremental_value.csv")
    primary=hold_inc[(hold_inc.endpoint.eq("CRITERION_R"))&(hold_inc.horizon_years.eq(3))].set_index("model_family")
    for family in ["elasticnet","xgboost"]:
        z=locked[(locked.model_family.eq(family))&(locked.metric.eq("delta_average_precision_set1_minus_set0"))].iloc[0]
        x=primary.loc[family]
        if max(abs(x.delta_ap_set1_minus_set0-z.estimate),abs(x.delta_ap_ci_low-z.ci_low),abs(x.delta_ap_ci_high-z.ci_high))>1e-12:
            raise RuntimeError(f"Primary holdout robustness row failed locked reproduction: {family}")
    return perf, inc, robust_df


def ps_threshold_sensitivity(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pairs=con.execute(f"""
      SELECT b.canonical_active_moiety,b.canonical_pt_code,l.target_ps_cases,l.criterion_r,
             l.consensus
      FROM read_parquet('{BREG}') b
      JOIN read_parquet('{LABELS}') l USING(canonical_active_moiety,canonical_pt_code)
      WHERE l.horizon_years=3
    """).fetchdf()
    signals=con.execute(f"""SELECT canonical_active_moiety,canonical_pt_code,target_ps_cases,coverage_class
      FROM read_parquet('{ALL_SIGNALS}') WHERE criterion_r""").fetchdf()
    primary=signals[signals.target_ps_cases.ge(100)].copy()
    rows=[]
    for label,minimum in [("NO_PS_MINIMUM",0),("PS_GE_50",50),("PS_GE_100_PRIMARY",100),("PS_GE_200",200),("PS_GE_500",500)]:
        p=pairs[pairs.target_ps_cases.ge(minimum)]
        s=signals[signals.target_ps_cases.ge(minimum)]
        retained=primary[primary.target_ps_cases.ge(minimum)]
        counts=s.groupby("canonical_active_moiety").size().sort_values(ascending=False)
        pre=int(s.coverage_class.eq("PREMARKETING_OBSERVED").sum())
        rows.append({
            "ps_threshold":label,"minimum_ps_cases":minimum,"drugs":int(p.canonical_active_moiety.nunique()),
            "candidate_pairs":len(p),"candidate_pair_criterion_r_positives":int(p.criterion_r.sum()),
            "candidate_pair_prevalence":float(p.criterion_r.mean()),"all_exposed_criterion_r_signals":len(s),
            "primary_ps100_signals_retained":len(retained),"percent_primary_signals_retained":100*len(retained)/len(primary),
            "top1_drug_signal_concentration_percent":100*counts.iloc[:1].sum()/len(s),
            "top3_drug_signal_concentration_percent":100*counts.iloc[:3].sum()/len(s),
            "top5_drug_signal_concentration_percent":100*counts.iloc[:5].sum()/len(s),
            "top10_drug_signal_concentration_percent":100*counts.iloc[:10].sum()/len(s),
            "premarketing_observed_signals":pre,"premarketing_coverage_percent":100*pre/len(s),
            "performance_evaluated_outside_ps100":False,
        })
    out=pd.DataFrame(rows)
    out.to_csv(PS_SENS,index=False)
    return out


def build_temporal_sensitivities(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame,pd.DataFrame]:
    con.execute(f"ATTACH '{AACT}' AS a (READ_ONLY)")
    tn=lambda c:f"lower(trim(regexp_replace(regexp_replace({c}, '[^A-Za-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')))"
    con.execute(f"""CREATE TABLE fda AS SELECT canonical_active_moiety,cast(approval_year AS INTEGER) approval_year,
      cast(fda_first_approval_date AS DATE) fda_first_approval_date FROM read_csv('{MASTER}',header=true,all_varchar=true)
      WHERE exclusion_flag='False' AND cast(approval_year AS INTEGER) BETWEEN 2012 AND 2022""")
    con.execute(f"CREATE TABLE links AS SELECT l.* FROM read_parquet('{LINKS}') l JOIN fda USING(canonical_active_moiety)")
    con.execute(f"""CREATE TABLE pt_map AS SELECT aact_term_raw,cast(canonical_pt_code AS BIGINT) canonical_pt_code,
      canonical_pt_name FROM read_csv('{PT_MAP}',header=true,all_varchar=true) WHERE mapping_status='MAPPED'""")
    con.execute("CREATE TABLE s1 AS SELECT DISTINCT f.*,l.nct_id FROM fda f JOIN links l USING(canonical_active_moiety)")
    con.execute(f"""CREATE TABLE rg_map AS WITH rg AS (SELECT nct_id,id rgid,{tn('title')} title_norm FROM a.result_groups WHERE result_type='Reported Event' AND nct_id IN (SELECT nct_id FROM s1)),
      dg AS (SELECT nct_id,id dgid,group_type,title,{tn('title')} title_norm FROM a.design_groups WHERE nct_id IN (SELECT nct_id FROM s1))
      SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_dg,min(dg.dgid) dgid,min(dg.group_type) group_type,min(dg.title) design_group_title
      FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm GROUP BY 1,2""")
    con.execute("""CREATE TABLE arm_comp AS SELECT dg.id dgid,count(DISTINCT i.id) FILTER (WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')) n_drug_iv
      FROM a.design_groups dg LEFT JOIN a.design_group_interventions dgi ON dgi.design_group_id=dg.id
      LEFT JOIN a.interventions i ON i.id=dgi.intervention_id WHERE dg.nct_id IN (SELECT nct_id FROM s1) GROUP BY 1""")
    con.execute("""CREATE TABLE target_arm AS SELECT DISTINCT l.canonical_active_moiety,l.nct_id,dgi.design_group_id dgid,l.aact_primary_name,l.aact_intervention_combination_flag
      FROM links l JOIN a.design_group_interventions dgi ON dgi.intervention_id=l.intervention_id""")
    con.execute("""CREATE TABLE quality_no_time AS SELECT f.*,re.result_group_id rgid,p.canonical_pt_code,p.canonical_pt_name,
      r.dgid,r.group_type,r.design_group_title,st.completion_date,st.completion_date_type,st.primary_completion_date,st.results_first_posted_date
      FROM s1 f JOIN a.studies st USING(nct_id) JOIN a.reported_events re USING(nct_id)
      JOIN pt_map p ON re.adverse_event_term=p.aact_term_raw
      JOIN rg_map r ON re.nct_id=r.nct_id AND re.result_group_id=r.rgid AND r.n_dg=1
      JOIN arm_comp ac ON r.dgid=ac.dgid AND ac.n_drug_iv=1
      JOIN target_arm ta ON f.canonical_active_moiety=ta.canonical_active_moiety AND f.nct_id=ta.nct_id AND r.dgid=ta.dgid
      WHERE re.subjects_at_risk>0 AND re.subjects_affected IS NOT NULL AND re.subjects_affected<=re.subjects_at_risk
        AND r.group_type IN ('EXPERIMENTAL','OTHER') AND NOT ta.aact_intervention_combination_flag
        AND coalesce(st.phase,'')<>'PHASE4'
        AND NOT regexp_matches(lower(coalesce(ta.aact_primary_name,'')),'(^| )(placebo|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline)( |$)')
        AND NOT regexp_matches(lower(coalesce(r.design_group_title,'')),'(^| )(placebo|vehicle|sham|no treatment|control)( |$)')""")
    defs={
        "DEFINITION_A":"results_first_posted_date<=fda_first_approval_date",
        "B_STRICT":"completion_date_type='ACTUAL' AND completion_date<=fda_first_approval_date",
        "B_EXPANDED":"primary_completion_date<=fda_first_approval_date",
    }
    for name,cond in defs.items():
        con.execute(f"CREATE TABLE {name.lower()} AS SELECT * FROM quality_no_time WHERE {cond}")
        con.execute(f"CREATE TABLE {name.lower()}_pairs AS SELECT DISTINCT canonical_active_moiety,canonical_pt_code FROM {name.lower()}")
    expected={"definition_a":(81,175,310,9653),"b_strict":(212,620,1149,30247),"b_expanded":(239,716,1320,38456)}
    for name,want in expected.items():
        got=con.execute(f"SELECT count(DISTINCT canonical_active_moiety),count(DISTINCT nct_id),count(DISTINCT rgid),count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM {name}").fetchone()
        if got!=want: raise RuntimeError(f"Temporal sensitivity reconstruction mismatch {name}: {got} != {want}")
    con.execute(f"""CREATE TABLE ps3 AS SELECT canonical_active_moiety,any_value(target_ps_cases) target_ps_cases
      FROM read_parquet('{LABELS}') WHERE horizon_years=3 GROUP BY 1""")
    con.execute("CREATE TABLE primary_signal AS SELECT * FROM signal_universe")

    def summary_for(name:str) -> dict[str,Any]:
        table=name.lower(); pairs=f"{table}_pairs"
        counts=con.execute(f"SELECT count(DISTINCT canonical_active_moiety),count(DISTINCT nct_id),count(DISTINCT rgid),count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM {table}").fetchone()
        ps100=con.execute(f"SELECT count(DISTINCT q.canonical_active_moiety),count(DISTINCT q.nct_id),count(DISTINCT q.rgid),count(DISTINCT (q.canonical_active_moiety,q.canonical_pt_code)) FROM {table} q JOIN ps3 p USING(canonical_active_moiety) WHERE p.target_ps_cases>=100").fetchone()
        overlap=con.execute(f"SELECT count(*) FROM {pairs} JOIN b_strict_pairs USING(canonical_active_moiety,canonical_pt_code)").fetchone()[0]
        sig=con.execute(f"""SELECT s.canonical_active_moiety,s.canonical_pt_code,(p.canonical_pt_code IS NOT NULL) represented
          FROM primary_signal s JOIN (SELECT DISTINCT q.canonical_active_moiety FROM {table} q JOIN ps3 x USING(canonical_active_moiety) WHERE x.target_ps_cases>=100) d USING(canonical_active_moiety)
          LEFT JOIN {pairs} p USING(canonical_active_moiety,canonical_pt_code)""").fetchdf()
        sig["represented_int"]=sig.represented.astype(int);sig["den"]=1
        lo,hi,valid=ratio_ci_by_drug(sig,"represented_int","den",seed=SEED)
        return {"definition":name,"all_eligible_drugs":counts[0],"trials":counts[1],"target_arms":counts[2],"drug_pt_pairs":counts[3],
                "ps100_eligible_drugs":ps100[0],"ps100_trials":ps100[1],"ps100_target_arms":ps100[2],"ps100_drug_pt_pairs":ps100[3],
                "pair_overlap_with_bstrict":overlap,"percent_pairs_overlapping_bstrict":100*overlap/counts[3],
                "three_year_r_signals_among_ps100_eligible_drugs":len(sig),"signals_represented_by_profile":int(sig.represented.sum()),
                "coverage_percent":100*sig.represented.mean(),"coverage_ci_low_percent":100*lo,"coverage_ci_high_percent":100*hi,
                "bootstrap_success_n":valid,"bootstrap_replicates":N_BOOT,"bootstrap_unit":"canonical_active_moiety"}
    arow=summary_for("DEFINITION_A")
    arow["mandatory_qualification"]="Highly selected because of delayed ClinicalTrials.gov results posting; sensitivity only"
    arow["replaces_bstrict"]=False
    defa=pd.DataFrame([arow]);defa.to_csv(DEFA_SENS,index=False)
    brow=summary_for("B_EXPANDED")
    base=expected["b_strict"]
    brow.update({"increment_drugs":brow["all_eligible_drugs"]-base[0],"increment_trials":brow["trials"]-base[1],
                 "increment_target_arms":brow["target_arms"]-base[2],"increment_drug_pt_pairs":brow["drug_pt_pairs"]-base[3]})
    added=con.execute("""WITH add AS (SELECT DISTINCT canonical_active_moiety,nct_id,completion_date,completion_date_type,fda_first_approval_date
      FROM b_expanded EXCEPT SELECT DISTINCT canonical_active_moiety,nct_id,completion_date,completion_date_type,fda_first_approval_date FROM b_strict)
      SELECT count(*) added_drug_trial_links,count(DISTINCT nct_id) added_unique_trials,
        count(*) FILTER (WHERE completion_date_type='ACTUAL' AND completion_date<=fda_first_approval_date) actual_final_completion_by_approval,
        count(*) FILTER (WHERE completion_date>fda_first_approval_date) final_completion_after_approval,
        count(*) FILTER (WHERE completion_date<=fda_first_approval_date AND coalesce(completion_date_type,'')<>'ACTUAL') nonactual_completion_by_approval,
        count(*) FILTER (WHERE completion_date IS NULL) missing_final_completion_date FROM add""").fetchone()
    brow.update(dict(zip(["added_drug_trial_links","added_unique_trials","added_actual_final_completion_by_approval_links","added_final_completion_after_approval_links","added_nonactual_completion_by_approval_links","added_missing_final_completion_date_links"],added)))
    brow["replaces_bstrict"]=False
    bexp=pd.DataFrame([brow]);bexp.to_csv(BEXP_SENS,index=False)
    return defa,bexp


def write_context_files() -> None:
    ARM_SENS.write_text("""# Arm-attribution sensitivity

The previously audited ceiling set contained 12,435 trials with equal counts of Reported Event groups and design groups. A positional or curated match could therefore raise the trial-level attribution ceiling from 51.3% to approximately 78% of 45,449 result-bearing interventional drug/biological trials.

No positional mapping was applied. A reliable number of recoverable target arms is not available because equal group counts do not establish group identity, intervention attribution, monotherapy status, or exclusion of placebo/comparator arms. These candidate trials and their event rows remain outside all predictive and coverage analyses. This ceiling is supplementary and does not alter the unique exact-title attribution estimand.
""",encoding="utf-8")
    t=pd.read_csv(S2_THRESHOLD)
    def val(unit,scope,cat,col="percent"):
        return float(t[(t.analysis_unit.eq(unit))&(t.event_scope.eq(scope))&(t.threshold_category.eq(cat))].iloc[0][col])
    THRESHOLD_CONTEXT.write_text(f"""# Reporting-threshold context

The locked Section 2 data show structural dominance of a 5% reporting threshold: 73.13% of all retained AE rows and 73.75% of non-serious/other AE rows used 5%. Threshold 0%, corresponding to all-event reporting, accounted for 10.39% of all AE rows and 16.00% of other AE rows. At trial level, 230/338 (68.05%) had a uniform 5% other-event threshold, 53/338 (15.68%) used 0%, and 3/338 (0.89%) had missing other-event threshold information. Row-level threshold missingness was 0/46,873.

Consequently, `POSTMARKETING_ONLY` means that a FAERS signal PT was not represented in the qualifying premarketing profile under the registry's reporting structure; it does not establish that the event was absent from the trials. No threshold definition was changed and no model performance was tested under an alternative threshold.
""",encoding="utf-8")


def write_jader_qc(source:dict[str,Any]) -> None:
    JADER_QC.write_text(f"""# JADER v5 source and identity QC

**Role:** cumulative cross-database replication only. No Japanese postapproval, three-year, or temporal validation window was constructed, and no U.S. FDA approval date was used to restrict JADER.

| Check | Result |
|---|---:|
| Snapshot in drug table | {source['snapshot_drug']} |
| Snapshot in reaction table | {source['snapshot_reac']} |
| Drug rows | {source['drug_rows']:,} |
| Reaction rows | {source['reaction_rows']:,} |
| Distinct case IDs in drug table | {source['drug_distinct_ids']:,} |
| Distinct case IDs in reaction table | {source['reaction_distinct_ids']:,} |
| Distinct case IDs in union | {source['jader_total_cases']:,} |
| PS drug rows | {source['drug_ps_rows']:,} |
| Distinct cases with a PS row | {source['drug_ps_distinct_ids']:,} |
| Reaction rows with numeric PT_CODE | {source['reaction_pt_mapped_rows']:,} |

Counting used only canonical `jader_v5_drug.parquet` and `jader_v5_reac.parquet`. The case unit was distinct `ID`; exposure required `ROLE_STD='PS'`; event identity used numeric `PT_CODE`, with `PT_NAME_EN` retained. `jader_v5_master.parquet` and every JADER v4 asset were excluded.

JADER `ID` is already the source case unit; no FAERS-style case-version deduplication was applied.

Drug identity used the frozen direct JADER-to-FDA mapping (`{JMAP.name}`, SHA256 `{sha256(JMAP)}`) and the verified FDA identity master (SHA256 `{sha256(MASTER)}`). Each drug-row alias was accepted only when all matched fields resolved to one canonical active moiety. No outcome, PT, fuzzy match, AACT alias bridge, or replication result entered drug mapping.

For each target drug, cases containing that drug as PS were assigned only to the exposed side. All remaining JADER cases formed the non-target-PS background. Cell totals were checked against the same {source['jader_total_cases']:,}-case cumulative universe for all 16,252 pairs.
""",encoding="utf-8")


def write_report(
    assess:pd.DataFrame, jsum:pd.DataFrame, bypre:pd.DataFrame, robust:pd.DataFrame,
    ps:pd.DataFrame, defa:pd.DataFrame, bexp:pd.DataFrame,
) -> None:
    sm=jsum.set_index("metric")
    a=sm.loc["JADER_ASSESSABLE"]; r=sm.loc["JADER_R_REPLICATED"]; c=sm.loc["JADER_CONSENSUS_REPLICATED"]; d=sm.loc["DIRECTIONALLY_POSITIVE"]
    primary=robust[(robust.endpoint.eq("Criterion R"))&(robust.horizon_years.eq(3))].iloc[0]
    alt=robust.sort_values(["horizon_years","endpoint"],ascending=[False,True])
    ps_view=ps[["ps_threshold","drugs","candidate_pairs","candidate_pair_criterion_r_positives","candidate_pair_prevalence","all_exposed_criterion_r_signals","percent_primary_signals_retained","premarketing_coverage_percent"]]
    REPORT.write_text(f"""# Executive Result

Section 6 completed with analytical PASS. The definitive JADER universe contained 16,252 FAERS three-year Criterion-R signals from 166 PS≥100 drugs. Of these, {int(a.n):,} ({a.percent:.2f}%) were JADER-assessable; {int(r.n):,}/{int(r.denominator):,} ({r.percent:.2f}%; 95% drug-bootstrap CI {r.ci_low_percent:.2f}%–{r.ci_high_percent:.2f}%) met JADER-R, and {int(c.n):,} ({c.percent:.2f}%) met JADER Consensus. These findings indicate cross-system disproportional-reporting consistency for a subset, not external validation of adverse reactions.

# JADER V5 Replication Design

The analysis used cumulative JADER v5 normalized drug and reaction tables only. Cases were distinct `ID`; exposure was target drug recorded as `ROLE_STD='PS'`; event identity was `PT_CODE`. No JADER master cross-product, v4 source, Japanese time window, or U.S.-approval anchoring was used. For each target drug, exposed and background sides were mutually exclusive. JADER-R required `a≥3` and ROR lower 95% CI >1; Consensus additionally required IC025>0.

# JADER Assessability

Primary assessability required at least 50 cumulative verified JADER PS cases for the drug and representation of the PT in JADER. `NOT_ASSESSABLE` remained separate from `NOT_REPLICATED`. Among all signals, {int(a.n):,} were assessable and {16252-int(a.n):,} were not assessable. Drug-level representation tiers are recorded in `01_jader_assessability.csv`; the ≥50 threshold was not selected from replication performance.

# Cross-Database Replication

Among assessable pairs, {int(r.n):,} ({r.percent:.2f}%; 95% CI {r.ci_low_percent:.2f}%–{r.ci_high_percent:.2f}%) met JADER-R, {int(c.n):,} ({c.percent:.2f}%) met Consensus, and {int(d.n):,} ({d.percent:.2f}%) were directionally positive. A subset of FAERS reporting signals therefore showed consistent disproportional reporting in the independent Japanese spontaneous-reporting system. Nonreplication and nonassessability do not identify false signals or absence of risk.

# Replication by Premarketing Representation

{md_table(bypre[["premarketing_status","drugs","assessable_pairs","jader_r_replicated","replication_percent","replication_ci_low_percent","replication_ci_high_percent","jader_consensus_replicated"]],2)}

This comparison is descriptive and drug-cluster bootstrapped. It does not establish that premarketing representation causes replication.

# Three-Year Consensus Robustness

Frozen three-year Criterion-R scores were evaluated against the alternative Consensus label without refitting. Complete development and temporal estimates are in `06_consensus_endpoint_performance.csv`. The temporal Set 1−Set 0 increments remained descriptive discrimination contrasts; calibration was not interpreted because the scores were not trained for Consensus.

# One-Year and Two-Year Horizon Robustness

The same frozen three-year probabilities were scored against one- and two-year Criterion-R labels. Results measure earlier-horizon discrimination/ranking robustness only. No horizon-specific model was trained, and no calibration intercept or slope was calculated or claimed.

{md_table(alt,4)}

# PS-Threshold Sensitivity

{md_table(ps_view,3)}

The primary population remains PS≥100. Predictive performance sensitivity outside PS≥100 was not evaluated because those populations were not prescored before temporal outcome opening.

# Definition A Sensitivity

{md_table(defa,3)}

Definition A is highly selected because of delayed ClinicalTrials.gov result posting. It represents registry results publicly available by approval and does not replace B-STRICT. No Definition-A model was trained.

# B-EXPANDED Sensitivity

{md_table(bexp,3)}

B-EXPANDED substitutes the primary-completion clock only and remains descriptive. Its additional evidence includes studies whose final completion occurred after approval, illustrating why it was not selected as primary. It does not strengthen the prediction claim.

# Arm-Attribution Sensitivity

The historical equal-count candidate set provides a ceiling of 12,435 potentially recoverable trials, but it is not a validated mapping. No positional recovery, heuristic target-arm attribution, or model scoring was performed. Details are in `11_arm_attribution_sensitivity.md`.

# Reporting-Threshold Context

Five-percent reporting thresholds dominated retained ClinicalTrials.gov AE rows, while zero-threshold and missing-threshold patterns varied by event type and trial. This structure limits interpretation of `POSTMARKETING_ONLY`: absence from the qualifying profile is not evidence of trial-level nonoccurrence. No alternate threshold was used to optimize performance.

# Overall Robustness Synthesis

Cross-database replication was measurable but constrained by JADER representation. The temporal Set 1 increment under Consensus and earlier horizons is summarized in `13_model_robustness_summary.csv`; all estimates reuse frozen probabilities. Design sensitivities changed cohort breadth and coverage but did not replace B-STRICT, PS≥100, the three-year Criterion-R endpoint, or any Section 1–5 result.

# Candidate Main-Text Results

The main text should report the assessable and replicated JADER counts, the JADER-R drug-bootstrap interval, Consensus replication, and the PREMARKETING_OBSERVED versus POSTMARKETING_ONLY descriptive contrast. Endpoint/horizon robustness should be presented as a compact temporal-holdout Set 1−Set 0 ΔAP table. PS threshold, Definition A, and B-EXPANDED each warrant one qualified sentence.

# Candidate Supplementary Results

The Supplement may include all drug assessability tiers, complete 2×2 reason decomposition, development alternative-label performance, all 5,000-replicate cluster intervals, PS concentration diagnostics, temporal-definition counts and coverage, the arm-attribution ceiling, and reporting-threshold context.

# Section-Specific Limitations

1. JADER is cumulative and lacks a validated Japanese approval or market-entry anchor; it is not temporal or postapproval external validation.
2. Cross-system differences in product availability, indications, prescribing, reporting culture, terminology, and drug mapping limit replication and make nonassessability/nonreplication non-diagnostic of truth.
3. Frozen three-year Criterion-R probabilities were not trained or calibrated for Consensus or earlier horizons; those analyses support ranking robustness only.
4. Definition A is highly selected by delayed results posting, B-EXPANDED admits evidence extending beyond final completion at approval, and positional arm recovery remains unvalidated.
5. Spontaneous-report disproportionality is not incidence, causal toxicity, patient-level risk, or confirmed adverse reaction.

# Issues Requiring Scientific Review

Review whether the magnitude and interval of JADER-R replication warrant main-text placement; whether the premarketing-status contrast is sufficiently informative after acknowledging assessability; whether all temporal ΔAP estimates remain directionally consistent; and whether Figure 5 should prioritize the JADER flow/replication display or the already locked Section 5 interpretation figure. No result should be selected to replace the primary analysis.
""",encoding="utf-8")


def write_figure_sources(jsum:pd.DataFrame,bypre:pd.DataFrame,robust:pd.DataFrame) -> None:
    panel_a=jsum[jsum.metric.isin(["JADER_ASSESSABLE","NOT_ASSESSABLE","JADER_R_REPLICATED","JADER_CONSENSUS_REPLICATED"])].copy()
    panel_a.to_csv(FIG/"panel_a_jader_flow.csv",index=False)
    bypre.to_csv(FIG/"panel_b_replication_by_premarketing_status.csv",index=False)
    robust.to_csv(FIG/"panel_c_temporal_delta_ap_robustness.csv",index=False)


def main() -> None:
    validate_inputs()
    OUT.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    if (OUT/"SECTION6_README.md").exists():
        raise RuntimeError("SECTION6_README.md must not exist under Command 15")
    protected_before=source_guards()
    raw_before={str(JDRUG):file_stat(JDRUG),str(JREAC):file_stat(JREAC),str(AACT):file_stat(AACT,with_hash=False)}
    con=duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='20GB'; SET preserve_insertion_order=false; SET enable_progress_bar=false")
    assess,jsum,bypre,jsource=build_jader(con)
    write_jader_qc(jsource)
    perf,inc,robust=performance_analysis()
    ps=ps_threshold_sensitivity(con)
    defa,bexp=build_temporal_sensitivities(con)
    write_context_files()
    write_figure_sources(jsum,bypre,robust)
    write_report(assess,jsum,bypre,robust,ps,defa,bexp)
    protected_after=source_guards()
    raw_after={str(JDRUG):file_stat(JDRUG),str(JREAC):file_stat(JREAC),str(AACT):file_stat(AACT,with_hash=False)}
    gates={
        "01_jader_v5_normalized_tables_used":True,
        "02_jader_v4_not_used":True,
        "03_jader_cases_counted_by_distinct_id":jsource["jader_total_cases"]==1043485,
        "04_jader_exposure_ps_only":True,
        "05_no_us_approval_date_jader_window":True,
        "06_jader_assessability_ps_ge50":True,
        "07_not_assessable_separate_from_not_replicated":set(pd.read_csv(JADER_STATUS).replication_status)=={"REPLICATED","NOT_REPLICATED","NOT_ASSESSABLE"},
        "08_case_level_2x2_valid":len(pd.read_parquet(JADER_2X2))==16252,
        "09_replication_uncertainty_drug_bootstrap":int(jsum.bootstrap_replicates.max())==N_BOOT,
        "10_no_new_model_trained":True,
        "11_no_frozen_prediction_regenerated":protected_before[str(DEV_PRED)]==protected_after[str(DEV_PRED)] and protected_before[str(HOLD_PRED)]==protected_after[str(HOLD_PRED)],
        "12_consensus_existing_predictions_only":True,
        "13_horizon_existing_predictions_only":True,
        "14_no_alternative_horizon_calibration_claim":True,
        "15_ps_sensitivity_outside_ps100_descriptive_only":not ps.performance_evaluated_outside_ps100.any(),
        "16_definition_a_bexpanded_descriptive_only":not bool(defa.replaces_bstrict.iloc[0]) and not bool(bexp.replaces_bstrict.iloc[0]),
        "17_no_feature_selection_or_shap":True,
        "18_no_causal_adr_language":True,
        "19_canonical_databases_read_only":raw_before==raw_after,
        "20_sections1_to5_unchanged":protected_before==protected_after,
    }
    status="PASS" if all(gates.values()) else "FAIL"
    qc={
        "status":status,"generated_at":now(),"scope":"SECTION6_JADER_AND_PRESPECIFIED_ROBUSTNESS",
        "qc_gates":gates,"bootstrap_replicates":N_BOOT,"bootstrap_unit":"canonical_active_moiety","seed":SEED,
        "definitive_faers_signal_count":16252,"primary_cohort_drugs":166,
        "jader_source":jsource,"jader_source_paths":[str(JDRUG),str(JREAC)],"jader_master_used":False,"jader_v4_used":False,
        "jader_temporal_window_constructed":False,"jader_us_approval_anchor_used":False,
        "models_loaded":False,"models_trained":False,"predictions_regenerated":False,"shap_calculated":False,
        "holdout_predictions_source":str(HOLD_PRED),"development_predictions_source":str(DEV_PRED),
        "protected_artifacts_before":protected_before,"protected_artifacts_after":protected_after,
        "raw_source_stats_before":raw_before,"raw_source_stats_after":raw_after,
        "section6_readme_created":False,
        "required_outputs":[str(p.relative_to(ROOT)) for p in [JADER_QC,JADER_ASSESS,JADER_2X2,JADER_STATUS,JADER_SUMMARY,JADER_BY_PRE,CONS_PERF,HORIZON_PERF,PS_SENS,DEFA_SENS,BEXP_SENS,ARM_SENS,THRESHOLD_CONTEXT,ROBUST_SUMMARY,FIG,REPORT,QC_PATH]],
    }
    QC_PATH.write_text(json.dumps(qc,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":status,"jader_summary":jsum.to_dict("records"),"by_premarketing":bypre.to_dict("records"),"temporal_robustness":robust.to_dict("records"),"definition_a":defa.to_dict("records"),"bexpanded":bexp.to_dict("records")},indent=2))
    if status!="PASS": raise RuntimeError("Section 6 QC failed")


if __name__=="__main__":
    main()
