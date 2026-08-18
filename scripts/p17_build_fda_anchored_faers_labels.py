#!/usr/bin/env python3
"""Build exact FDA-anchored 1/2/3-year FAERS case-level labels.

Drug identity is consumed only from the frozen master-derived mapping.  Event
identity is consumed only from the repaired latest-case MedDRA 28.0 layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/project")
RAW = Path("/path/to/Database/Faers/FAERS_SUPERMASTER_V5_1_2004-2025.parquet")
MASTER = PROJECT / "preflight_v2/drug_identity_master.csv"
MASTER_SHA = PROJECT / "preflight_v2/drug_identity_master.sha256"
DRUG_MAP = PROJECT / "data/processed/preflight_v2/faers_name_to_fda_identity.csv"
LATEST = PROJECT / "data/processed/preflight_v2/faers_pt_repair/faers_latest_cases.parquet"
CASE_PT = PROJECT / "data/processed/preflight_v2/faers_pt_repair/faers_latest_case_pt_meddra28.parquet"
PT_CANON = PROJECT / "preflight_v2/faers_pt_repair/faers_meddra28_canonical.csv"
REGISTRY = PROJECT / "preflight_v2/bstrict_candidate_registry.parquet"
PROC = PROJECT / "data/processed/preflight_v2"
CASE_DRUG = PROC / "faers_latest_case_drug_ps_fda.parquet"
LABELS = PROC / "faers_fda_anchored_labels_1_2_3y.parquet"
ASSESS = PROC / "faers_drug_assessability_3y.csv"
METRICS = PROC / "faers_label_metrics.json"


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    expected = MASTER_SHA.read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(MASTER.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(f"Frozen identity master hash mismatch: {observed} != {expected}")

    con = duckdb.connect()
    con.execute(
        "SET threads=8; SET memory_limit='20GB'; SET temp_directory='/private/tmp/pvml_duckdb'; "
        "SET preserve_insertion_order=false; SET enable_progress_bar=false"
    )

    # The composite caseid+primaryid join is mandatory because primaryid is not
    # globally unique across caseids in the source.  One PS case contributes at
    # most once to a canonical moiety after GROUP BY.
    con.execute(
        f"""
        COPY (
          WITH map AS (
            SELECT faers_drugname_u, canonical_active_moiety
            FROM read_csv('{DRUG_MAP}', header=true, all_varchar=true)
          ), cohort AS (
            SELECT canonical_active_moiety
            FROM read_csv('{MASTER}', header=true, all_varchar=true)
            WHERE cast(approval_year AS INTEGER) BETWEEN 2012 AND 2022
              AND exclusion_flag='False'
          )
          SELECT r.caseid, l.primaryid,
                 try_strptime(cast(l.fda_dt AS VARCHAR), '%Y%m%d')::DATE report_date,
                 m.canonical_active_moiety
          FROM read_parquet('{RAW}') r
          JOIN read_parquet('{LATEST}') l
            ON r.caseid=l.caseid AND r.primaryid=l.primaryid
          JOIN map m ON r.drugname_u=m.faers_drugname_u
          JOIN cohort c USING(canonical_active_moiety)
          WHERE r.role_cod_u='PS'
            AND try_strptime(cast(l.fda_dt AS VARCHAR), '%Y%m%d') IS NOT NULL
          GROUP BY r.caseid, l.primaryid, report_date, m.canonical_active_moiety
        ) TO '{CASE_DRUG}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    con.execute(
        f"""
        CREATE TABLE pairs AS
        SELECT r.canonical_active_moiety, r.canonical_pt_code,
               any_value(r.canonical_pt_name) canonical_pt_name,
               min(r.fda_first_approval_date) approval_date,
               min(r.approval_year) approval_year,
               any_value(r.nda_bla) nda_bla,
               any_value(r.orphan_designation) orphan_designation,
               any_value(r.accelerated_approval) accelerated_approval
        FROM read_parquet('{REGISTRY}') r
        GROUP BY r.canonical_active_moiety, r.canonical_pt_code
        """
    )
    con.execute(
        """
        CREATE TABLE pairs_h AS
        SELECT p.*, h.horizon_years,
               (p.approval_date + h.horizon_years * INTERVAL 1 YEAR)::DATE window_end
        FROM pairs p CROSS JOIN (VALUES (1),(2),(3)) h(horizon_years)
        """
    )
    con.execute(
        f"""
        CREATE TABLE cases_daily AS
        SELECT try_strptime(cast(fda_dt AS VARCHAR), '%Y%m%d')::DATE report_date,
               count(*) n_cases
        FROM read_parquet('{LATEST}')
        WHERE try_strptime(cast(fda_dt AS VARCHAR), '%Y%m%d') IS NOT NULL
        GROUP BY 1
        """
    )
    con.execute(
        f"""
        CREATE TABLE pt_daily AS
        SELECT try_strptime(cast(fda_dt AS VARCHAR), '%Y%m%d')::DATE report_date,
               canonical_pt_code, count(*) n_cases_pt
        FROM read_parquet('{CASE_PT}')
        WHERE try_strptime(cast(fda_dt AS VARCHAR), '%Y%m%d') IS NOT NULL
        GROUP BY 1,2
        """
    )
    con.execute(
        f"""
        CREATE TABLE ps_daily AS
        SELECT report_date, canonical_active_moiety, count(*) n_cases_ps
        FROM read_parquet('{CASE_DRUG}') GROUP BY 1,2
        """
    )

    # Total window cases and total PT cases are calendar-window marginals.
    con.execute(
        """
        CREATE TABLE margins AS
        SELECT ph.canonical_active_moiety, ph.canonical_pt_code, ph.horizon_years,
               coalesce(any_value(n.total_cases),0)::BIGINT total_cases,
               coalesce(any_value(e.total_pt_cases),0)::BIGINT total_pt_cases,
               coalesce(any_value(x.target_ps_cases),0)::BIGINT target_ps_cases
        FROM pairs_h ph
        LEFT JOIN (
          SELECT ph.canonical_active_moiety, ph.horizon_years, sum(d.n_cases) total_cases
          FROM (SELECT DISTINCT canonical_active_moiety, approval_date, horizon_years, window_end FROM pairs_h) ph
          JOIN cases_daily d ON d.report_date BETWEEN ph.approval_date AND ph.window_end
          GROUP BY 1,2
        ) n USING(canonical_active_moiety,horizon_years)
        LEFT JOIN (
          SELECT ph.canonical_active_moiety, ph.canonical_pt_code, ph.horizon_years,
                 sum(d.n_cases_pt) total_pt_cases
          FROM pairs_h ph JOIN pt_daily d
            ON d.canonical_pt_code=ph.canonical_pt_code
           AND d.report_date BETWEEN ph.approval_date AND ph.window_end
          GROUP BY 1,2,3
        ) e USING(canonical_active_moiety,canonical_pt_code,horizon_years)
        LEFT JOIN (
          SELECT ph.canonical_active_moiety, ph.horizon_years, sum(d.n_cases_ps) target_ps_cases
          FROM (SELECT DISTINCT canonical_active_moiety, approval_date, horizon_years, window_end FROM pairs_h) ph
          JOIN ps_daily d ON d.canonical_active_moiety=ph.canonical_active_moiety
                         AND d.report_date BETWEEN ph.approval_date AND ph.window_end
          GROUP BY 1,2
        ) x USING(canonical_active_moiety,horizon_years)
        GROUP BY 1,2,3
        """
    )

    # a is calculated from distinct latest cases that simultaneously carry the
    # target moiety as PS and the target canonical PT.  Neither expanded source
    # rows nor legacy PT-code columns participate.
    con.execute(
        f"""
        CREATE TABLE exposed_events AS
        SELECT ph.canonical_active_moiety, ph.canonical_pt_code, ph.horizon_years,
               count(DISTINCT d.caseid) a
        FROM pairs_h ph
        JOIN read_parquet('{CASE_DRUG}') d
          ON d.canonical_active_moiety=ph.canonical_active_moiety
         AND d.report_date BETWEEN ph.approval_date AND ph.window_end
        JOIN read_parquet('{CASE_PT}') e
          ON e.caseid=d.caseid AND e.primaryid=d.primaryid
         AND e.canonical_pt_code=ph.canonical_pt_code
        GROUP BY 1,2,3
        """
    )

    con.execute(
        f"""
        COPY (
          WITH cells AS (
            SELECT ph.*, m.total_cases, m.target_ps_cases, m.total_pt_cases,
                   coalesce(a.a,0)::BIGINT a,
                   (m.target_ps_cases-coalesce(a.a,0))::BIGINT b,
                   (m.total_pt_cases-coalesce(a.a,0))::BIGINT c,
                   (m.total_cases-m.target_ps_cases-m.total_pt_cases+coalesce(a.a,0))::BIGINT d
            FROM pairs_h ph JOIN margins m
              USING(canonical_active_moiety,canonical_pt_code,horizon_years)
            LEFT JOIN exposed_events a
              USING(canonical_active_moiety,canonical_pt_code,horizon_years)
          ), stats AS (
            SELECT *,
              CASE WHEN a>0 AND b>0 AND c>0 AND d>0
                   THEN (a::DOUBLE*d)/(b::DOUBLE*c) END ror,
              CASE WHEN a>0 AND b>0 AND c>0 AND d>0
                   THEN exp(ln((a::DOUBLE*d)/(b::DOUBLE*c))
                            -1.96*sqrt(1.0/a+1.0/b+1.0/c+1.0/d)) END ror_lcl95,
              log2((a+0.5)/((target_ps_cases::DOUBLE*total_pt_cases/NULLIF(total_cases,0))+0.5)) ic,
              log2((a+0.5)/((target_ps_cases::DOUBLE*total_pt_cases/NULLIF(total_cases,0))+0.5))
                -3.3*pow(a+0.5,-0.5)-2.0*pow(a+0.5,-1.5) ic025
            FROM cells
          ), pt_soc AS (
            SELECT cast(canonical_pt_code AS BIGINT) canonical_pt_code,
                   any_value(canonical_soc_name) canonical_soc_name,
                   any_value(canonical_soc_code) canonical_soc_code
            FROM read_csv('{PT_CANON}', header=true, all_varchar=true)
            WHERE mapping_status='MAPPED' GROUP BY 1
          )
          SELECT s.*, p.canonical_soc_name, p.canonical_soc_code,
                 (a>=3 AND ror_lcl95>1) criterion_r,
                 (a>=3 AND ic025>0) criterion_ic,
                 (a>=3 AND ror_lcl95>1 AND ic025>0) consensus
          FROM stats s LEFT JOIN pt_soc p USING(canonical_pt_code)
        ) TO '{LABELS}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    qc = con.execute(
        f"""
        SELECT count(*) label_rows,
               count(*) FILTER (WHERE a<0 OR b<0 OR c<0 OR d<0) negative_cells,
               count(*) FILTER (WHERE a+b+c+d<>total_cases) invalid_totals,
               count(*) FILTER (WHERE a+b<>target_ps_cases) invalid_exposed,
               count(*) FILTER (WHERE a+c<>total_pt_cases) invalid_pt_margin,
               count(DISTINCT canonical_active_moiety) drugs,
               count(DISTINCT (canonical_active_moiety,canonical_pt_code)) pairs
        FROM read_parquet('{LABELS}')
        """
    ).fetchone()
    if any(qc[i] for i in range(1,5)):
        raise RuntimeError(f"2x2 QC failed: {qc}")

    con.execute(
        f"""
        COPY (
          SELECT canonical_active_moiety, approval_year, approval_date,
                 max(target_ps_cases) ps_reports_3y,
                 count(*) candidate_pairs,
                 sum(criterion_r::INT) criterion_r_positive_pairs,
                 sum(criterion_ic::INT) criterion_ic_positive_pairs,
                 sum(consensus::INT) consensus_positive_pairs
          FROM read_parquet('{LABELS}') WHERE horizon_years=3
          GROUP BY 1,2,3 ORDER BY approval_year,canonical_active_moiety
        ) TO '{ASSESS}' (HEADER, DELIMITER ',')
        """
    )
    horizons = con.execute(
        f"""
        SELECT horizon_years, count(DISTINCT canonical_active_moiety) drugs, count(*) pairs,
               sum(criterion_r::INT) r_pos, sum(criterion_ic::INT) ic_pos,
               sum(consensus::INT) consensus_pos,
               sum((criterion_r<>criterion_ic)::INT) disagreement
        FROM read_parquet('{LABELS}') GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    metrics = {
        "identity_master_sha256": observed,
        "case_drug_ps_rows": con.execute(f"SELECT count(*) FROM read_parquet('{CASE_DRUG}')").fetchone()[0],
        "label_rows": qc[0],
        "drugs": qc[5],
        "candidate_pairs": qc[6],
        "horizons": [dict(zip(["horizon_years","drugs","pairs","criterion_r_positive","criterion_ic_positive","consensus_positive","r_ic_disagreement"], r)) for r in horizons],
        "qc": {"negative_cells": qc[1], "invalid_total_cells": qc[2], "invalid_exposed_margins": qc[3], "invalid_pt_margins": qc[4]},
        "date_interval": "inclusive FDA approval date through exact calendar anniversary; preapproval reports excluded",
        "exposure": "target canonical active moiety reported as PS in the selected latest FAERS case",
        "case_unit": "unique caseid",
        "ic025_formula": "IC - 3.3*(a+0.5)^-0.5 - 2*(a+0.5)^-1.5; IC=log2((a+0.5)/(E+0.5))",
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
