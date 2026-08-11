#!/usr/bin/env python3
"""Rebuild the project-local latest-case FAERS case x MedDRA 28.0 PT layer.

The event identity is joined by the original SuperMaster ``pt`` text to the
audited canonical table. Legacy FAERS PT code columns are never selected.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/PDS")
RAW = Path("/path/to/Database/Faers/FAERS_SUPERMASTER_V5_1_2004-2025.parquet")
CANONICAL = PROJECT / "preflight_v2/faers_pt_repair/faers_meddra28_canonical.csv"
OUT = PROJECT / "preflight_v2/faers_pt_repair"
PROC = PROJECT / "data/processed/preflight_v2/faers_pt_repair"
LATEST = PROC / "faers_latest_cases.parquet"
CASE_PT = PROC / "faers_latest_case_pt_meddra28.parquet"
METRICS = PROC / "faers_case_pt_rebuild_metrics.json"


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        "SET threads=8; SET memory_limit='16GB'; SET preserve_insertion_order=false; "
        "SET enable_progress_bar=false"
    )
    raw = str(RAW)
    canonical = str(CANONICAL)

    # One row per report before ranking prevents the expanded drug x reaction
    # grain from influencing report selection. Versionless legacy reports use
    # report date, then primaryid, as an explicit deterministic fallback.
    con.execute(
        f"""
        COPY (
          WITH reports AS (
            SELECT caseid, primaryid, caseversion, max(fda_dt) AS fda_dt
            FROM read_parquet('{raw}')
            WHERE caseid IS NOT NULL AND primaryid IS NOT NULL
            GROUP BY caseid, primaryid, caseversion
          ), ranked AS (
            SELECT *,
                   row_number() OVER (
                     PARTITION BY caseid
                     ORDER BY (caseversion IS NOT NULL) DESC,
                              caseversion DESC NULLS LAST,
                              fda_dt DESC NULLS LAST,
                              primaryid DESC
                   ) AS rn,
                   count(caseversion) OVER (PARTITION BY caseid) AS nonnull_version_reports,
                   count(*) OVER (PARTITION BY caseid) AS report_count
            FROM reports
          )
          SELECT caseid, primaryid, caseversion, fda_dt,
                 CASE WHEN nonnull_version_reports=0
                      THEN 'VERSIONLESS_FDA_DT_PRIMARYID_FALLBACK'
                      ELSE 'MAX_CASEVERSION' END AS latest_rule,
                 report_count
          FROM ranked WHERE rn=1
        ) TO '{LATEST}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # Exact source-term join is possible because the canonical table retains
    # each original raw spelling. Only mapped terms enter the event object.
    con.execute(
        f"""
        COPY (
          WITH mapping AS (
            SELECT faers_term_raw,
                   cast(canonical_pt_code AS BIGINT) AS canonical_pt_code
            FROM read_csv('{canonical}', header=true, all_varchar=true)
            WHERE mapping_status='MAPPED' AND canonical_pt_code IS NOT NULL
          )
          SELECT r.caseid, l.primaryid, l.caseversion, l.fda_dt,
                 m.canonical_pt_code
          FROM read_parquet('{raw}') r
          JOIN read_parquet('{LATEST}') l
            ON r.caseid=l.caseid AND r.primaryid=l.primaryid
          JOIN mapping m ON r.pt=m.faers_term_raw
          GROUP BY r.caseid, l.primaryid, l.caseversion, l.fda_dt,
                   m.canonical_pt_code
        ) TO '{CASE_PT}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    source = con.execute(
        f"""
        SELECT count(DISTINCT primaryid) AS raw_primaryids,
               count(DISTINCT caseid) AS unique_nonnull_caseids,
               count(DISTINCT primaryid) FILTER (WHERE caseid IS NULL) AS null_caseid_primaryids,
               count(*) FILTER (WHERE caseid IS NULL) AS expanded_null_caseid_rows
        FROM read_parquet('{raw}')
        """
    ).fetchone()
    cross_case_primary = con.execute(
        f"""
        WITH x AS (
          SELECT primaryid, count(DISTINCT caseid) n_cases
          FROM read_parquet('{raw}')
          WHERE primaryid IS NOT NULL AND caseid IS NOT NULL
          GROUP BY primaryid HAVING count(DISTINCT caseid)>1
        )
        SELECT count(*) shared_primaryids, coalesce(sum(n_cases-1),0) excess_case_links
        FROM x
        """
    ).fetchone()
    latest = con.execute(
        f"""
        SELECT count(*) AS latest_cases,
               count(*) FILTER (WHERE latest_rule='MAX_CASEVERSION') AS versioned_latest_cases,
               count(*) FILTER (WHERE latest_rule LIKE 'VERSIONLESS%') AS versionless_fallback_cases,
               count(*) FILTER (WHERE report_count>1) AS cases_with_followup_reports,
               count(*) FILTER (WHERE fda_dt IS NULL) AS latest_cases_missing_fda_dt,
               count(DISTINCT primaryid) AS selected_primaryids
        FROM read_parquet('{LATEST}')
        """
    ).fetchone()
    events = con.execute(
        f"""
        SELECT count(*) AS case_pt_rows,
               count(DISTINCT caseid) AS cases_with_mapped_pt,
               count(DISTINCT canonical_pt_code) AS unique_pts
        FROM read_parquet('{CASE_PT}')
        """
    ).fetchone()
    unresolved = con.execute(
        f"""
        WITH mapping AS (
          SELECT faers_term_raw, mapping_status
          FROM read_csv('{canonical}', header=true, all_varchar=true)
        ), unresolved_case_term AS (
          SELECT r.caseid, r.pt
          FROM read_parquet('{raw}') r
          JOIN read_parquet('{LATEST}') l
            ON r.caseid=l.caseid AND r.primaryid=l.primaryid
          LEFT JOIN mapping m ON r.pt=m.faers_term_raw
          WHERE r.pt IS NOT NULL AND trim(r.pt)<>''
            AND (m.faers_term_raw IS NULL OR m.mapping_status<>'MAPPED')
          GROUP BY r.caseid, r.pt
        )
        SELECT count(*) AS unresolved_event_rows,
               count(DISTINCT caseid) AS unresolved_event_cases
        FROM unresolved_case_term
        """
    ).fetchone()

    qc = con.execute(
        f"""
        WITH duplicates AS (
          SELECT caseid, canonical_pt_code, count(*) n
          FROM read_parquet('{CASE_PT}') GROUP BY 1,2 HAVING count(*)>1
        ), canonical_codes AS (
          SELECT DISTINCT cast(canonical_pt_code AS BIGINT) canonical_pt_code
          FROM read_csv('{canonical}', header=true, all_varchar=true)
          WHERE mapping_status='MAPPED'
        )
        SELECT (SELECT count(*) FROM duplicates) duplicate_case_pt_keys,
               (SELECT count(*) FROM read_parquet('{CASE_PT}') e
                LEFT JOIN read_parquet('{LATEST}') l USING(caseid)
                WHERE l.caseid IS NULL) orphan_event_rows,
               (SELECT count(*) FROM read_parquet('{CASE_PT}') e
                LEFT JOIN canonical_codes c USING(canonical_pt_code)
                WHERE c.canonical_pt_code IS NULL) noncanonical_event_rows
        """
    ).fetchone()

    metrics = {
        "latest_case_rule": {
            "versioned": "maximum non-null caseversion; fda_dt then primaryid break ties",
            "all_versions_missing": "maximum fda_dt; primaryid breaks ties",
            "null_caseid": "excluded because a case-level identity cannot be established",
        },
        "raw_primaryids": source[0],
        "unique_nonnull_caseids": source[1],
        "null_caseid_primaryids_excluded": source[2],
        "expanded_null_caseid_rows": source[3],
        "primaryids_shared_across_caseids": cross_case_primary[0],
        "excess_cross_case_primaryid_links": cross_case_primary[1],
        "latest_cases": latest[0],
        "versioned_latest_cases": latest[1],
        "versionless_fallback_cases": latest[2],
        "cases_with_followup_reports": latest[3],
        "latest_cases_missing_fda_dt": latest[4],
        "selected_primaryids": latest[5],
        "case_pt_rows": events[0],
        "cases_with_mapped_pt": events[1],
        "unique_pts": events[2],
        "unresolved_event_rows": unresolved[0],
        "unresolved_event_cases": unresolved[1],
        "case_pt_coverage_pct": pct(events[1], latest[0]),
        "qc": {
            "duplicate_case_pt_keys": qc[0],
            "orphan_event_rows": qc[1],
            "noncanonical_event_rows": qc[2],
        },
        "legacy_pt_codes_used": False,
        "source_modified": False,
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if latest[0] == source[1] and all(x == 0 for x in qc) else "FAIL"
    report = f"""# FAERS latest-case × repaired PT rebuild audit

**Status:** {status}

## Latest-case rule

- Reports are first collapsed to `caseid × primaryid × caseversion` before ranking.
- Cases with a non-null version use maximum `caseversion`; `fda_dt`, then `primaryid`, deterministically break any tie.
- Cases for which every report lacks `caseversion` use maximum `fda_dt`, then maximum `primaryid`. This explicit fallback retains {latest[2]:,} legacy cases that the former `arg_max(primaryid, caseversion)` implementation silently omitted.
- Seven primary IDs ({source[3]:,} expanded rows) have null `caseid` and are excluded because no case-level identity can be established.
- {cross_case_primary[0]:,} non-null primary IDs are linked to multiple case IDs ({cross_case_primary[1]:,} excess links). All source joins therefore use the composite `caseid + primaryid`; joining on `primaryid` alone fails duplicate-key QC.

## Rebuilt object

| Quantity | Value |
|---|---:|
| Raw primary IDs | {source[0]:,} |
| Unique non-null case IDs | {source[1]:,} |
| Latest-version/fallback cases | {latest[0]:,} |
| Versioned latest cases | {latest[1]:,} |
| Versionless fallback cases | {latest[2]:,} |
| Case–PT rows | {events[0]:,} |
| Cases with ≥1 mapped PT | {events[1]:,} |
| Unique canonical PTs represented | {events[2]:,} |
| Unresolved `caseid × source term` rows | {unresolved[0]:,} |
| Cases containing an unresolved term | {unresolved[1]:,} |

## QC

| Check | Failures |
|---|---:|
| Duplicate `caseid × canonical_pt_code` keys | {qc[0]:,} |
| Case–PT rows without selected latest case | {qc[1]:,} |
| Case–PT rows outside canonical mapped-code set | {qc[2]:,} |

The derived object is `{CASE_PT}`. The raw SuperMaster was read only. No legacy FAERS PT-code field participated in event identity.
"""
    (OUT / "faers_case_event_rebuild_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, **metrics}, indent=2))


if __name__ == "__main__":
    main()
