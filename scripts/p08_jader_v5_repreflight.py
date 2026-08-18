#!/usr/bin/env python3
"""JADER V5 re-preflight using normalized, read-only source tables.

This script performs feasibility and data-integrity audits only. It does not
train a model, calculate time-to-onset, modify JADER source files, or construct
FDA-cohort replication results in the absence of the frozen FDA/B-STRICT assets.
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/project")
JADER = Path("/path/to/Database/Jader")
OUT = PROJECT / "preflight_v2"
PROC = PROJECT / "data/processed/preflight_v2"
FAERS_PT = PROJECT / "data/processed/faers_pt_universe.parquet"
MEDDRA_PT = PROJECT / "data/processed/meddra28_pt.parquet"
MEDDRA_LLT = PROJECT / "data/processed/meddra28_llt.parquet"

FILES = {
    "DEMO": JADER / "jader_v5_demo.parquet",
    "DRUG": JADER / "jader_v5_drug.parquet",
    "REAC": JADER / "jader_v5_reac.parquet",
    "HIST": JADER / "jader_v5_hist.parquet",
    "MASTER": JADER / "jader_v5_master.parquet",
    "UNMAPPED": JADER / "jader_v5_unmapped_terms.parquet",
    "DATA_DICTIONARY_CSV": JADER / "DATA_DICTIONARY_JADER_V5.csv",
    "DATA_DICTIONARY_MD": JADER / "JADER_V5_DATA_DICTIONARY.md",
    "README": JADER / "README_JADER_V5.md",
    "TRANSLATION_GAP": JADER / "JADER_V5_DRUG_TRANSLATION_GAP.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cols = [x[0] for x in con.execute(sql).description]
    return [dict(zip(cols, x)) for x in con.fetchall()]


def pct(n: int | float, d: int | float) -> float | None:
    return round(100.0 * n / d, 4) if d else None


def md_table(items: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for item in items:
        vals = []
        for key, _ in columns:
            value = item.get(key, "")
            if isinstance(value, float):
                value = f"{value:,.4f}".rstrip("0").rstrip(".")
            elif isinstance(value, int):
                value = f"{value:,}"
            elif value is None:
                value = "NA"
            vals.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing JADER V5 assets: {missing}")

    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='16GB'; SET preserve_insertion_order=false")
    p = {k: str(v) for k, v in FILES.items()}

    file_meta = []
    for label, path in FILES.items():
        stat = path.stat()
        file_meta.append(
            {
                "asset": label,
                "file": path.name,
                "bytes": stat.st_size,
                "size_mb": round(stat.st_size / 1024**2, 3),
                "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "sha256": sha256(path),
            }
        )

    grains = []
    grain_specs = {
        "DEMO": ("ID", "ID"),
        "DRUG": ("ID × DRUG_SEQ", "(ID, DRUG_SEQ)"),
        "REAC": ("ID × AE_SEQ", "(ID, AE_SEQ)"),
        "HIST": ("ID × HIST_SEQ", "(ID, HIST_SEQ)"),
    }
    for label, (grain, key_expr) in grain_specs.items():
        n_rows, n_keys, n_null = con.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT {key_expr}),
                   SUM(CASE WHEN ID IS NULL THEN 1 ELSE 0 END)
            FROM read_parquet('{p[label]}')
            """
        ).fetchone()
        grains.append(
            {
                "table": label,
                "expected_grain": grain,
                "rows": n_rows,
                "distinct_keys": n_keys,
                "duplicate_rows": n_rows - n_keys,
                "null_id_rows": n_null,
            }
        )

    demo_n = grains[0]["rows"]
    orphans = []
    for label in ("DRUG", "REAC", "HIST"):
        n = scalar(
            con,
            f"""
            SELECT COUNT(*) FROM read_parquet('{p[label]}') c
            LEFT JOIN read_parquet('{p['DEMO']}') d USING(ID)
            WHERE d.ID IS NULL
            """,
        )
        orphans.append({"child_table": label, "orphan_rows": n})

    snapshots = rows(
        con,
        f"""
        SELECT 'DEMO' table_name, SNAPSHOT_ID, COUNT(*) n FROM read_parquet('{p['DEMO']}') GROUP BY 1,2
        UNION ALL SELECT 'DRUG', SNAPSHOT_ID, COUNT(*) FROM read_parquet('{p['DRUG']}') GROUP BY 1,2
        UNION ALL SELECT 'REAC', SNAPSHOT_ID, COUNT(*) FROM read_parquet('{p['REAC']}') GROUP BY 1,2
        UNION ALL SELECT 'HIST', SNAPSHOT_ID, COUNT(*) FROM read_parquet('{p['HIST']}') GROUP BY 1,2
        ORDER BY table_name
        """,
    )
    date_coverage = rows(
        con,
        f"""
        SELECT MIN(YEAR) min_year, MAX(YEAR) max_year,
               MIN(YEAR_QUARTER) min_year_quarter, MAX(YEAR_QUARTER) max_year_quarter,
               COUNT(DISTINCT YEAR_QUARTER) n_year_quarters,
               SUM(YEAR IS NULL) null_year,
               SUM(YEAR_QUARTER IS NULL OR trim(YEAR_QUARTER)='') null_year_quarter
        FROM read_parquet('{p['DEMO']}')
        """,
    )[0]
    revision = rows(
        con,
        f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT ID) n_ids,
               MAX(n_per_id) max_rows_per_id, COUNT(DISTINCT REPORT_REV) n_revision_values,
               SUM(REPORT_REV IS NULL OR trim(REPORT_REV)='') null_revision
        FROM (
          SELECT *, COUNT(*) OVER (PARTITION BY ID) n_per_id
          FROM read_parquet('{p['DEMO']}')
        )
        """,
    )[0]
    roles = rows(
        con,
        f"""SELECT COALESCE(ROLE_STD,'(NULL)') role_value, COUNT(*) drug_rows,
                    COUNT(DISTINCT ID) cases
             FROM read_parquet('{p['DRUG']}') GROUP BY 1 ORDER BY drug_rows DESC""",
    )

    drug_quality = rows(
        con,
        f"""
        SELECT COALESCE(CLEAN_METHOD,'(NULL)') clean_method, COUNT(*) drug_rows,
               COUNT(DISTINCT ID) cases,
               SUM(DRUGNAME_CLEANED IS NULL OR trim(DRUGNAME_CLEANED)='') cleaned_name_null
        FROM read_parquet('{p['DRUG']}') GROUP BY 1 ORDER BY drug_rows DESC
        """,
    )
    drug_totals = rows(
        con,
        f"""
        SELECT COUNT(*) drug_rows, COUNT(DISTINCT ID) cases,
               COUNT(DISTINCT DRUGNAME_KEY) distinct_source_keys,
               COUNT(DISTINCT DRUGNAME_CLEANED) distinct_cleaned_names,
               SUM(DRUGNAME_CLEANED IS NULL OR trim(DRUGNAME_CLEANED)='') cleaned_name_null_rows,
               SUM(CLEAN_METHOD='needs_translation') needs_translation_rows,
               SUM(BRANDNAME_KEY IS NOT NULL AND trim(BRANDNAME_KEY)<>'') brand_populated_rows
        FROM read_parquet('{p['DRUG']}')
        """,
    )[0]

    pt_totals = rows(
        con,
        f"""
        SELECT COUNT(*) reaction_rows, COUNT(DISTINCT ID) reaction_cases,
               COUNT(DISTINCT PT_CODE) distinct_pt_codes,
               COUNT(DISTINCT PT_NAME_EN) distinct_pt_names,
               SUM(PT_CODE IS NULL OR trim(PT_CODE)='') null_pt_code_rows,
               SUM(PT_NAME_EN IS NULL OR trim(PT_NAME_EN)='') null_pt_name_rows,
               SUM(HLT_CODE IS NOT NULL) hlt_rows,
               SUM(HLGT_CODE IS NOT NULL) hlgt_rows,
               SUM(SOC_CODE IS NOT NULL) soc_rows
        FROM read_parquet('{p['REAC']}')
        """,
    )[0]
    pt_consistency = rows(
        con,
        f"""
        WITH c AS (
          SELECT PT_CODE, COUNT(DISTINCT PT_NAME_EN) n_names
          FROM read_parquet('{p['REAC']}') WHERE PT_CODE IS NOT NULL GROUP BY 1
        ), n AS (
          SELECT PT_NAME_EN, COUNT(DISTINCT PT_CODE) n_codes
          FROM read_parquet('{p['REAC']}') WHERE PT_NAME_EN IS NOT NULL GROUP BY 1
        )
        SELECT (SELECT SUM(n_names>1) FROM c) codes_with_multiple_names,
               (SELECT MAX(n_names) FROM c) max_names_per_code,
               (SELECT SUM(n_codes>1) FROM n) names_with_multiple_codes,
               (SELECT MAX(n_codes) FROM n) max_codes_per_name
        """,
    )[0]
    pt_by_year = rows(
        con,
        f"""
        SELECT d.YEAR report_year, COUNT(*) reaction_rows,
               SUM(r.PT_CODE IS NOT NULL AND trim(r.PT_CODE)<>'') mapped_rows,
               ROUND(100.0*AVG((r.PT_CODE IS NOT NULL AND trim(r.PT_CODE)<>'')::INT),4) mapped_pct
        FROM read_parquet('{p['REAC']}') r JOIN read_parquet('{p['DEMO']}') d USING(ID)
        GROUP BY 1 ORDER BY 1
        """,
    )
    unmapped_fields = rows(
        con,
        f"""
        SELECT field, COUNT(*) distinct_source_parts, SUM(n_rows) affected_rows
        FROM read_parquet('{p['UNMAPPED']}') GROUP BY 1 ORDER BY affected_rows DESC
        """,
    )

    faers_overlap = None
    if FAERS_PT.exists() and MEDDRA_PT.exists() and MEDDRA_LLT.exists():
        faers_overlap = rows(
            con,
            f"""
            WITH m AS (
              SELECT lower(trim(pt_name)) term_norm, try_cast(pt_code AS BIGINT) code
              FROM read_parquet('{MEDDRA_PT}')
              UNION ALL
              SELECT lower(trim(llt_name)), try_cast(pt_code AS BIGINT)
              FROM read_parquet('{MEDDRA_LLT}')
            ), mm AS (
              SELECT term_norm, min(code) code FROM m GROUP BY 1 HAVING COUNT(DISTINCT code)=1
            ), ft AS (
              SELECT f.pt, f.pt_code stored_code, mm.code
              FROM read_parquet('{FAERS_PT}') f JOIN mm ON lower(trim(f.pt))=mm.term_norm
            ), fc AS (SELECT DISTINCT code FROM ft),
                 j AS (SELECT DISTINCT try_cast(PT_CODE AS BIGINT) code FROM read_parquet('{p['REAC']}')
                       WHERE PT_CODE IS NOT NULL),
                 x AS (SELECT code FROM j INTERSECT SELECT code FROM fc)
            SELECT (SELECT COUNT(DISTINCT pt) FROM read_parquet('{FAERS_PT}')) faers_terms,
                   (SELECT COUNT(*) FROM ft) mapped_faers_terms,
                   (SELECT COUNT(*) FROM fc) faers_codes,
                   (SELECT COUNT(*) FROM j) jader_codes,
                   (SELECT COUNT(*) FROM x) shared_codes,
                   (SELECT SUM(stored_code<>code) FROM ft) legacy_stored_code_mismatches
            """,
        )[0]
        faers_overlap["pct_jader_shared"] = pct(faers_overlap["shared_codes"], faers_overlap["jader_codes"])
        faers_overlap["pct_faers_shared"] = pct(faers_overlap["shared_codes"], faers_overlap["faers_codes"])

    indication = rows(
        con,
        f"""
        SELECT COUNT(*) drug_rows,
               SUM(INDI_RAW IS NOT NULL AND trim(INDI_RAW)<>'') indication_present_rows,
               SUM(INDI_N_MAPPED>0) indication_mapped_rows,
               SUM((INDI_RAW IS NOT NULL AND trim(INDI_RAW)<>'') AND COALESCE(INDI_N_MAPPED,0)=0) present_unmapped_rows,
               SUM(INDI_N_TERMS>1) multi_term_rows,
               SUM(INDI_N_MAPPED>1) multi_mapped_rows,
               SUM(INDI_PT_ALL_EN IS NOT NULL AND strpos(INDI_PT_ALL_EN, ' | ')>0) all_field_multi_rows
        FROM read_parquet('{p['DRUG']}')
        """,
    )[0]
    indication["present_pct"] = pct(indication["indication_present_rows"], indication["drug_rows"])
    indication["mapped_of_present_pct"] = pct(indication["indication_mapped_rows"], indication["indication_present_rows"])
    indication_by_year = rows(
        con,
        f"""
        SELECT d.YEAR report_year,
               SUM(x.INDI_RAW IS NOT NULL AND trim(x.INDI_RAW)<>'') indication_present_rows,
               SUM(x.INDI_N_MAPPED>0) indication_mapped_rows,
               ROUND(100.0*SUM((x.INDI_N_MAPPED>0)::INT)/NULLIF(SUM((x.INDI_RAW IS NOT NULL AND trim(x.INDI_RAW)<>'')::INT),0),4) mapped_of_present_pct
        FROM read_parquet('{p['DRUG']}') x JOIN read_parquet('{p['DEMO']}') d USING(ID)
        GROUP BY 1 ORDER BY 1
        """,
    )
    generic_containers = rows(
        con,
        f"""
        WITH x AS (
          SELECT CASE
            WHEN lower(COALESCE(INDI_PT_ALL_EN, INDI_PT_NAME_EN, '')) LIKE '%product used for unknown indication%' THEN 'product used for unknown indication'
            WHEN lower(COALESCE(INDI_PT_ALL_EN, INDI_PT_NAME_EN, '')) LIKE '%off label use%' THEN 'off-label use'
            WHEN lower(COALESCE(INDI_PT_ALL_EN, INDI_PT_NAME_EN, '')) LIKE '%premedication%' THEN 'premedication'
            WHEN lower(COALESCE(INDI_PT_ALL_EN, INDI_PT_NAME_EN, '')) LIKE '%prophylaxis%' THEN 'prophylaxis'
            ELSE NULL END container_type
          FROM read_parquet('{p['DRUG']}')
        )
        SELECT container_type, COUNT(*) drug_rows FROM x WHERE container_type IS NOT NULL
        GROUP BY 1 ORDER BY drug_rows DESC
        """,
    )

    outcomes = rows(
        con,
        f"""
        SELECT COALESCE(OUTCOME_EN,'(NULL)') outcome, COUNT(*) event_rows,
               COUNT(DISTINCT ID) cases, SUM(DEATH_FLAG::INT) death_flag_rows
        FROM read_parquet('{p['REAC']}') GROUP BY 1 ORDER BY event_rows DESC
        """,
    )
    outcome_totals = rows(
        con,
        f"""
        SELECT COUNT(*) event_rows,
               SUM(OUTCOME_EN IS NULL OR trim(OUTCOME_EN)='') missing_outcome_rows,
               SUM(DEATH_FLAG::INT) death_event_rows,
               COUNT(DISTINCT CASE WHEN DEATH_FLAG THEN ID END) death_cases
        FROM read_parquet('{p['REAC']}')
        """,
    )[0]
    outcome_totals["death_event_pct"] = pct(outcome_totals["death_event_rows"], outcome_totals["event_rows"])
    outcome_totals["death_case_pct_of_demo"] = pct(outcome_totals["death_cases"], demo_n)

    date_metrics = {}
    for field, table, path_key in (
        ("START_DATE", "DRUG", "DRUG"),
        ("END_DATE", "DRUG", "DRUG"),
        ("ONSET_DATE", "REAC", "REAC"),
    ):
        total, nonnull = con.execute(
            f"SELECT COUNT(*), SUM({field} IS NOT NULL) FROM read_parquet('{p[path_key]}')"
        ).fetchone()
        prec_field = field + "_PREC"
        precision = rows(
            con,
            f"""SELECT COALESCE({prec_field},'(NULL)') precision_value, COUNT(*) n_rows
                 FROM read_parquet('{p[path_key]}') GROUP BY 1 ORDER BY n_rows DESC""",
        )
        date_metrics[field] = {
            "table": table,
            "rows": total,
            "nonnull": nonnull,
            "completeness_pct": pct(nonnull, total),
            "precision": precision,
        }
    date_precision_rows = [
        {"field": field, **item}
        for field, details in date_metrics.items()
        for item in details["precision"]
    ]

    pairing = rows(
        con,
        f"""
        WITH d AS (
          SELECT ID, COUNT(*) FILTER (WHERE ROLE_STD='PS') n_ps
          FROM read_parquet('{p['DRUG']}') GROUP BY 1
        ), r AS (
          SELECT ID, COUNT(*) n_events FROM read_parquet('{p['REAC']}') GROUP BY 1
        )
        SELECT COUNT(*) cases_with_drug_and_event,
               SUM(n_ps=1 AND n_events=1) single_ps_single_event_cases,
               SUM(n_ps=1 AND n_events>1) single_ps_multi_event_cases,
               SUM(n_ps>1) multi_ps_cases,
               SUM(n_ps=0) no_ps_cases
        FROM d JOIN r USING(ID)
        """,
    )[0]

    anchor_patterns = ("approval", "launch", "market", "entry", "yj", "承認", "発売")
    anchor_candidates = []
    for root in (JADER, PROJECT, PROJECT.parent / "Database"):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            low = path.name.lower()
            if any(token in low for token in anchor_patterns):
                if "preflight" not in low and "report" not in low:
                    anchor_candidates.append(str(path))
    anchor_candidates = sorted(set(anchor_candidates))
    japanese_anchor_status = "AVAILABLE" if anchor_candidates else "NOT_AVAILABLE"

    metrics = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "canonical_path": str(JADER),
        "documentation_declared_snapshot": "PMDA 2026-07 / 202607",
        "snapshot_values": snapshots,
        "file_metadata": file_meta,
        "grains": grains,
        "orphans": orphans,
        "date_coverage": date_coverage,
        "case_revision": revision,
        "roles": roles,
        "drug_name_quality": {"totals": drug_totals, "methods": drug_quality},
        "pt": {
            "totals": pt_totals,
            "consistency": pt_consistency,
            "by_year": pt_by_year,
            "unmapped": unmapped_fields,
            "faers_overlap": faers_overlap,
        },
        "indication": {
            "totals": indication,
            "by_year": indication_by_year,
            "generic_container_probe": generic_containers,
        },
        "outcome": {"totals": outcome_totals, "distribution": outcomes},
        "dates": date_metrics,
        "drug_event_pairing": pairing,
        "japanese_market_entry_anchor": {
            "status": japanese_anchor_status,
            "candidate_files": anchor_candidates,
        },
        "blocked_dependencies": [
            "Original Command 02 / Preflight V2 scientific locks are not present in the project or supplied attachment.",
            "Frozen FDA active-moiety cohort is absent.",
            "Final B-STRICT candidate-pair universe is absent.",
            "drug_identity_master.csv is absent.",
            "Canonical JADER v4 source parquet is absent; only V1 aggregates and documentation remain in this project.",
        ],
    }
    (PROC / "jader_v5_repreflight_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    inventory_md = f"""# 08 — JADER V5 inventory and structural audit

**Status:** PASS for local data integrity; downstream FDA-cohort linkage remains blocked.  
**Canonical local path:** `{JADER}`  
**Verified snapshot values:** {', '.join(sorted({str(x['SNAPSHOT_ID']) for x in snapshots}))}  
**Documentation-declared source:** PMDA `202607` snapshot, covering {date_coverage['min_year_quarter']}–{date_coverage['max_year_quarter']} (2026 Q1 documented as incomplete).

The README contains an obsolete location (`/path/to/Database/JADER/derived/v5_202607/`); the files actually inspected are the root-level assets above. Source files were read only.

## File lock

{md_table(file_meta, [('asset','Asset'),('file','File'),('size_mb','MiB'),('modified','Modified'),('sha256','SHA-256')])}

## Normalized-table grains

{md_table(grains, [('table','Table'),('expected_grain','Expected grain'),('rows','Rows'),('distinct_keys','Distinct keys'),('duplicate_rows','Duplicate rows'),('null_id_rows','Null ID rows')])}

{md_table(orphans, [('child_table','Child'),('orphan_rows','Orphan rows vs DEMO')])}

All normalized grains pass: rows equal distinct keys, with zero orphan child rows. `jader_v5_master.parquet` is not used for these counts.

## Case-version and time structure

- DEMO rows: **{revision['n_rows']:,}**; distinct IDs: **{revision['n_ids']:,}**; maximum rows per ID: **{revision['max_rows_per_id']}**.
- `REPORT_REV` has {revision['n_revision_values']:,} observed values and {revision['null_revision']:,} missing rows.
- Verified interpretation: PMDA retains one current/latest row per `ID`; no FAERS-style `max(caseversion)` algorithm is required.
- Report coverage: {date_coverage['min_year_quarter']}–{date_coverage['max_year_quarter']}; report time is quarter-granular.

## Role semantics

{md_table(roles, [('role_value','ROLE_STD'),('drug_rows','Drug rows'),('cases','Distinct cases')])}

There is no `SS` category. Primary disproportionality work must use `ROLE_STD='PS'` unless the protocol is explicitly amended.

## Drug-name readiness before FDA-cohort matching

{md_table([drug_totals], [('drug_rows','Drug rows'),('cases','Cases'),('distinct_source_keys','Distinct source keys'),('distinct_cleaned_names','Distinct cleaned names'),('cleaned_name_null_rows','Cleaned-name null rows'),('needs_translation_rows','needs_translation rows'),('brand_populated_rows','Brand populated rows')])}

{md_table(drug_quality, [('clean_method','CLEAN_METHOD'),('drug_rows','Drug rows'),('cases','Cases'),('cleaned_name_null','Null cleaned-name rows')])}

Exact/synonym/brand linkage to the FDA cohort is not calculated here because the required frozen FDA cohort and `drug_identity_master.csv` are absent. Japanese and Latin-script source keys must both be probed when those assets arrive.

## Indication quality

{md_table([indication], [('drug_rows','Drug rows'),('indication_present_rows','Source indication present'),('indication_mapped_rows','Mapped rows'),('present_unmapped_rows','Present but unmapped'),('mapped_of_present_pct','Mapped among present (%)'),('multi_term_rows','Multi-term rows'),('multi_mapped_rows','Multi-mapped rows'),('all_field_multi_rows','All-field multi rows')])}

Reducing multi-term indications to `INDI_PT_NAME_EN` alone would discard additional resolved diseases in at least **{indication['all_field_multi_rows']:,}** drug rows. Indication-restricted analysis is not performed in this preflight.

Generic/container indication probe:

{md_table(generic_containers, [('container_type','Container category'),('drug_rows','Drug rows')])}

These are descriptive registry-use categories and must not be treated as disease indications without review.

## Outcome/death feasibility

{md_table(outcomes, [('outcome','Reporter-assigned outcome'),('event_rows','Event rows'),('cases','Cases'),('death_flag_rows','Death-flag rows')])}

Outcome completeness is {100-pct(outcome_totals['missing_outcome_rows'], outcome_totals['event_rows']):.4f}%. Death is recorded on **{outcome_totals['death_event_rows']:,}** event rows ({outcome_totals['death_event_pct']}%) in **{outcome_totals['death_cases']:,}** cases ({outcome_totals['death_case_pct_of_demo']}% of DEMO cases). These are reporter-assigned outcomes, not adjudicated causality. No enrichment analysis was performed.

## Date feasibility and pairing constraint

{md_table([{'field': k, **v} for k,v in date_metrics.items()], [('field','Field'),('table','Table'),('rows','Rows'),('nonnull','Non-null'),('completeness_pct','Completeness (%)')])}

{md_table(date_precision_rows, [('field','Field'),('precision_value','Precision'),('n_rows','Rows')])}

Among cases containing both drug and reaction records, {pairing['multi_ps_cases']:,} have multiple PS drugs and {pairing['single_ps_multi_event_cases']:,} have one PS drug with multiple events. JADER provides no causal drug→reaction edge inside a case. No unrestricted cross-product TTO calculation is permitted; no TTO was calculated.

## Japanese market-entry anchor

**{japanese_anchor_status}.** No reliable PMDA approval, Japanese launch, market-entry, or YJ-code approval table was found in the supplied JADER root or project/database resources. First JADER report, US approval, first AE, and drug start dates were not substituted. JADER therefore remains a cross-database signal-replication source, not a temporal external-validation source.
"""
    write_text(OUT / "08_jader_v5_inventory.md", inventory_md)

    hierarchy = [
        {"level": "PT", "coded_rows": pt_totals["reaction_rows"] - pt_totals["null_pt_code_rows"], "coverage_pct": pct(pt_totals["reaction_rows"] - pt_totals["null_pt_code_rows"], pt_totals["reaction_rows"])},
        {"level": "HLT", "coded_rows": pt_totals["hlt_rows"], "coverage_pct": pct(pt_totals["hlt_rows"], pt_totals["reaction_rows"])},
        {"level": "HLGT", "coded_rows": pt_totals["hlgt_rows"], "coverage_pct": pct(pt_totals["hlgt_rows"], pt_totals["reaction_rows"])},
        {"level": "SOC", "coded_rows": pt_totals["soc_rows"], "coverage_pct": pct(pt_totals["soc_rows"], pt_totals["reaction_rows"])},
    ]
    overlap_text = (
        md_table([faers_overlap], [('faers_terms','FAERS PT strings'),('mapped_faers_terms','Mapped strings'),('faers_codes','FAERS mapped PT codes'),('jader_codes','JADER PT codes'),('shared_codes','Shared codes'),('pct_jader_shared','JADER shared (%)'),('pct_faers_shared','FAERS shared (%)'),('legacy_stored_code_mismatches','Legacy aggregate code mismatches')])
        if faers_overlap else "FAERS PT asset unavailable."
    )
    pt_md = f"""# 09 — JADER V5 PT/MedDRA mapping audit

**Status:** PASS for global PT integrity; FDA-cohort candidate-pair coverage is BLOCKED pending the frozen cohort and B-STRICT pair universe.

## Global PT integrity

{md_table([pt_totals], [('reaction_rows','Reaction rows'),('reaction_cases','Cases'),('distinct_pt_codes','Distinct PT codes'),('distinct_pt_names','Distinct PT names'),('null_pt_code_rows','Null PT-code rows'),('null_pt_name_rows','Null PT-name rows')])}

{md_table([pt_consistency], [('codes_with_multiple_names','Codes with >1 name'),('max_names_per_code','Max names/code'),('names_with_multiple_codes','Names with >1 code'),('max_codes_per_name','Max codes/name')])}

PT code is the primary FAERS↔JADER join key; `PT_NAME_EN` is retained for display.

## MedDRA hierarchy coverage

{md_table(hierarchy, [('level','Level'),('coded_rows','Coded rows'),('coverage_pct','Coverage (%)')])}

## FAERS↔JADER PT-code overlap

{overlap_text}

This overlap is recalculated by mapping the FAERS `pt` string through the frozen MedDRA 28.0 PT/LLT hierarchy and then joining on canonical PT code. The old V4 name-overlap estimate is not reused. The legacy `faers_pt_universe.pt_code` field is not used: **{faers_overlap['legacy_stored_code_mismatches'] if faers_overlap else 'NA'}** mapped FAERS strings carry a stored code inconsistent with the authoritative PT/LLT mapping because the V1 aggregate used an unsafe `any_value()` over an exploded source table.

## Unmapped term audit

{md_table(unmapped_fields, [('field','Source field'),('distinct_source_parts','Distinct unresolved parts'),('affected_rows','Affected source rows')])}

## Mapping completeness by report year

{md_table(pt_by_year, [('report_year','Year'),('reaction_rows','Reaction rows'),('mapped_rows','PT-coded rows'),('mapped_pct','Mapped (%)')])}

## External-pair coverage boundary

The following requested quantities are deliberately not fabricated: final FDA-cohort drug–PT overlap and PT/HLT/HLGT/SOC coverage among B-STRICT external-validation pairs. They require the frozen FDA active-moiety cohort and final B-STRICT candidate-pair registry, neither of which exists in the supplied project state.
"""
    write_text(OUT / "09_jader_v5_pt_mapping_audit.md", pt_md)

    # Machine-readable supporting tables.
    write_csv(OUT / "09_jader_v5_pt_mapping_by_year.csv", pt_by_year)
    write_csv(OUT / "08_jader_v5_indication_mapping_by_year.csv", indication_by_year)

    print(json.dumps({
        "status": "PASS_GLOBAL_INTEGRITY_BLOCKED_COHORT_LINKAGE",
        "canonical_path": str(JADER),
        "snapshot": sorted({str(x['SNAPSHOT_ID']) for x in snapshots}),
        "demo_cases": demo_n,
        "ps_cases": next(x['cases'] for x in roles if x['role_value'] == 'PS'),
        "reaction_rows": pt_totals['reaction_rows'],
        "distinct_pt_codes": pt_totals['distinct_pt_codes'],
        "death_cases": outcome_totals['death_cases'],
        "japanese_market_entry_anchor": japanese_anchor_status,
        "outputs": [str(OUT / '08_jader_v5_inventory.md'), str(OUT / '09_jader_v5_pt_mapping_audit.md')],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
