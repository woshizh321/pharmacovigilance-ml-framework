#!/usr/bin/env python3
"""Rebuild the AACT quality-filtered PT ceiling on canonical MedDRA 28.0.

This does not claim to be B-STRICT: the required FDA first-approval dates and
frozen drug identity registry are absent. It rebuilds every approval-independent
component and records the remaining blocker without substituting a proxy date.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

from p10_faers_meddra28_repair import index_add, norm_l1, norm_l2, read_asc, resolve_index


PROJECT = Path("/path/to/project")
AACT = Path("/path/to/Database/AACT/aact.duckdb")
OUT = PROJECT / "preflight_v2/faers_pt_repair"
PROC = PROJECT / "data/processed/preflight_v2/faers_pt_repair"
MAPPING_CSV = OUT / "aact_meddra28_term_mapping.csv"
PAIR_PARQUET = PROC / "aact_quality_candidate_pairs_meddra28.parquet"
METRICS_JSON = PROC / "aact_meddra28_rebuild_metrics.json"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    pt_rows = read_asc("pt.asc")
    llt_rows = read_asc("llt.asc")
    history_rows = read_asc("meddra_history_english.asc")
    pt_name = {r[0]: r[1] for r in pt_rows}
    llt_parent = {r[0]: r[2] for r in llt_rows}

    indexes: dict[str, dict[str, set[str]]] = {
        k: defaultdict(set)
        for k in ("pt_exact", "pt_l1", "pt_l2", "llt_exact", "llt_l1", "llt_l2",
                  "hist_exact", "hist_l1", "hist_l2")
    }
    for code, name in pt_name.items():
        index_add(indexes["pt_exact"], name, code)
        index_add(indexes["pt_l1"], norm_l1(name), code)
        index_add(indexes["pt_l2"], norm_l2(name), code)
    for row in llt_rows:
        code, name, parent = row[0], row[1], row[2]
        index_add(indexes["llt_exact"], name, parent)
        index_add(indexes["llt_l1"], norm_l1(name), parent)
        index_add(indexes["llt_l2"], norm_l2(name), parent)
    for row in history_rows:
        code, term, _version, level, _currency, _action = row
        if level not in {"PT", "LLT"}:
            continue
        current = code if code in pt_name else llt_parent.get(code)
        if current is None:
            continue
        index_add(indexes["hist_exact"], term, current)
        index_add(indexes["hist_l1"], norm_l1(term), current)
        index_add(indexes["hist_l2"], norm_l2(term), current)

    cascade = [
        ("L0_PT_EXACT", "pt_exact", lambda x: x),
        ("L1_PT_CASE_WHITESPACE", "pt_l1", norm_l1),
        ("L2_PT_PUNCT_UNICODE", "pt_l2", norm_l2),
        ("L3_LLT_EXACT", "llt_exact", lambda x: x),
        ("L3_LLT_CASE_WHITESPACE", "llt_l1", norm_l1),
        ("L3_LLT_PUNCT_UNICODE", "llt_l2", norm_l2),
        ("L4_HISTORY_EXACT", "hist_exact", lambda x: x),
        ("L4_HISTORY_CASE_WHITESPACE", "hist_l1", norm_l1),
        ("L4_HISTORY_PUNCT_UNICODE", "hist_l2", norm_l2),
    ]

    def map_term(term: str) -> dict:
        for level, idx, transform in cascade:
            code, candidates = resolve_index(indexes[idx], transform(term))
            if code:
                return {
                    "canonical_pt_code": code,
                    "canonical_pt_name": pt_name[code],
                    "mapping_level": level,
                    "mapping_status": "MAPPED",
                    "candidate_codes": "",
                }
            if len(candidates) > 1:
                return {
                    "canonical_pt_code": "",
                    "canonical_pt_name": "",
                    "mapping_level": level,
                    "mapping_status": "AMBIGUOUS",
                    "candidate_codes": ";".join(candidates),
                }
        return {
            "canonical_pt_code": "",
            "canonical_pt_name": "",
            "mapping_level": "UNRESOLVED",
            "mapping_status": "UNRESOLVED",
            "candidate_codes": "",
        }

    # Read-only connection to the frozen AACT snapshot for compact source terms.
    src = duckdb.connect(str(AACT), read_only=True)
    src.execute("SET threads=8; SET memory_limit='16GB'; SET preserve_insertion_order=false")
    term_cursor = src.execute(
        """
        WITH ae_trials AS (
          SELECT DISTINCT s.nct_id
          FROM studies s
          JOIN reported_events re USING(nct_id)
          JOIN interventions i USING(nct_id)
          WHERE s.study_type='INTERVENTIONAL'
            AND i.intervention_type IN ('DRUG','BIOLOGICAL')
        )
        SELECT re.adverse_event_term,
               count(*) AS ae_rows,
               count(DISTINCT re.nct_id) AS trials,
               sum(coalesce(re.subjects_affected,0)) AS affected
        FROM reported_events re JOIN ae_trials USING(nct_id)
        WHERE re.adverse_event_term IS NOT NULL AND trim(re.adverse_event_term)<>''
        GROUP BY re.adverse_event_term ORDER BY re.adverse_event_term
        """
    )
    term_rows = term_cursor.fetchall()
    mapping_rows = []
    for term, ae_rows, trials, affected in term_rows:
        mapping_rows.append(
            {
                "aact_term_raw": term,
                "aact_term_normalized": norm_l2(term),
                **map_term(term),
                "ae_rows": ae_rows,
                "trials": trials,
                "affected": affected,
            }
        )
    fields = [
        "aact_term_raw", "aact_term_normalized", "canonical_pt_code", "canonical_pt_name",
        "mapping_level", "mapping_status", "candidate_codes", "ae_rows", "trials", "affected",
    ]
    write_csv(MAPPING_CSV, mapping_rows, fields)
    src.close()

    # Work in an in-memory database while keeping AACT explicitly read-only.
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='16GB'; SET preserve_insertion_order=false")
    con.execute(f"ATTACH '{AACT}' AS a (READ_ONLY)")
    tn = lambda c: (
        f"lower(trim(regexp_replace(regexp_replace({c}, '[^A-Za-z0-9]+', ' ', 'g'), "
        f"'\\s+', ' ', 'g')))"
    )
    # Drug-name normalization is used only for the approval-independent ceiling;
    # it is not promoted to a frozen FDA drug identity.
    n2 = lambda c: f"""regexp_replace(trim(regexp_replace(regexp_replace(regexp_replace(regexp_replace(
       lower({c}), '\\([^)]*\\)|\\[[^]]*\\]', ' ', 'g'),
       '\\b[0-9]+([.,][0-9]+)?\\s*(mg|mcg|ug|g|kg|ml|l|iu|units?|%)\\b', ' ', 'g'),
       '\\b(tablet|tablets|capsule|capsules|oral|injection|infusion|solution|iv|intravenous|subcutaneous|film|coated|extended|release|matching|hydrochloride|hcl|sodium|sulfate|sulphate|mesylate|maleate|tartrate|citrate|acetate|phosphate|succinate|fumarate|besylate|dihydrate|monohydrate)\\b', ' ', 'g'),
       '[^a-z0-9 -]+', ' ', 'g')), '\\s+', ' ', 'g')"""

    con.execute(
        f"""
        CREATE TABLE pt_map AS
        SELECT aact_term_raw, cast(canonical_pt_code AS BIGINT) canonical_pt_code,
               canonical_pt_name, mapping_level, mapping_status
        FROM read_csv('{MAPPING_CSV}', header=true, all_varchar=true)
        """
    )
    con.execute(
        """
        CREATE TABLE ae_trials AS
        SELECT DISTINCT s.nct_id, s.completion_date, s.primary_completion_date
        FROM a.studies s
        JOIN a.reported_events re USING(nct_id)
        JOIN a.interventions i USING(nct_id)
        WHERE s.study_type='INTERVENTIONAL'
          AND i.intervention_type IN ('DRUG','BIOLOGICAL')
        """
    )
    con.execute(
        f"""
        CREATE TABLE rg_map AS
        WITH rg AS (
          SELECT nct_id, id rgid, {tn('title')} title_norm
          FROM a.result_groups
          WHERE result_type='Reported Event' AND nct_id IN (SELECT nct_id FROM ae_trials)
        ), dg AS (
          SELECT nct_id, id dgid, group_type, {tn('title')} title_norm
          FROM a.design_groups WHERE nct_id IN (SELECT nct_id FROM ae_trials)
        )
        SELECT rg.nct_id, rg.rgid, count(DISTINCT dg.dgid) n_dg,
               min(dg.dgid) dgid, min(dg.group_type) group_type
        FROM rg LEFT JOIN dg
          ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm
        GROUP BY rg.nct_id, rg.rgid
        """
    )
    con.execute(
        f"""
        CREATE TABLE arm_drug AS
        SELECT m.nct_id, m.rgid, m.dgid, m.group_type,
               count(DISTINCT i.id) n_drug_iv,
               any_value(i.name) agent_raw,
               any_value({n2('i.name')}) agent_normalized
        FROM rg_map m
        JOIN a.design_group_interventions dgi ON dgi.design_group_id=m.dgid
        JOIN a.interventions i ON i.id=dgi.intervention_id
          AND i.intervention_type IN ('DRUG','BIOLOGICAL')
        WHERE m.n_dg=1
        GROUP BY m.nct_id, m.rgid, m.dgid, m.group_type
        HAVING count(DISTINCT i.id)=1
        """
    )
    con.execute(
        """
        CREATE TABLE ev AS
        SELECT re.id, re.nct_id, re.result_group_id rgid, re.adverse_event_term,
               re.event_type, re.frequency_threshold, re.subjects_affected,
               re.subjects_at_risk, re.organ_system, re.vocab
        FROM a.reported_events re JOIN ae_trials t USING(nct_id)
        """
    )

    steps: dict[str, int] = {}
    steps["S0_all_ae_rows"] = con.execute("SELECT count(*) FROM ev").fetchone()[0]
    con.execute(
        """CREATE TABLE e1 AS
           SELECT e.*, p.canonical_pt_code, p.canonical_pt_name, p.mapping_level
           FROM ev e JOIN pt_map p ON e.adverse_event_term=p.aact_term_raw
           WHERE p.mapping_status='MAPPED'"""
    )
    steps["S1_canonical_pt_mapped"] = con.execute("SELECT count(*) FROM e1").fetchone()[0]
    con.execute(
        """CREATE TABLE e2 AS SELECT * FROM e1
           WHERE subjects_at_risk>0 AND subjects_affected IS NOT NULL
             AND subjects_affected<=subjects_at_risk"""
    )
    steps["S2_valid_denominator"] = con.execute("SELECT count(*) FROM e2").fetchone()[0]
    con.execute(
        """CREATE TABLE e3 AS SELECT e.* FROM e2 e
           JOIN rg_map m ON e.nct_id=m.nct_id AND e.rgid=m.rgid
           WHERE m.n_dg=1"""
    )
    steps["S3_unique_arm_title_map"] = con.execute("SELECT count(*) FROM e3").fetchone()[0]
    con.execute(
        """CREATE TABLE e4 AS SELECT e.*, d.dgid, d.group_type,
                  d.agent_raw, d.agent_normalized
           FROM e2 e JOIN arm_drug d ON e.nct_id=d.nct_id AND e.rgid=d.rgid"""
    )
    steps["S4_monotherapy_arm"] = con.execute("SELECT count(*) FROM e4").fetchone()[0]
    con.execute(
        """CREATE TABLE e5 AS SELECT * FROM e4
           WHERE group_type IN ('EXPERIMENTAL','OTHER')
             AND agent_normalized IS NOT NULL AND length(agent_normalized)>2
             AND NOT regexp_matches(agent_normalized,
                 '(^| )(placebo|placebos|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline|normal saline)( |$)')"""
    )
    steps["S5_treatment_monotherapy_nonplacebo"] = con.execute("SELECT count(*) FROM e5").fetchone()[0]

    con.execute(
        f"""
        COPY (
          SELECT agent_normalized, canonical_pt_code,
                 any_value(agent_raw) agent_raw,
                 any_value(canonical_pt_name) canonical_pt_name,
                 count(DISTINCT nct_id) n_trials,
                 count(DISTINCT rgid) n_arms,
                 count(*) n_ae_rows,
                 sum(subjects_affected) n_affected,
                 sum(subjects_at_risk) n_at_risk,
                 max((event_type='serious')::INT) any_serious,
                 min(frequency_threshold) min_frequency_threshold
          FROM e5 GROUP BY agent_normalized, canonical_pt_code
        ) TO '{PAIR_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    ceiling = con.execute(
        f"""
        SELECT count(DISTINCT agent_normalized) drugs,
               count(DISTINCT nct_id) trials,
               count(DISTINCT rgid) arms,
               count(DISTINCT adverse_event_term) source_terms,
               count(DISTINCT canonical_pt_code) canonical_pts,
               count(DISTINCT (agent_normalized, canonical_pt_code)) drug_pt_pairs
        FROM e5
        """
    ).fetchone()
    mapped_terms = [r for r in mapping_rows if r["mapping_status"] == "MAPPED"]
    unresolved_terms = [r for r in mapping_rows if r["mapping_status"] != "MAPPED"]
    valid_code_fail = sum(str(r["canonical_pt_code"]) not in pt_name for r in mapped_terms)
    name_code_fail = sum(pt_name.get(str(r["canonical_pt_code"])) != r["canonical_pt_name"] for r in mapped_terms)
    levels = Counter(r["mapping_level"] for r in mapping_rows)
    total_ae_rows = sum(r["ae_rows"] for r in mapping_rows)
    mapped_ae_rows = sum(r["ae_rows"] for r in mapped_terms)

    metrics = {
        "status": "APPROVAL_INDEPENDENT_AACT_PT_LAYER_PASS_BSTRICT_BLOCKED",
        "aact_snapshot": "2026-05-01",
        "raw_distinct_source_terms": len(mapping_rows),
        "mapped_source_terms": len(mapped_terms),
        "unresolved_or_ambiguous_source_terms": len(unresolved_terms),
        "term_mapping_rate_pct": pct(len(mapped_terms), len(mapping_rows)),
        "ae_row_mapping_rate_pct": pct(mapped_ae_rows, total_ae_rows),
        "mapping_levels": dict(levels),
        "cascade_ae_rows": steps,
        "approval_independent_quality_ceiling": {
            "drugs_normalized_not_fda_frozen": ceiling[0],
            "trials": ceiling[1],
            "arms": ceiling[2],
            "unique_retained_source_terms": ceiling[3],
            "mapped_canonical_pts": ceiling[4],
            "drug_pt_pairs": ceiling[5],
        },
        "b_strict": {
            "status": "BLOCKED",
            "drugs": None,
            "trials": None,
            "arms": None,
            "unique_source_terms": None,
            "mapped_canonical_pts": None,
            "drug_pt_pairs": None,
            "unresolved_terms": None,
            "blockers": [
                "FDA first-approval dates absent",
                "frozen FDA cohort absent",
                "drug_identity_master.csv absent",
            ],
        },
        "qc": {"invalid_code_failures": valid_code_fail, "name_code_failures": name_code_fail},
        "legacy_faers_codes_used": False,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    cascade_rows = "\n".join(
        f"| {name} | {value:,} | {pct(value, steps['S0_all_ae_rows']):.4f}% |"
        for name, value in steps.items()
    )
    level_rows = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(levels.items()))
    report = f"""# AACT B-STRICT PT rebuild audit

**Canonical AACT terminology layer:** PASS.  
**B-STRICT status:** BLOCKED — required FDA dates and frozen drug identities are absent.

## Rebuilt MedDRA 28.0 mapping

The AACT mapping was rebuilt directly against the same local MedDRA 28.0 PT, LLT and history files used for repaired FAERS. No legacy FAERS or JADER term/code was used as a target vocabulary, and ambiguous normalized keys remain unresolved.

| Quantity | Value |
|---|---:|
| Unique AACT source AE terms | {len(mapping_rows):,} |
| Mapped source terms | {len(mapped_terms):,} ({pct(len(mapped_terms),len(mapping_rows)):.4f}%) |
| Unresolved/ambiguous source terms | {len(unresolved_terms):,} |
| AE-row-weighted mapping | {pct(mapped_ae_rows,total_ae_rows):.4f}% |
| Invalid retained PT codes | {valid_code_fail:,} |
| PT name↔code inconsistencies | {name_code_fail:,} |

| Mapping level | Source terms |
|---|---:|
{level_rows}

## Approval-independent quality cascade

The cascade enforces interventional drug/biological trials, canonical PT mapping, valid denominators, 1:1 normalized result-arm title linkage, exactly one drug/biological intervention, treatment-arm group type (`EXPERIMENTAL`/`OTHER`), and a token-aware non-placebo name filter.

| Step | AE rows | % of S0 |
|---|---:|---:|
{cascade_rows}

| Approval-independent ceiling | Value |
|---|---:|
| Normalized drug strings (not FDA-frozen identities) | {ceiling[0]:,} |
| Trials | {ceiling[1]:,} |
| Arms | {ceiling[2]:,} |
| Unique retained source AE terms | {ceiling[3]:,} |
| Mapped canonical PTs | {ceiling[4]:,} |
| Drug–PT pairs | {ceiling[5]:,} |

## B-STRICT required fields

| Required result | Value |
|---|---:|
| B-STRICT drugs | NOT COMPUTABLE |
| B-STRICT trials | NOT COMPUTABLE |
| B-STRICT arms | NOT COMPUTABLE |
| B-STRICT unique source AE terms | NOT COMPUTABLE |
| B-STRICT mapped canonical PTs | NOT COMPUTABLE |
| B-STRICT drug–PT pairs | NOT COMPUTABLE |
| B-STRICT unresolved terms | NOT COMPUTABLE |

`study_completion_date <= FDA first approval date` cannot be evaluated without the FDA date registry. An AACT completion date, trial posting date, first FAERS report date, or current normalized drug string was not substituted. The rebuilt approval-independent pair object is `{PAIR_PARQUET}` and must not be called B-STRICT.
"""
    (OUT / "aact_bstrict_pt_rebuild_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
