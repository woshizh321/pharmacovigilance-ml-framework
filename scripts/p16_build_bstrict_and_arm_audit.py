#!/usr/bin/env python3
"""Build FDA-anchored AACT temporal cohorts and arm-selection audit."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/project")
OUT = PROJECT / "preflight_v2"
PROC = PROJECT / "data/processed/preflight_v2"
AACT = Path("/path/to/Database/AACT/aact.duckdb")
MASTER = OUT / "drug_identity_master.csv"
LINKS = PROC / "aact_fda_intervention_links.parquet"
PT_MAP = OUT / "faers_pt_repair/aact_meddra28_term_mapping.csv"
REGISTRY = OUT / "bstrict_candidate_registry.parquet"
SUMMARY = OUT / "bstrict_candidate_registry_summary.csv"
METRICS = PROC / "bstrict_metrics.json"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def smd_cont(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt((va + vb) / 2)
    return (ma - mb) / pooled if pooled else 0.0


def smd_binary(pa: float, pb: float) -> float:
    denom = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return (pa - pb) / denom if denom else 0.0


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='16GB'; SET preserve_insertion_order=false")
    con.execute(f"ATTACH '{AACT}' AS a (READ_ONLY)")
    tn = lambda c: f"lower(trim(regexp_replace(regexp_replace({c}, '[^A-Za-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')))"

    con.execute(
        f"""
        CREATE TABLE fda AS
        SELECT canonical_active_moiety, cast(approval_year AS INTEGER) approval_year,
               cast(fda_first_approval_date AS DATE) fda_first_approval_date,
               nda_bla, orphan_designation, accelerated_approval,
               breakthrough_therapy_designation, fast_track_designation,
               priority_review, route, dosage_form
        FROM read_csv('{MASTER}', header=true, all_varchar=true)
        WHERE exclusion_flag='False' AND cast(approval_year AS INTEGER) BETWEEN 2012 AND 2022
        """
    )
    con.execute(
        f"""
        CREATE TABLE links AS
        SELECT l.* FROM read_parquet('{LINKS}') l JOIN fda USING(canonical_active_moiety)
        """
    )
    con.execute(
        f"""
        CREATE TABLE pt_map AS
        SELECT aact_term_raw, cast(canonical_pt_code AS BIGINT) canonical_pt_code,
               canonical_pt_name, mapping_level
        FROM read_csv('{PT_MAP}', header=true, all_varchar=true)
        WHERE mapping_status='MAPPED'
        """
    )
    con.execute(
        """
        CREATE TABLE s1_link_trials AS
        -- One row per FDA active moiety and trial.  Intervention-level links are
        -- retained separately in `links` for arm attribution; carrying them
        -- here would multiply every reported-event row when a trial has more
        -- than one synonymous intervention record for the same moiety.
        SELECT DISTINCT f.*, l.nct_id
        FROM fda f JOIN links l USING(canonical_active_moiety)
        """
    )
    con.execute(
        """
        CREATE TABLE s2_bstrict_trials AS
        SELECT s.*, st.completion_date, st.completion_date_type, st.primary_completion_date,
               st.results_first_posted_date, st.phase
        FROM s1_link_trials s JOIN a.studies st USING(nct_id)
        WHERE st.completion_date IS NOT NULL
          AND st.completion_date_type='ACTUAL'
          AND st.completion_date<=s.fda_first_approval_date
          AND coalesce(st.phase,'')<>'PHASE4'
        """
    )
    con.execute(
        """
        CREATE TABLE s3_ae AS
        SELECT s.*, re.id ae_id, re.result_group_id rgid, re.adverse_event_term,
               re.subjects_affected, re.subjects_at_risk, re.event_type,
               re.frequency_threshold, re.organ_system, re.vocab
        FROM s2_bstrict_trials s JOIN a.reported_events re USING(nct_id)
        """
    )
    con.execute(
        """
        CREATE TABLE s4_den AS SELECT * FROM s3_ae
        WHERE subjects_at_risk>0 AND subjects_affected IS NOT NULL
          AND subjects_affected<=subjects_at_risk
        """
    )
    con.execute(
        """
        CREATE TABLE s5_pt AS
        SELECT s.*, p.canonical_pt_code, p.canonical_pt_name, p.mapping_level
        FROM s4_den s JOIN pt_map p ON s.adverse_event_term=p.aact_term_raw
        """
    )
    con.execute(
        f"""
        CREATE TABLE rg_map AS
        WITH rg AS (
          SELECT nct_id, id rgid, {tn('title')} title_norm
          FROM a.result_groups
          WHERE result_type='Reported Event' AND nct_id IN (SELECT nct_id FROM s1_link_trials)
        ), dg AS (
          SELECT nct_id, id dgid, group_type, title, {tn('title')} title_norm
          FROM a.design_groups WHERE nct_id IN (SELECT nct_id FROM s1_link_trials)
        )
        SELECT rg.nct_id, rg.rgid, count(DISTINCT dg.dgid) n_dg,
               min(dg.dgid) dgid, min(dg.group_type) group_type,
               min(dg.title) design_group_title
        FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm
        GROUP BY rg.nct_id, rg.rgid
        """
    )
    con.execute(
        """
        CREATE TABLE arm_comp AS
        SELECT dg.id dgid,
               count(DISTINCT i.id) FILTER (WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')) n_drug_iv,
               string_agg(DISTINCT i.name, ' | ') FILTER (WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')) arm_drug_names
        FROM a.design_groups dg
        LEFT JOIN a.design_group_interventions dgi ON dgi.design_group_id=dg.id
        LEFT JOIN a.interventions i ON i.id=dgi.intervention_id
        WHERE dg.nct_id IN (SELECT nct_id FROM s1_link_trials)
        GROUP BY dg.id
        """
    )
    con.execute(
        """
        CREATE TABLE target_arm AS
        SELECT DISTINCT l.canonical_active_moiety, l.nct_id, dgi.design_group_id dgid,
               l.intervention_id, l.aact_primary_name,
               l.aact_intervention_combination_flag
        FROM links l JOIN a.design_group_interventions dgi ON dgi.intervention_id=l.intervention_id
        """
    )
    con.execute(
        """
        CREATE TABLE s6_attr AS
        SELECT s.*, r.dgid, r.group_type, r.design_group_title
        FROM s5_pt s JOIN rg_map r ON s.nct_id=r.nct_id AND s.rgid=r.rgid
        WHERE r.n_dg=1
        """
    )
    con.execute(
        """
        CREATE TABLE s7_target_mono AS
        SELECT s.*, a.n_drug_iv, a.arm_drug_names, t.intervention_id target_intervention_id,
               t.aact_primary_name target_intervention_name,
               t.aact_intervention_combination_flag
        FROM s6_attr s
        JOIN arm_comp a USING(dgid)
        JOIN target_arm t ON s.canonical_active_moiety=t.canonical_active_moiety
                         AND s.nct_id=t.nct_id AND s.dgid=t.dgid
        WHERE a.n_drug_iv=1
        """
    )
    con.execute(
        """
        CREATE TABLE s8_final AS SELECT * FROM s7_target_mono
        WHERE group_type IN ('EXPERIMENTAL','OTHER')
          AND NOT aact_intervention_combination_flag
          AND NOT regexp_matches(lower(coalesce(target_intervention_name,'')),
              '(^| )(placebo|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline)( |$)')
          AND NOT regexp_matches(lower(coalesce(design_group_title,'')),
              '(^| )(placebo|vehicle|sham|no treatment|control)( |$)')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT canonical_active_moiety, approval_year, fda_first_approval_date,
                 nda_bla, orphan_designation, accelerated_approval,
                 breakthrough_therapy_designation, fast_track_designation,
                 priority_review, route, dosage_form,
                 canonical_pt_code, any_value(canonical_pt_name) canonical_pt_name,
                 count(DISTINCT nct_id) n_trials,
                 count(DISTINCT rgid) n_arms,
                 count(*) n_ae_rows,
                 sum(subjects_affected) n_affected,
                 sum(subjects_at_risk) n_at_risk,
                 max((event_type='serious')::INT) any_serious,
                 sum((event_type='serious')::INT) serious_ae_rows,
                 min(frequency_threshold) min_frequency_threshold,
                 max(frequency_threshold) max_frequency_threshold
          FROM s8_final
          GROUP BY canonical_active_moiety, approval_year, fda_first_approval_date,
                   nda_bla, orphan_designation, accelerated_approval,
                   breakthrough_therapy_designation, fast_track_designation,
                   priority_review, route, dosage_form, canonical_pt_code
        ) TO '{REGISTRY}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    stages = [
        ("S0_FDA_SINGLE_ACTIVE", "fda"),
        ("S1_AACT_MATCHED", "s1_link_trials"),
        ("S2_BSTRICT_ACTUAL_COMPLETION", "s2_bstrict_trials"),
        ("S3_AE_RESULTS", "s3_ae"),
        ("S4_VALID_DENOMINATOR", "s4_den"),
        ("S5_MEDDRA_MAPPABLE", "s5_pt"),
        ("S6_EXACT_UNIQUE_ARM_ATTRIBUTION", "s6_attr"),
        ("S7_TARGET_MONOTHERAPY", "s7_target_mono"),
        ("S8_NONPLACEBO_TREATMENT_ARM", "s8_final"),
        ("S9_CANONICAL_DRUG_PT_PAIRS", f"read_parquet('{REGISTRY}')"),
    ]
    summary = []
    for start in (2012, 2013, 2014, 2015):
        for stage, table in stages:
            cols = {x[0] for x in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()}
            def expr(col, distinct=True):
                if col not in cols:
                    return "NULL"
                return f"count(DISTINCT {col})" if distinct else f"count({col})"
            pair_expr = (
                "count(DISTINCT (canonical_active_moiety, canonical_pt_code)) "
                "FILTER (WHERE canonical_pt_code IS NOT NULL)"
                if "canonical_pt_code" in cols else "NULL"
            )
            row = con.execute(
                f"""
                SELECT {expr('canonical_active_moiety')} active_moieties,
                       {expr('nct_id')} trials,
                       {expr('rgid')} arms,
                       {expr('ae_id',False)} ae_rows,
                       {expr('canonical_pt_code')} distinct_canonical_pts,
                       {pair_expr} drug_pt_pairs
                FROM {table} WHERE approval_year BETWEEN {start} AND 2022
                """
            ).fetchone()
            summary.append(dict(zip(
                ["active_moieties", "trials", "arms", "ae_rows", "distinct_canonical_pts", "drug_pt_pairs"], row
            )) | {"window": f"{start}-2022", "stage": stage})
    fields = ["window", "stage", "active_moieties", "trials", "arms", "ae_rows", "distinct_canonical_pts", "drug_pt_pairs"]
    write_csv(SUMMARY, summary, fields)

    # Reapply the same final quality object to the two sensitivity clocks.
    # quality_no_time uses all FDA-linked trials, with all non-temporal quality
    # restrictions identical to B-STRICT.
    con.execute(
        """
        CREATE TABLE quality_no_time AS
        SELECT f.*, re.id ae_id, re.result_group_id rgid, re.adverse_event_term,
               re.subjects_affected, re.subjects_at_risk, re.event_type,
               re.frequency_threshold, p.canonical_pt_code, p.canonical_pt_name,
               r.dgid, r.group_type, r.design_group_title,
               st.completion_date, st.completion_date_type,
               st.primary_completion_date, st.results_first_posted_date, st.phase
        FROM s1_link_trials f
        JOIN a.studies st USING(nct_id)
        JOIN a.reported_events re USING(nct_id)
        JOIN pt_map p ON re.adverse_event_term=p.aact_term_raw
        JOIN rg_map r ON re.nct_id=r.nct_id AND re.result_group_id=r.rgid AND r.n_dg=1
        JOIN arm_comp ac ON r.dgid=ac.dgid AND ac.n_drug_iv=1
        JOIN target_arm ta ON f.canonical_active_moiety=ta.canonical_active_moiety
                          AND f.nct_id=ta.nct_id AND r.dgid=ta.dgid
        WHERE re.subjects_at_risk>0 AND re.subjects_affected IS NOT NULL
          AND re.subjects_affected<=re.subjects_at_risk
          AND r.group_type IN ('EXPERIMENTAL','OTHER')
          AND NOT ta.aact_intervention_combination_flag
          AND coalesce(st.phase,'')<>'PHASE4'
          AND NOT regexp_matches(lower(coalesce(ta.aact_primary_name,'')),
              '(^| )(placebo|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline)( |$)')
          AND NOT regexp_matches(lower(coalesce(r.design_group_title,'')),
              '(^| )(placebo|vehicle|sham|no treatment|control)( |$)')
        """
    )
    temporal = []
    for start in (2012, 2013, 2014, 2015):
        defs = {
            "A_RESULTS_POSTED": "results_first_posted_date<=fda_first_approval_date",
            "B_STRICT_ACTUAL_COMPLETION": "completion_date_type='ACTUAL' AND completion_date<=fda_first_approval_date",
            "B_EXPANDED_PRIMARY_COMPLETION": "primary_completion_date<=fda_first_approval_date",
        }
        for name, cond in defs.items():
            x = con.execute(
                f"""
                SELECT count(DISTINCT canonical_active_moiety), count(DISTINCT nct_id),
                       count(DISTINCT rgid), count(*), count(DISTINCT canonical_pt_code),
                       count(DISTINCT (canonical_active_moiety,canonical_pt_code))
                FROM quality_no_time
                WHERE approval_year BETWEEN {start} AND 2022 AND {cond}
                """
            ).fetchone()
            temporal.append({"window": f"{start}-2022", "definition": name,
                             **dict(zip(["active_moieties","trials","arms","ae_rows","canonical_pts","drug_pt_pairs"], x))})

    # Arm-mapping selection audit across the original result-bearing trial base.
    trial_rows = con.execute(
        f"""
        WITH ae_trials AS (
          SELECT DISTINCT s.nct_id, s.start_date, s.phase, s.enrollment, s.number_of_arms,
                 s.results_first_posted_date
          FROM a.studies s JOIN a.reported_events re USING(nct_id)
          JOIN a.interventions i USING(nct_id)
          WHERE s.study_type='INTERVENTIONAL' AND i.intervention_type IN ('DRUG','BIOLOGICAL')
        ), group_map AS (
          WITH rg AS (SELECT nct_id,id rgid,{tn('title')} tn FROM a.result_groups
                      WHERE result_type='Reported Event'),
               dg AS (SELECT nct_id,id dgid,{tn('title')} tn FROM a.design_groups)
          SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_dg
          FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.tn=dg.tn GROUP BY 1,2
        ), gm AS (
          SELECT nct_id, count(*) n_groups, sum((n_dg=1)::INT) n_mapped FROM group_map GROUP BY 1
        ), ae AS (SELECT nct_id,count(*) ae_rows FROM a.reported_events GROUP BY 1),
        sp AS (SELECT nct_id, any_value(agency_class) FILTER (WHERE lead_or_collaborator='lead') sponsor_class
               FROM a.sponsors GROUP BY 1),
        onc AS (SELECT nct_id,max(regexp_matches(lower(name),'cancer|carcinoma|leukemia|leukaemia|lymphoma|melanoma|neoplasm|tumor|tumour|myeloma')::INT) oncology
                FROM a.conditions GROUP BY 1)
        SELECT t.nct_id, (gm.n_mapped>0) mapped_any,
               year(t.start_date) study_year, t.phase, t.enrollment, t.number_of_arms,
               (d.allocation='RANDOMIZED') randomized,
               (d.masking IS NOT NULL AND d.masking<>'NONE') masked,
               sp.sponsor_class, d.intervention_model,
               year(t.results_first_posted_date) result_posting_year,
               ae.ae_rows, coalesce(onc.oncology,0) oncology
        FROM ae_trials t JOIN gm USING(nct_id) JOIN ae USING(nct_id)
        LEFT JOIN a.designs d USING(nct_id) LEFT JOIN sp USING(nct_id) LEFT JOIN onc USING(nct_id)
        """
    ).fetchall()
    cols = [x[0] for x in con.description]
    trials = [dict(zip(cols, r)) for r in trial_rows]
    mapped = [r for r in trials if r["mapped_any"]]
    unmapped = [r for r in trials if not r["mapped_any"]]
    audit_rows = []
    for var in ("study_year", "enrollment", "number_of_arms", "result_posting_year", "ae_rows"):
        a = [float(r[var]) for r in mapped if r[var] is not None]
        b = [float(r[var]) for r in unmapped if r[var] is not None]
        audit_rows.append({"variable": var, "type": "continuous", "mapped_n": len(a), "mapped_value": sum(a)/len(a),
                           "unmapped_n": len(b), "unmapped_value": sum(b)/len(b), "smd": smd_cont(a,b)})
    for var in ("randomized", "masked", "oncology"):
        a = [bool(r[var]) for r in mapped if r[var] is not None]
        b = [bool(r[var]) for r in unmapped if r[var] is not None]
        pa, pb = sum(a)/len(a), sum(b)/len(b)
        audit_rows.append({"variable": var, "type": "binary", "mapped_n": len(a), "mapped_value": pa,
                           "unmapped_n": len(b), "unmapped_value": pb, "smd": smd_binary(pa,pb)})
    for var in ("phase", "sponsor_class", "intervention_model"):
        levels = sorted({str(r[var]) for r in trials if r[var] is not None})
        for level in levels:
            pa = sum(r[var] == level for r in mapped)/len(mapped)
            pb = sum(r[var] == level for r in unmapped)/len(unmapped)
            audit_rows.append({"variable": f"{var}:{level}", "type": "categorical_level",
                               "mapped_n": len(mapped), "mapped_value": pa,
                               "unmapped_n": len(unmapped), "unmapped_value": pb,
                               "smd": smd_binary(pa,pb)})
    write_csv(PROC / "arm_mapping_selection_metrics.csv", audit_rows)

    metrics = {
        "cascade": summary,
        "temporal_definitions": temporal,
        "arm_selection": {
            "trials_total": len(trials), "mapped_any_trials": len(mapped), "unmapped_trials": len(unmapped),
            "metrics": audit_rows,
        },
        "primary_definition": "B_STRICT_ACTUAL_COMPLETION",
        "positional_arm_recovery_used": False,
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    rows_2012 = [r for r in summary if r["window"]=="2012-2022"]
    cascade_md = "\n".join(
        f"| {r['stage']} | {r['active_moieties'] or 0:,} | {r['trials'] or 0:,} | {r['arms'] or 0:,} | {r['ae_rows'] or 0:,} | {r['distinct_canonical_pts'] or 0:,} | {r['drug_pt_pairs'] or 0:,} |"
        for r in rows_2012
    )
    temporal_md = "\n".join(
        f"| {r['window']} | {r['definition']} | {r['active_moieties']:,} | {r['trials']:,} | {r['arms']:,} | {r['drug_pt_pairs']:,} |"
        for r in temporal
    )
    max_smd = max((abs(r["smd"]) for r in audit_rows if r["smd"] is not None), default=0)
    top_smd = sorted((r for r in audit_rows if r["smd"] is not None), key=lambda r: -abs(r["smd"]))[:12]
    top_md = "\n".join(
        f"| {r['variable']} | {r['mapped_value']:.4f} | {r['unmapped_value']:.4f} | {r['smd']:+.4f} |"
        for r in top_smd
    )
    (OUT / "02_temporal_definition_reaudit.md").write_text(f"""# 02 — Temporal definition re-audit

Primary B-STRICT uses actual `completion_date <= FDA_first_approval_date`, excludes Phase 4, and retains the locked denominator, terminology, exact title attribution, target-monotherapy and non-placebo criteria. Definition A and B-EXPANDED are sensitivity descriptions only.

| Window | Definition | Active moieties | Trials | Arms | Drug–PT pairs |
|---|---|---:|---:|---:|---:|
{temporal_md}

## B-STRICT 2012–2022 quality cascade

| Stage | Active moieties | Trials | Arms | AE rows | Canonical PTs | Drug–PT pairs |
|---|---:|---:|---:|---:|---:|---:|
{cascade_md}

Estimated-only completion dates are absent from B-STRICT. B-EXPANDED does not replace the primary definition.
""", encoding="utf-8")

    (OUT / "03_arm_mapping_selection_audit.md").write_text(f"""# 03 — Arm-mapping selection audit

Comparison population: {len(trials):,} result-bearing interventional drug/biological trials. A trial is “mapped” when at least one Reported Event result group has a unique normalized-title match to a design group; positional recovery is not used.

- Mapped-any trials: {len(mapped):,}.
- No mapped result group: {len(unmapped):,}.
- Maximum absolute standardized difference across recorded levels: {max_smd:.4f}.

| Variable/level | Mapped | Unmapped | Standardized difference |
|---|---:|---:|---:|
{top_md}

Continuous summaries are means; binary/categorical summaries are proportions. Absolute SMD >0.10 is treated as evidence of meaningful selection, not as a hypothesis-test result. Full levels are in `data/processed/preflight_v2/arm_mapping_selection_metrics.csv`.
""", encoding="utf-8")
    print(json.dumps({
        "registry_rows": con.execute(f"SELECT count(*) FROM read_parquet('{REGISTRY}')").fetchone()[0],
        "bstrict_2012_final": rows_2012[-1],
        "temporal_2012": [r for r in temporal if r['window']=='2012-2022'],
        "arm_trials": len(trials), "mapped": len(mapped), "unmapped": len(unmapped), "max_abs_smd": max_smd,
    }, indent=2))


if __name__ == "__main__":
    main()
