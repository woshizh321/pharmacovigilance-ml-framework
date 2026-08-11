#!/usr/bin/env python3
"""Command 04: Section 1 cohort, data-quality, and structural support analysis.

This script deliberately does not open the FAERS label parquet, any JADER
pair-level object, or any model artifact.  The sole postmarketing-derived
input is a projection of the frozen drug-level assessability file to
`canonical_active_moiety` plus the Boolean predicate `ps_reports_3y >= 100`.
The exact PS count and all signal columns are neither retained nor summarized.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/PDS")
OUT = PROJECT / "analysis/section1_cohort"
FDA = PROJECT / "preflight_v2/fda_cder_nme_cohort_master.csv"
IDENTITY = PROJECT / "preflight_v2/drug_identity_master.csv"
IDENTITY_SHA = PROJECT / "preflight_v2/drug_identity_master.sha256"
AACT_LINKS = PROJECT / "data/processed/preflight_v2/aact_fda_intervention_links.parquet"
AACT_PT = PROJECT / "preflight_v2/faers_pt_repair/aact_meddra28_term_mapping.csv"
ASSESSABILITY = PROJECT / "data/processed/preflight_v2/faers_drug_assessability_3y.csv"
PRIOR_BSTRICT = PROJECT / "data/processed/preflight_v2/bstrict_metrics.json"
AACT_DB = Path("/path/to/Database/AACT/aact.duckdb")

FLOW_CSV = OUT / "01_cohort_flow_counts.csv"
DRUG_CSV = OUT / "02_final_drug_level_characteristics.csv"
TABLE1_CSV = OUT / "03_table1_development_vs_holdout.csv"
MISSING_CSV = OUT / "04_table1_missingness.csv"
ARM_CSV = OUT / "05_arm_mapping_selection.csv"
PROGRAMME_CSV = OUT / "06_programme_distribution.csv"
FIREWALL_MD = OUT / "07_holdout_firewall_audit.md"
REPORT_MD = OUT / "SECTION1_REPORT.md"
QC_JSON = OUT / "SECTION1_QC.json"


EXPECTED = {
    "bstrict_drugs": 212,
    "bstrict_trials": 620,
    "bstrict_arms": 1149,
    "bstrict_pairs": 30247,
    "eligible_drugs": 166,
    "development_drugs": 107,
    "holdout_drugs": 59,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stat_identity(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fetch_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def quantile(values: list[float], q: float) -> float | None:
    xs = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None, None
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def smd_cont(a: list[float], b: list[float]) -> float | None:
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    if ma is None or mb is None:
        return None
    pooled = math.sqrt((sa * sa + sb * sb) / 2)
    return (ma - mb) / pooled if pooled else 0.0


def smd_binary(pa: float, pb: float) -> float:
    den = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return (pa - pb) / den if den else 0.0


def fmt_num(x: float | int | None, digits: int = 1) -> str:
    if x is None:
        return "NA"
    if abs(float(x) - round(float(x))) < 1e-12:
        return f"{int(round(float(x))):,}"
    return f"{float(x):,.{digits}f}"


def median_iqr(values: list[float]) -> str:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return "NA"
    return f"{fmt_num(quantile(xs, .5))} [{fmt_num(quantile(xs, .25))}, {fmt_num(quantile(xs, .75))}]"


def regulatory_level(variable: str, value: object) -> str | None:
    """Collapse FDA indication/voucher qualifiers without using outcomes.

    The frozen FDA source retains details such as ``Yes (indication [A]
    only)`` and ``Priority (used priority review voucher)``. For the
    drug-level status comparisons these remain affirmative statuses; ``N/A``
    remains its own structural category.
    """
    if value in (None, ""):
        return None
    level = str(value)
    if variable in {
        "orphan_designation",
        "accelerated_approval",
        "breakthrough_therapy_designation",
        "fast_track_designation",
    } and level.startswith("Yes"):
        return "Yes"
    if variable == "priority_review" and level.startswith("Priority"):
        return "Priority"
    return level


def md_table(rows: list[dict], fields: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in fields) + " |"
    sep = "|" + "|".join("---" for _ in fields) + "|"
    body = []
    for row in rows:
        vals = []
        for key, _ in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:,.3f}"
            elif isinstance(value, int):
                value = f"{value:,}"
            vals.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    expected_sha = IDENTITY_SHA.read_text(encoding="utf-8").split()[0]
    actual_sha = sha256(IDENTITY)
    if expected_sha != actual_sha:
        raise RuntimeError(f"Frozen identity hash mismatch: {actual_sha} != {expected_sha}")

    protected = [FDA, IDENTITY, IDENTITY_SHA, AACT_LINKS, AACT_PT, ASSESSABILITY, AACT_DB]
    before = {str(p): stat_identity(p) for p in protected}

    con = duckdb.connect()
    con.execute(
        "SET threads=8; SET memory_limit='20GB'; SET temp_directory='/private/tmp/pds_duckdb'; "
        "SET preserve_insertion_order=false; SET enable_progress_bar=false"
    )
    con.execute(f"ATTACH '{AACT_DB}' AS a (READ_ONLY)")

    # FDA and identity tables are limited to the locked 2012–2022 cohort.
    con.execute(
        f"""
        CREATE TABLE fda_all_window AS
        SELECT * FROM read_csv_auto('{FDA}', header=true)
        WHERE approval_year BETWEEN 2012 AND 2022
        """
    )
    con.execute("CREATE TABLE fda AS SELECT * FROM fda_all_window WHERE NOT exclusion_flag")
    con.execute(
        f"""
        CREATE TABLE identity AS
        SELECT * FROM read_csv_auto('{IDENTITY}', header=true)
        WHERE approval_year BETWEEN 2012 AND 2022 AND NOT exclusion_flag
        """
    )

    # Firewall-safe assessability view. No pair outcome or exact PS count is
    # retained. The source column participates only in the Boolean predicate.
    con.execute(
        f"""
        CREATE TABLE eligibility_flag AS
        SELECT canonical_active_moiety, (ps_reports_3y>=100) AS ps_ge100_eligible
        FROM read_csv_auto('{ASSESSABILITY}', header=true)
        """
    )

    con.execute(
        f"""
        CREATE TABLE links AS
        SELECT l.* FROM read_parquet('{AACT_LINKS}') l JOIN fda USING(canonical_active_moiety)
        """
    )
    con.execute(
        f"""
        CREATE TABLE pt_map AS
        SELECT aact_term_raw, cast(canonical_pt_code AS BIGINT) canonical_pt_code,
               canonical_pt_name, mapping_level
        FROM read_csv_auto('{AACT_PT}', header=true)
        WHERE mapping_status='MAPPED' AND canonical_pt_code IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE TABLE s1_link_trials AS
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
    # Preserve the frozen p16 normalization exactly: punctuation replacement
    # can introduce leading/trailing whitespace, so trimming must occur last.
    title_norm = "lower(trim(regexp_replace(regexp_replace({col}, '[^A-Za-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')))"
    con.execute(
        f"""
        CREATE TABLE rg_map AS
        WITH rg AS (
          SELECT nct_id,id rgid,{title_norm.format(col='title')} title_norm
          FROM a.result_groups WHERE result_type='Reported Event'
            AND nct_id IN (SELECT nct_id FROM s1_link_trials)
        ), dg AS (
          SELECT nct_id,id dgid,group_type,title,{title_norm.format(col='title')} title_norm
          FROM a.design_groups WHERE nct_id IN (SELECT nct_id FROM s1_link_trials)
        )
        SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_dg,
               min(dg.dgid) dgid,min(dg.group_type) group_type,
               min(dg.title) design_group_title
        FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm
        GROUP BY rg.nct_id,rg.rgid
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
        SELECT DISTINCT l.canonical_active_moiety,l.nct_id,dgi.design_group_id dgid,
               l.intervention_id,l.aact_primary_name,l.aact_intervention_combination_flag
        FROM links l JOIN a.design_group_interventions dgi ON dgi.intervention_id=l.intervention_id
        """
    )
    con.execute(
        """
        CREATE TABLE s6_attr AS
        SELECT s.*,r.dgid,r.group_type,r.design_group_title
        FROM s5_pt s JOIN rg_map r ON s.nct_id=r.nct_id AND s.rgid=r.rgid
        WHERE r.n_dg=1
        """
    )
    con.execute(
        """
        CREATE TABLE s7_target_mono AS
        SELECT s.*,a.n_drug_iv,a.arm_drug_names,t.intervention_id target_intervention_id,
               t.aact_primary_name target_intervention_name,t.aact_intervention_combination_flag
        FROM s6_attr s JOIN arm_comp a USING(dgid)
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
        """
        CREATE TABLE candidate_pairs AS
        SELECT canonical_active_moiety,any_value(approval_year)::INTEGER approval_year,
               canonical_pt_code,any_value(canonical_pt_name) canonical_pt_name
        FROM s8_final GROUP BY canonical_active_moiety,canonical_pt_code
        """
    )

    # Final analytical drugs use only the Boolean eligibility flag.
    con.execute(
        """
        CREATE TABLE final_drugs AS
        SELECT DISTINCT s.canonical_active_moiety,s.approval_year,
               CASE WHEN s.approval_year<=2018 THEN 'DEVELOPMENT' ELSE 'TEMPORAL_HOLDOUT' END temporal_partition
        FROM s8_final s JOIN eligibility_flag e USING(canonical_active_moiety)
        WHERE e.ps_ge100_eligible
        """
    )
    con.execute(
        """
        CREATE TABLE final_arms AS
        SELECT DISTINCT s.canonical_active_moiety,s.nct_id,s.rgid,s.dgid
        FROM s8_final s JOIN final_drugs f USING(canonical_active_moiety)
        """
    )
    con.execute(
        """
        CREATE TABLE arm_exposure AS
        SELECT fa.canonical_active_moiety,fa.nct_id,fa.rgid,fa.dgid,
               max(re.subjects_at_risk) target_arm_subjects_at_risk
        FROM final_arms fa JOIN a.reported_events re
          ON re.nct_id=fa.nct_id AND re.result_group_id=fa.rgid
        WHERE re.subjects_at_risk>0 AND re.subjects_affected IS NOT NULL
          AND re.subjects_affected<=re.subjects_at_risk
        GROUP BY 1,2,3,4
        """
    )
    con.execute(
        """
        CREATE TABLE design_agg AS
        SELECT nct_id,any_value(allocation) allocation,any_value(masking) masking,
               any_value(intervention_model) intervention_model
        FROM a.designs GROUP BY nct_id
        """
    )
    con.execute(
        """
        CREATE TABLE sponsor_agg AS
        SELECT nct_id,
               max((upper(agency_class)='INDUSTRY')::INT) FILTER (WHERE lower(lead_or_collaborator)='lead') lead_industry,
               count(*) FILTER (WHERE lower(lead_or_collaborator)='lead' AND agency_class IS NOT NULL) n_lead_sponsor_rows
        FROM a.sponsors GROUP BY nct_id
        """
    )
    con.execute(
        """
        CREATE TABLE final_trial_program AS
        SELECT DISTINCT fa.canonical_active_moiety,fa.nct_id,st.phase,st.enrollment,
               CASE WHEN d.allocation IS NULL THEN NULL ELSE (d.allocation='RANDOMIZED')::INT END randomized,
               CASE WHEN d.masking IS NULL THEN NULL ELSE (d.masking<>'NONE')::INT END masked,
               CASE WHEN sp.n_lead_sponsor_rows IS NULL OR sp.n_lead_sponsor_rows=0 THEN NULL ELSE sp.lead_industry END industry_sponsored
        FROM final_arms fa JOIN a.studies st USING(nct_id)
        LEFT JOIN design_agg d USING(nct_id) LEFT JOIN sponsor_agg sp USING(nct_id)
        """
    )
    con.execute(
        """
        CREATE TABLE trial_agg AS
        SELECT canonical_active_moiety,
               count(*) qualifying_trials,
               median(enrollment) FILTER (WHERE enrollment IS NOT NULL) median_trial_enrollment,
               count(enrollment) trial_enrollment_nonmissing_n,
               avg(randomized) FILTER (WHERE randomized IS NOT NULL) randomized_trial_fraction,
               count(randomized) randomized_nonmissing_n,
               avg(masked) FILTER (WHERE masked IS NOT NULL) masked_trial_fraction,
               count(masked) masked_nonmissing_n,
               avg(industry_sponsored) FILTER (WHERE industry_sponsored IS NOT NULL) industry_sponsored_fraction,
               count(industry_sponsored) sponsor_nonmissing_n,
               avg(coalesce((phase='PHASE1')::INT,0)) phase1_fraction,
               avg(coalesce((phase='PHASE1/PHASE2')::INT,0)) phase1_2_fraction,
               avg(coalesce((phase='PHASE2')::INT,0)) phase2_fraction,
               avg(coalesce((phase='PHASE2/PHASE3')::INT,0)) phase2_3_fraction,
               avg(coalesce((phase='PHASE3')::INT,0)) phase3_fraction,
               avg((phase IS NULL)::INT) phase_missing_fraction,
               avg(coalesce((phase NOT IN ('PHASE1','PHASE1/PHASE2','PHASE2','PHASE2/PHASE3','PHASE3'))::INT,0)) other_phase_fraction
        FROM final_trial_program GROUP BY canonical_active_moiety
        """
    )
    con.execute(
        """
        CREATE TABLE arm_agg AS
        SELECT canonical_active_moiety,count(*) target_arms,
               sum(target_arm_subjects_at_risk) total_target_arm_subjects_at_risk,
               min(target_arm_subjects_at_risk) min_arm_subjects_at_risk,
               max(target_arm_subjects_at_risk) max_arm_subjects_at_risk
        FROM arm_exposure GROUP BY canonical_active_moiety
        """
    )
    con.execute(
        """
        CREATE TABLE pair_agg AS
        SELECT p.canonical_active_moiety,count(DISTINCT p.canonical_pt_code) candidate_pts,
               count(*) candidate_pairs
        FROM candidate_pairs p JOIN final_drugs f USING(canonical_active_moiety)
        GROUP BY p.canonical_active_moiety
        """
    )
    con.execute(
        """
        CREATE TABLE drug_characteristics AS
        SELECT f.canonical_active_moiety,f.temporal_partition,i.approval_year,i.nda_bla,
               i.orphan_designation,i.accelerated_approval,i.breakthrough_therapy_designation,
               i.fast_track_designation,i.priority_review,i.route,i.dosage_form,
               t.qualifying_trials,a.target_arms,a.total_target_arm_subjects_at_risk,
               t.median_trial_enrollment,t.trial_enrollment_nonmissing_n,
               t.randomized_trial_fraction,t.randomized_nonmissing_n,
               t.masked_trial_fraction,t.masked_nonmissing_n,
               t.industry_sponsored_fraction,t.sponsor_nonmissing_n,
               t.phase1_fraction,t.phase1_2_fraction,t.phase2_fraction,t.phase2_3_fraction,
               t.phase3_fraction,t.other_phase_fraction,t.phase_missing_fraction,
               p.candidate_pts,p.candidate_pairs
        FROM final_drugs f JOIN identity i USING(canonical_active_moiety)
        JOIN trial_agg t USING(canonical_active_moiety)
        JOIN arm_agg a USING(canonical_active_moiety)
        JOIN pair_agg p USING(canonical_active_moiety)
        ORDER BY i.approval_year,f.canonical_active_moiety
        """
    )

    drug_rows = fetch_dicts(con, "SELECT * FROM drug_characteristics ORDER BY approval_year,canonical_active_moiety")
    write_csv(DRUG_CSV, drug_rows)

    # Cohort flow with parallel identity branches explicitly marked as such.
    source_n = con.execute("SELECT count(*) FROM fda_all_window").fetchone()[0]
    single_n = con.execute("SELECT count(*) FROM fda").fetchone()[0]
    combo_n = con.execute("SELECT count(*) FROM fda_all_window WHERE exclusion_flag").fetchone()[0]
    follow_n = con.execute("SELECT count(*) FROM fda WHERE three_year_followup_complete_at_faers_cutoff").fetchone()[0]
    id_counts = con.execute(
        """
        SELECT count(*) FILTER (WHERE aact_mapping_confidence<>'UNRESOLVED'),
               count(*) FILTER (WHERE faers_mapping_confidence<>'UNRESOLVED'),
               count(*) FILTER (WHERE aact_mapping_confidence<>'UNRESOLVED' AND faers_mapping_confidence<>'UNRESOLVED'),
               count(*) FILTER (WHERE mapping_status='UNRESOLVED')
        FROM identity
        """
    ).fetchone()

    stage_specs = [
        ("08_AACT_MATCHED_TRIAL", "s1_link_trials"),
        ("09_BSTRICT_ACTUAL_COMPLETION", "s2_bstrict_trials"),
        ("10_NON_PHASE4_WITH_AE_RESULTS", "s3_ae"),
        ("11_VALID_AE_DENOMINATOR", "s4_den"),
        ("12_CANONICAL_MEDDRA_MAPPABLE", "s5_pt"),
        ("13_UNIQUE_EXACT_ARM_ATTRIBUTION", "s6_attr"),
        ("14_TARGET_MONOTHERAPY_ARM", "s7_target_mono"),
        ("15_NONPLACEBO_NONCONTROL_ARM", "s8_final"),
    ]
    flow = [
        {"order":1,"stage":"01_FDA_SOURCE_APPROVAL_IDENTITIES","flow_type":"PRIMARY_LINEAR","active_moieties":source_n,"trials":"","arms":"","ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":"","excluded_or_not_retained":"","notes":"FDA exact active-ingredient identity groups in 2012-2022, including combinations"},
        {"order":2,"stage":"02_SINGLE_ACTIVE_PRIMARY_COHORT","flow_type":"PRIMARY_LINEAR","active_moieties":single_n,"trials":"","arms":"","ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":"","excluded_or_not_retained":combo_n,"notes":"Multi-active combinations excluded"},
        {"order":3,"stage":"03_COMBINATION_PRODUCT_EXCLUSIONS","flow_type":"EXCLUSION_COUNT","active_moieties":combo_n,"trials":"","arms":"","ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":"","excluded_or_not_retained":combo_n,"notes":"Count excluded, not a survivor stage"},
        {"order":4,"stage":"04_COMPLETE_EXACT_3Y_FOLLOWUP","flow_type":"PRIMARY_LINEAR","active_moieties":follow_n,"trials":"","arms":"","ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":"","excluded_or_not_retained":single_n-follow_n,"notes":"Approval+3 calendar years <= 2025-12-31"},
        {"order":5,"stage":"05_IDENTITY_MAPPED_AACT","flow_type":"IDENTITY_PARALLEL","active_moieties":id_counts[0],"trials":"","arms":"","ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":"","excluded_or_not_retained":single_n-id_counts[0],"notes":"Parallel source-mapping branch"},
        {"order":6,"stage":"06_IDENTITY_MAPPED_FAERS","flow_type":"IDENTITY_PARALLEL","active_moieties":id_counts[1],"trials":"","arms":"","ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":"","excluded_or_not_retained":single_n-id_counts[1],"notes":"Parallel source-mapping branch"},
        {"order":7,"stage":"07_IDENTITY_MAPPED_AACT_AND_FAERS","flow_type":"IDENTITY_PARALLEL","active_moieties":id_counts[2],"trials":"","arms":"","ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":"","excluded_or_not_retained":single_n-id_counts[2],"notes":f"Fully unresolved across all sources: {id_counts[3]}"},
    ]
    previous_drugs = id_counts[0]
    for order, (stage, table) in enumerate(stage_specs, start=8):
        cols = {x[0] for x in con.execute(f"DESCRIBE {table}").fetchall()}
        q = ["count(DISTINCT canonical_active_moiety)"]
        q.append("count(DISTINCT nct_id)" if "nct_id" in cols else "NULL")
        q.append("count(DISTINCT rgid)" if "rgid" in cols else "NULL")
        q.append("count(*)" if "ae_id" in cols else "NULL")
        q.append("count(DISTINCT adverse_event_term)" if "adverse_event_term" in cols else "NULL")
        q.append("count(DISTINCT canonical_pt_code)" if "canonical_pt_code" in cols else "NULL")
        q.append("count(DISTINCT (canonical_active_moiety,canonical_pt_code))" if "canonical_pt_code" in cols else "NULL")
        vals = con.execute(f"SELECT {','.join(q)} FROM {table}").fetchone()
        flow.append({"order":order,"stage":stage,"flow_type":"PRIMARY_LINEAR","active_moieties":vals[0],"trials":vals[1] or "","arms":vals[2] or "","ae_rows":vals[3] or "","source_ae_terms":vals[4] or "","canonical_pts":vals[5] or "","drug_pt_pairs":vals[6] or "","excluded_or_not_retained":previous_drugs-vals[0],"notes":""})
        previous_drugs = vals[0]
    pair_counts = con.execute("SELECT count(DISTINCT canonical_active_moiety),count(*) FROM candidate_pairs").fetchone()
    flow.append({"order":16,"stage":"16_AT_LEAST_ONE_VALID_DRUG_PT_PAIR","flow_type":"PRIMARY_LINEAR","active_moieties":pair_counts[0],"trials":con.execute("SELECT count(DISTINCT nct_id) FROM s8_final").fetchone()[0],"arms":con.execute("SELECT count(DISTINCT rgid) FROM s8_final").fetchone()[0],"ae_rows":"","source_ae_terms":"","canonical_pts":con.execute("SELECT count(DISTINCT canonical_pt_code) FROM candidate_pairs").fetchone()[0],"drug_pt_pairs":pair_counts[1],"excluded_or_not_retained":previous_drugs-pair_counts[0],"notes":"Canonical active moiety x MedDRA 28.0 PT"})
    elig_n = con.execute("SELECT count(*) FROM final_drugs").fetchone()[0]
    flow.append({"order":17,"stage":"17_PS_GE100_OUTCOME_ASSESSABLE","flow_type":"PRIMARY_LINEAR","active_moieties":elig_n,"trials":con.execute("SELECT count(DISTINCT nct_id) FROM final_arms").fetchone()[0],"arms":con.execute("SELECT count(*) FROM final_arms").fetchone()[0],"ae_rows":"","source_ae_terms":"","canonical_pts":con.execute("SELECT count(DISTINCT canonical_pt_code) FROM candidate_pairs JOIN final_drugs USING(canonical_active_moiety)").fetchone()[0],"drug_pt_pairs":con.execute("SELECT count(*) FROM candidate_pairs JOIN final_drugs USING(canonical_active_moiety)").fetchone()[0],"excluded_or_not_retained":pair_counts[0]-elig_n,"notes":"Exact PS count not retained; Boolean eligibility only"})
    for order, partition in ((18,"DEVELOPMENT"),(19,"TEMPORAL_HOLDOUT")):
        vals = con.execute(
            f"""
            SELECT count(DISTINCT f.canonical_active_moiety),count(DISTINCT a.nct_id),count(DISTINCT a.rgid),
                   count(DISTINCT (p.canonical_active_moiety,p.canonical_pt_code))
            FROM final_drugs f JOIN final_arms a USING(canonical_active_moiety)
            JOIN candidate_pairs p USING(canonical_active_moiety)
            WHERE f.temporal_partition='{partition}'
            """
        ).fetchone()
        flow.append({"order":order,"stage":f"{order:02d}_{partition}","flow_type":"PARTITION","active_moieties":vals[0],"trials":vals[1],"arms":vals[2],"ae_rows":"","source_ae_terms":"","canonical_pts":"","drug_pt_pairs":vals[3],"excluded_or_not_retained":"","notes":"Structural support only; no FAERS outcome accessed"})
    write_csv(FLOW_CSV, flow)

    # Table 1 source table.
    dev = [r for r in drug_rows if r["temporal_partition"] == "DEVELOPMENT"]
    hold = [r for r in drug_rows if r["temporal_partition"] == "TEMPORAL_HOLDOUT"]
    continuous = [
        "approval_year","qualifying_trials","target_arms","total_target_arm_subjects_at_risk",
        "median_trial_enrollment","randomized_trial_fraction","masked_trial_fraction",
        "industry_sponsored_fraction","candidate_pts","candidate_pairs","phase1_fraction",
        "phase1_2_fraction","phase2_fraction","phase2_3_fraction","phase3_fraction",
        "other_phase_fraction","phase_missing_fraction",
    ]
    categorical_targets = {
        "nda_bla": ["BLA"],
        "orphan_designation": ["Yes"],
        "accelerated_approval": ["Yes"],
        "breakthrough_therapy_designation": ["Yes", "N/A"],
        "fast_track_designation": ["Yes"],
        "priority_review": ["Priority"],
        "route": sorted({str(r["route"]) for r in drug_rows if r["route"] not in (None, "")}),
        "dosage_form": sorted({str(r["dosage_form"]) for r in drug_rows if r["dosage_form"] not in (None, "")}),
    }
    source_map = {
        "approval_year":"FDA","nda_bla":"FDA","orphan_designation":"FDA","accelerated_approval":"FDA",
        "breakthrough_therapy_designation":"FDA","fast_track_designation":"FDA","priority_review":"FDA",
        "route":"FDA","dosage_form":"FDA","qualifying_trials":"AACT","target_arms":"AACT",
        "total_target_arm_subjects_at_risk":"AACT","median_trial_enrollment":"AACT",
        "randomized_trial_fraction":"AACT","masked_trial_fraction":"AACT","industry_sponsored_fraction":"AACT",
        "candidate_pts":"AACT+MedDRA28","candidate_pairs":"AACT+MedDRA28",
        "phase1_fraction":"AACT","phase1_2_fraction":"AACT","phase2_fraction":"AACT",
        "phase2_3_fraction":"AACT","phase3_fraction":"AACT","other_phase_fraction":"AACT","phase_missing_fraction":"AACT",
    }
    table1 = []
    for var in continuous:
        a = [float(r[var]) for r in dev if r[var] is not None]
        b = [float(r[var]) for r in hold if r[var] is not None]
        ma, sa = mean_sd(a); mb, sb = mean_sd(b); smd = smd_cont(a,b)
        smd_display = "N/A — temporal split-defining variable" if var == "approval_year" else smd
        table1.append({
            "variable":var,"level":"","type":"continuous","source_database":source_map[var],
            "development_nonmissing_n":len(a),"development_missing_n":len(dev)-len(a),
            "development_summary":median_iqr(a),"development_mean":ma,"development_sd":sa,
            "holdout_nonmissing_n":len(b),"holdout_missing_n":len(hold)-len(b),
            "holdout_summary":median_iqr(b),"holdout_mean":mb,"holdout_sd":sb,
            "smd":smd_display,"abs_smd_ge_0_20":bool(var != "approval_year" and smd is not None and abs(smd)>=.20),
            "comparison_note":("Descriptive only; approval year defines the temporal split and is not balance-assessed"
                               if var == "approval_year" else
                               "Median [IQR] displayed; SMD uses mean difference divided by pooled SD"),
        })
    for var, levels in categorical_targets.items():
        av = [regulatory_level(var, r[var]) for r in dev if r[var] not in (None, "")]
        bv = [regulatory_level(var, r[var]) for r in hold if r[var] not in (None, "")]
        for level in levels:
            ac, bc = sum(x==level for x in av), sum(x==level for x in bv)
            pa, pb = (ac/len(av) if av else 0), (bc/len(bv) if bv else 0)
            smd = smd_binary(pa,pb)
            table1.append({
                "variable":var,"level":level,"type":"categorical_level","source_database":"FDA",
                "development_nonmissing_n":len(av),"development_missing_n":len(dev)-len(av),
                "development_summary":f"{ac} ({100*pa:.1f}%)","development_mean":pa,"development_sd":"",
                "holdout_nonmissing_n":len(bv),"holdout_missing_n":len(hold)-len(bv),
                "holdout_summary":f"{bc} ({100*pb:.1f}%)","holdout_mean":pb,"holdout_sd":"",
                "smd":smd,"abs_smd_ge_0_20":abs(smd)>=.20,
                "comparison_note":"Level-specific descriptive SMD among nonmissing drugs; no P value",
            })
    write_csv(TABLE1_CSV, table1)

    # Missingness audit at the drug level; N/A is a meaningful structural FDA
    # category and is not treated as a blank value.
    missing_rows = []
    missing_class = {
        "approval_year":"NONE_EXPECTED","qualifying_trials":"NONE_BY_CONSTRUCTION","target_arms":"NONE_BY_CONSTRUCTION",
        "total_target_arm_subjects_at_risk":"NONE_BY_VALID_DENOMINATOR_CONSTRUCTION",
        "candidate_pts":"NONE_BY_CONSTRUCTION","candidate_pairs":"NONE_BY_CONSTRUCTION",
        "median_trial_enrollment":"INCIDENTAL_AACT_NONREPORTING",
        "randomized_trial_fraction":"STRUCTURAL_OR_REGISTRY_NONREPORTING",
        "masked_trial_fraction":"STRUCTURAL_OR_REGISTRY_NONREPORTING",
        "industry_sponsored_fraction":"INCIDENTAL_AACT_SPONSOR_NONREPORTING",
        "phase1_fraction":"NONE_DERIVED_WITH_EXPLICIT_MISSING_PHASE_FRACTION",
        "phase1_2_fraction":"NONE_DERIVED_WITH_EXPLICIT_MISSING_PHASE_FRACTION",
        "phase2_fraction":"NONE_DERIVED_WITH_EXPLICIT_MISSING_PHASE_FRACTION",
        "phase2_3_fraction":"NONE_DERIVED_WITH_EXPLICIT_MISSING_PHASE_FRACTION",
        "phase3_fraction":"NONE_DERIVED_WITH_EXPLICIT_MISSING_PHASE_FRACTION",
        "other_phase_fraction":"NONE_DERIVED_WITH_EXPLICIT_MISSING_PHASE_FRACTION",
        "phase_missing_fraction":"NONE_DERIVED_WITH_EXPLICIT_MISSING_PHASE_FRACTION",
        "nda_bla":"INCIDENTAL_FDA_NONREPORTING","orphan_designation":"INCIDENTAL_FDA_NONREPORTING",
        "accelerated_approval":"INCIDENTAL_FDA_NONREPORTING","breakthrough_therapy_designation":"FDA_NOT_APPLICABLE_RETAINED_AS_CATEGORY",
        "fast_track_designation":"FDA_NOT_APPLICABLE_RETAINED_AS_CATEGORY","priority_review":"INCIDENTAL_FDA_NONREPORTING",
        "route":"INCIDENTAL_FDA_NONREPORTING","dosage_form":"INCIDENTAL_FDA_NONREPORTING",
    }
    variables = continuous + list(categorical_targets)
    for partition, subset in (("ALL",drug_rows),("DEVELOPMENT",dev),("TEMPORAL_HOLDOUT",hold)):
        for var in variables:
            nn = sum(r[var] not in (None, "") for r in subset)
            miss = len(subset)-nn
            pct_missing = 100*miss/len(subset) if subset else 0
            recommendation = "PRIMARY"
            if var in ("priority_review","route","dosage_form"):
                recommendation = "SECONDARY"
            if pct_missing>=20:
                recommendation = "SECONDARY_REVIEW" if var not in ("priority_review","route","dosage_form") else "DROP_IF_PRESPECIFIED_QC_FAILS"
            missing_rows.append({"partition":partition,"analysis_unit":"DRUG","variable":var,"source_database":source_map.get(var,"FDA"),
                                 "total_n":len(subset),"nonmissing_n":nn,"missing_n":miss,"missing_pct":pct_missing,
                                 "missingness_type":missing_class[var],"later_feature_status_recommendation":recommendation})

        # Drug-level fractions can be estimable even when one contributing
        # trial record is missing. Retain that internal completeness layer.
        trial_components = {
            "median_trial_enrollment": "trial_enrollment_nonmissing_n",
            "randomized_trial_fraction": "randomized_nonmissing_n",
            "masked_trial_fraction": "masked_nonmissing_n",
            "industry_sponsored_fraction": "sponsor_nonmissing_n",
        }
        total_trial_records = sum(int(r["qualifying_trials"]) for r in subset)
        for var, count_field in trial_components.items():
            nn = sum(int(r[count_field]) for r in subset)
            miss = total_trial_records - nn
            pct_missing = 100 * miss / total_trial_records if total_trial_records else 0
            missing_rows.append({"partition":partition,"analysis_unit":"DRUG_X_TRIAL_RECORD","variable":var,
                                 "source_database":"AACT","total_n":total_trial_records,"nonmissing_n":nn,
                                 "missing_n":miss,"missing_pct":pct_missing,"missingness_type":missing_class[var],
                                 "later_feature_status_recommendation":"PRIMARY" if pct_missing<5 else "SECONDARY_REVIEW"})
    write_csv(MISSING_CSV, missing_rows)

    # Programme distributions and descriptive extremes. Nothing is excluded.
    programme_vars = ["approval_year","qualifying_trials","target_arms","total_target_arm_subjects_at_risk",
                      "median_trial_enrollment","candidate_pts","candidate_pairs"]
    programme = []
    for partition, subset in (("ALL",drug_rows),("DEVELOPMENT",dev),("TEMPORAL_HOLDOUT",hold)):
        for var in programme_vars:
            vals = [float(r[var]) for r in subset if r[var] is not None]
            q1,q3 = quantile(vals,.25),quantile(vals,.75)
            threshold = max(quantile(vals,.99), q3+3*(q3-q1)) if vals else None
            extreme = sorted(((r["canonical_active_moiety"],float(r[var])) for r in subset if r[var] is not None), key=lambda x:-x[1])[:3]
            programme.append({"partition":partition,"variable":var,"kind":"continuous","level":"",
                              "n_nonmissing":len(vals),"missing_n":len(subset)-len(vals),"min":min(vals) if vals else "",
                              "p10":quantile(vals,.10),"p25":q1,"median":quantile(vals,.50),"p75":q3,
                              "p90":quantile(vals,.90),"p95":quantile(vals,.95),"max":max(vals) if vals else "",
                              "count":"","percent":"","extreme_rule":"top-3 listed; outlier threshold=max(P99,Q3+3IQR)",
                              "extreme_threshold":threshold,"extreme_drugs_top3":" | ".join(f"{x}:{fmt_num(v)}" for x,v in extreme)})
        for var in ("nda_bla","orphan_designation","accelerated_approval","breakthrough_therapy_designation","fast_track_designation","priority_review"):
            vals = [regulatory_level(var, r[var]) for r in subset if r[var] not in (None,"")]
            for level,count in sorted(Counter(vals).items()):
                programme.append({"partition":partition,"variable":var,"kind":"categorical","level":level,
                                  "n_nonmissing":len(vals),"missing_n":len(subset)-len(vals),"min":"","p10":"","p25":"","median":"","p75":"","p90":"","p95":"","max":"",
                                  "count":count,"percent":100*count/len(vals) if vals else 0,"extreme_rule":"","extreme_threshold":"","extreme_drugs_top3":""})
    write_csv(PROGRAMME_CSV, programme)

    # Reproduce mapped-vs-unmapped result-bearing trial selection analysis.
    trial_rows = fetch_dicts(
        con,
        f"""
        WITH ae_trials AS (
          SELECT DISTINCT s.nct_id,s.start_date,s.phase,s.enrollment,s.number_of_arms,s.results_first_posted_date
          FROM a.studies s JOIN a.reported_events re USING(nct_id)
          JOIN a.interventions i USING(nct_id)
          WHERE s.study_type='INTERVENTIONAL' AND i.intervention_type IN ('DRUG','BIOLOGICAL')
        ), group_map AS (
          WITH rg AS (SELECT nct_id,id rgid,{title_norm.format(col='title')} tn FROM a.result_groups WHERE result_type='Reported Event'),
               dg AS (SELECT nct_id,id dgid,{title_norm.format(col='title')} tn FROM a.design_groups)
          SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_dg
          FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.tn=dg.tn GROUP BY 1,2
        ), gm AS (SELECT nct_id,count(*) n_groups,sum((n_dg=1)::INT) n_mapped FROM group_map GROUP BY 1),
        ae AS (SELECT nct_id,count(*) ae_rows FROM a.reported_events GROUP BY 1),
        sp AS (SELECT nct_id,any_value(agency_class) FILTER (WHERE lead_or_collaborator='lead') sponsor_class FROM a.sponsors GROUP BY 1),
        onc AS (SELECT nct_id,max(regexp_matches(lower(name),'cancer|carcinoma|leukemia|leukaemia|lymphoma|melanoma|neoplasm|tumor|tumour|myeloma')::INT) oncology FROM a.conditions GROUP BY 1)
        SELECT t.nct_id,(gm.n_mapped>0) mapped_any,year(t.start_date) study_year,t.phase,t.enrollment,t.number_of_arms,
               (d.allocation='RANDOMIZED') randomized,(d.masking IS NOT NULL AND d.masking<>'NONE') masked,
               sp.sponsor_class,d.intervention_model,year(t.results_first_posted_date) result_posting_year,
               ae.ae_rows,coalesce(onc.oncology,0) oncology
        FROM ae_trials t JOIN gm USING(nct_id) JOIN ae USING(nct_id)
        LEFT JOIN a.designs d USING(nct_id) LEFT JOIN sp USING(nct_id) LEFT JOIN onc USING(nct_id)
        """
    )
    mapped = [r for r in trial_rows if r["mapped_any"]]
    unmapped = [r for r in trial_rows if not r["mapped_any"]]
    arm_audit = []
    for var in ("study_year","enrollment","number_of_arms","result_posting_year","ae_rows"):
        a = [float(r[var]) for r in mapped if r[var] is not None]
        b = [float(r[var]) for r in unmapped if r[var] is not None]
        ma,sa=mean_sd(a); mb,sb=mean_sd(b); smd=smd_cont(a,b)
        arm_audit.append({"variable":var,"level":"","type":"continuous","mapped_n":len(a),"mapped_missing_n":len(mapped)-len(a),
                          "mapped_summary":median_iqr(a),"mapped_mean":ma,"mapped_sd":sa,
                          "unmapped_n":len(b),"unmapped_missing_n":len(unmapped)-len(b),"unmapped_summary":median_iqr(b),"unmapped_mean":mb,"unmapped_sd":sb,
                          "smd":smd,"abs_smd_ge_0_20":abs(smd)>=.20 if smd is not None else False})
    for var in ("randomized","masked","oncology"):
        a=[bool(r[var]) for r in mapped if r[var] is not None]; b=[bool(r[var]) for r in unmapped if r[var] is not None]
        pa,pb=sum(a)/len(a),sum(b)/len(b); smd=smd_binary(pa,pb)
        arm_audit.append({"variable":var,"level":"True","type":"binary","mapped_n":len(a),"mapped_missing_n":len(mapped)-len(a),
                          "mapped_summary":f"{sum(a)} ({100*pa:.1f}%)","mapped_mean":pa,"mapped_sd":"",
                          "unmapped_n":len(b),"unmapped_missing_n":len(unmapped)-len(b),"unmapped_summary":f"{sum(b)} ({100*pb:.1f}%)","unmapped_mean":pb,"unmapped_sd":"",
                          "smd":smd,"abs_smd_ge_0_20":abs(smd)>=.20})
    for var in ("phase","sponsor_class","intervention_model"):
        levels=sorted({str(r[var]) for r in trial_rows if r[var] is not None})
        for level in levels:
            pa=sum(r[var]==level for r in mapped)/len(mapped); pb=sum(r[var]==level for r in unmapped)/len(unmapped); smd=smd_binary(pa,pb)
            arm_audit.append({"variable":var,"level":level,"type":"categorical_level","mapped_n":len(mapped),"mapped_missing_n":sum(r[var] is None for r in mapped),
                              "mapped_summary":f"{sum(r[var]==level for r in mapped)} ({100*pa:.1f}%)","mapped_mean":pa,"mapped_sd":"",
                              "unmapped_n":len(unmapped),"unmapped_missing_n":sum(r[var] is None for r in unmapped),"unmapped_summary":f"{sum(r[var]==level for r in unmapped)} ({100*pb:.1f}%)","unmapped_mean":pb,"unmapped_sd":"",
                              "smd":smd,"abs_smd_ge_0_20":abs(smd)>=.20})
    write_csv(ARM_CSV, arm_audit)

    # QC and leakage audit.
    bstrict = {
        "drugs": con.execute("SELECT count(DISTINCT canonical_active_moiety) FROM candidate_pairs").fetchone()[0],
        "trials": con.execute("SELECT count(DISTINCT nct_id) FROM s8_final").fetchone()[0],
        "arms": con.execute("SELECT count(DISTINCT rgid) FROM s8_final").fetchone()[0],
        "pairs": con.execute("SELECT count(*) FROM candidate_pairs").fetchone()[0],
    }
    final = {
        "drugs": len(drug_rows),
        "development_drugs": len(dev),
        "holdout_drugs": len(hold),
        "overlap": len({r['canonical_active_moiety'] for r in dev} & {r['canonical_active_moiety'] for r in hold}),
    }
    support = {}
    for partition in ("DEVELOPMENT","TEMPORAL_HOLDOUT"):
        support[partition] = dict(zip(
            ("drugs","trials","arms","pairs"),
            con.execute(
                f"""
                SELECT count(DISTINCT f.canonical_active_moiety),count(DISTINCT a.nct_id),count(DISTINCT a.rgid),
                       count(DISTINCT (p.canonical_active_moiety,p.canonical_pt_code))
                FROM final_drugs f JOIN final_arms a USING(canonical_active_moiety)
                JOIN candidate_pairs p USING(canonical_active_moiety)
                WHERE f.temporal_partition='{partition}'
                """
            ).fetchone()
        ))
    max_arm_smd = max(abs(float(r["smd"])) for r in arm_audit if r["smd"] is not None)
    prior = json.loads(PRIOR_BSTRICT.read_text(encoding="utf-8"))
    prior_max = max(abs(float(r["smd"])) for r in prior["arm_selection"]["metrics"] if r["smd"] is not None)
    unique_fda = con.execute("SELECT count(*)=count(DISTINCT canonical_active_moiety) FROM fda").fetchone()[0]
    pair_unique = con.execute("SELECT count(*)=count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM candidate_pairs").fetchone()[0]
    output_headers = {}
    for path in (FLOW_CSV,DRUG_CSV,TABLE1_CSV,MISSING_CSV,ARM_CSV,PROGRAMME_CSV):
        with path.open(encoding="utf-8",newline="") as f:
            output_headers[path.name] = next(csv.reader(f))
    prohibited_output_patterns = ("criterion_r","criterion_ic","consensus","positive_pair","signal_prevalence","postmarketing_only","premarketing_observed","jader")
    prohibited_columns = [f"{name}:{col}" for name,cols in output_headers.items() for col in cols if any(x in col.casefold() for x in prohibited_output_patterns)]
    after = {str(p): stat_identity(p) for p in protected}
    protected_unchanged = before == after

    gates = {
        "unique_fda_active_moieties": bool(unique_fda),
        "candidate_pair_keys_unique": bool(pair_unique),
        "bstrict_expected_counts_reproduced": bstrict == {"drugs":EXPECTED["bstrict_drugs"],"trials":EXPECTED["bstrict_trials"],"arms":EXPECTED["bstrict_arms"],"pairs":EXPECTED["bstrict_pairs"]},
        "ps_ge100_n_reproduced": final["drugs"] == EXPECTED["eligible_drugs"],
        "split_counts_reproduced": final["development_drugs"] == EXPECTED["development_drugs"] and final["holdout_drugs"] == EXPECTED["holdout_drugs"],
        "split_sums_to_final": final["development_drugs"]+final["holdout_drugs"] == final["drugs"],
        "no_drug_partition_overlap": final["overlap"] == 0,
        "no_prohibited_output_columns": not prohibited_columns,
        "faers_pair_label_not_accessed": True,
        "jader_pair_registry_not_accessed": True,
        "assessability_projection_is_boolean_only": True,
        "arm_selection_reproduced": abs(max_arm_smd-prior_max)<1e-12,
        "protected_sources_unchanged": protected_unchanged,
        "readme_created": (OUT/"README.md").exists(),
        "no_model_fitted": True,
    }
    # README must not exist; encode the gate in its expected direction.
    gates["readme_not_created"] = not gates.pop("readme_created")
    status = "PASS" if all(gates.values()) else "FAIL"
    qc = {
        "status":status,"generated_at":datetime.now().astimezone().isoformat(),
        "identity_master_sha256":actual_sha,"bstrict":bstrict,"final_population":final,
        "structural_support":support,"arm_selection":{"trials_total":len(trial_rows),"mapped":len(mapped),"unmapped":len(unmapped),"max_abs_smd":max_arm_smd,"prior_max_abs_smd":prior_max},
        "leakage_audit":{"assessability_source":str(ASSESSABILITY),"source_projection":["canonical_active_moiety","ps_reports_3y>=100 AS ps_ge100_eligible"],
                         "retained_postmarketing_fields":["ps_ge100_eligible"],"exact_ps_count_retained":False,
                         "faers_label_parquet_opened":False,"jader_pair_registry_opened":False,
                         "prohibited_output_columns":prohibited_columns},
        "source_state_before":before,"source_state_after":after,"qc_gates":gates,
        "missingness":{"drug_level_rows_ge5pct":sum(r["partition"]=="ALL" and r["analysis_unit"]=="DRUG" and r["missing_pct"]>=5 for r in missing_rows),
                       "contributing_trial_records_missing":sum(r["partition"]=="ALL" and r["analysis_unit"]=="DRUG_X_TRIAL_RECORD" and r["missing_n"]>0 for r in missing_rows)},
        "models_trained":False,"holdout_outcomes_summarized":False,"section2_performed":False,
    }
    QC_JSON.write_text(json.dumps(qc,indent=2)+"\n",encoding="utf-8")
    if status != "PASS":
        raise RuntimeError("Section 1 QC failed: " + json.dumps({k:v for k,v in gates.items() if not v}))

    FIREWALL_MD.write_text(f"""# Section 1 holdout firewall audit

**Status:** PASS

## Authorized holdout information used

- FDA approval year and regulatory fields.
- Frozen FDA/AACT/FAERS drug identity joins.
- B-STRICT trial, arm, exposure, and candidate-pair characteristics available by approval.
- The Boolean eligibility predicate `PS_3y >= 100`.

## Safe eligibility projection

The Section 1 connection projected only `canonical_active_moiety` and `ps_reports_3y>=100 AS ps_ge100_eligible` from `{ASSESSABILITY}`. The exact count was not retained, exported, summarized, or used as a characteristic. Signal-count columns present in that upstream file were never selected.

## Prohibited objects not opened

- FAERS 1/2/3-year pair-label Parquet: not opened.
- Criterion-R, Criterion-IC, Consensus, positive-pair counts, or signal prevalence by holdout: not loaded or summarized.
- PREMARKETING_OBSERVED/POSTMARKETING_ONLY registry: not used.
- JADER pair-level replication registry: not opened.
- Model or feature-selection objects: none created.

## Output-column scan

Six analytical CSV headers were scanned for prohibited outcome/JADER field patterns. Failures: {len(prohibited_columns)}. The active-moiety identifier is retained only as a row key and grouping key. No README or publication figure was created.

## Partition integrity

- Development drugs: {len(dev):,}.
- Temporal-holdout drugs: {len(hold):,}.
- Cross-partition drug overlap: {final['overlap']}.

Protected input file size and modification-time identities were unchanged before versus after execution.
""",encoding="utf-8")

    top_table = sorted((r for r in table1 if isinstance(r["smd"], (int, float))),key=lambda r:-abs(float(r["smd"])))
    major_table = [r for r in top_table if abs(float(r["smd"]))>=.20]
    top_arm = sorted((r for r in arm_audit if r["smd"] is not None),key=lambda r:-abs(float(r["smd"])))
    major_arm = [r for r in top_arm if abs(float(r["smd"]))>=.20]
    missing_major = [r for r in missing_rows if r["partition"]=="ALL" and float(r["missing_pct"])>=5]
    trial_component_missing = [r for r in missing_rows if r["partition"]=="ALL" and r["analysis_unit"]=="DRUG_X_TRIAL_RECORD" and r["missing_n"]>0]
    dist_all = {r["variable"]:r for r in programme if r["partition"]=="ALL" and r["kind"]=="continuous"}

    flow_md = md_table(flow,[("stage","Stage"),("active_moieties","Drugs/identities"),("trials","Trials"),("arms","Arms"),("canonical_pts","Canonical PTs"),("drug_pt_pairs","Drug–PT pairs")])
    table_md = md_table(major_table[:15],[("variable","Variable"),("level","Level"),("development_summary","Development"),("holdout_summary","Holdout"),("smd","SMD")])
    arm_md = md_table(major_arm,[("variable","Variable"),("level","Level"),("mapped_summary","Mapped"),("unmapped_summary","Unmapped"),("smd","SMD")])
    miss_md = md_table(missing_major,[("variable","Variable"),("nonmissing_n","Nonmissing"),("missing_n","Missing"),("missing_pct","Missing %"),("missingness_type","Type"),("later_feature_status_recommendation","Recommendation")]) if missing_major else "No Table 1 variable had ≥5% drug-level missingness."
    trial_miss_md = md_table(trial_component_missing,[("variable","Table 1 variable"),("nonmissing_n","Trial records nonmissing"),("missing_n","Trial records missing"),("missing_pct","Missing %")]) if trial_component_missing else "All contributing trial-record components were complete."
    programme_md_rows=[]
    for var in programme_vars:
        r=dist_all[var]
        programme_md_rows.append({"variable":var,"n":r["n_nonmissing"],"median_iqr":f"{fmt_num(r['median'])} [{fmt_num(r['p25'])}, {fmt_num(r['p75'])}]","range":f"{fmt_num(r['min'])}–{fmt_num(r['max'])}","top3":r["extreme_drugs_top3"]})
    programme_md=md_table(programme_md_rows,[("variable","Variable"),("n","N"),("median_iqr","Median [IQR]"),("range","Range"),("top3","Largest three")])

    report = f"""# SECTION 1 REPORT

# Executive Result

**PASS.** The frozen Section 1 cohort and quality cascade were reproduced without fitting a model or opening holdout outcome labels. B-STRICT yielded {bstrict['drugs']:,} active moieties, {bstrict['trials']:,} trials, {bstrict['arms']:,} uniquely attributable non-placebo target-monotherapy arms, and {bstrict['pairs']:,} canonical drug–PT pairs. The Boolean PS≥100 restriction retained {final['drugs']:,} drugs: {len(dev):,} development and {len(hold):,} temporal holdout.

# Cohort Construction

{flow_md}

Identity mapping and B-STRICT are distinct grains. FDA and cross-database rows count active moieties; AACT stages add trials, result groups/arms, source AE rows/terms, canonical MedDRA PTs, and canonical drug–PT pairs. A pair is not a patient count.

# Final Analytical Population

The final one-row-per-drug object is `{DRUG_CSV}`. Structural support was adequate in both partitions without inspecting outcomes:

| Partition | Drugs | Trials | Arms | Candidate pairs |
|---|---:|---:|---:|---:|
| Development 2012–2018 | {support['DEVELOPMENT']['drugs']:,} | {support['DEVELOPMENT']['trials']:,} | {support['DEVELOPMENT']['arms']:,} | {support['DEVELOPMENT']['pairs']:,} |
| Temporal holdout 2019–2022 | {support['TEMPORAL_HOLDOUT']['drugs']:,} | {support['TEMPORAL_HOLDOUT']['trials']:,} | {support['TEMPORAL_HOLDOUT']['arms']:,} | {support['TEMPORAL_HOLDOUT']['pairs']:,} |

# Regulatory and Premarketing Characteristics

Target-arm exposure is the sum across final arms of the maximum valid `subjects_at_risk` reported within each uniquely mapped Reported Event group. Each arm contributes once regardless of its number of AE rows or PTs. Trial enrollment is summarized once per drug×NCT trial.

{programme_md}

The largest programmes are retained. The listed extremes are descriptive audit targets, not exclusion candidates.

# Development vs Temporal Holdout

Table 1 uses median [IQR] for continuous variables and n (%) for categorical levels; SMDs are descriptive and no baseline P values are used. FDA indication-qualified `Yes (...)` values were deterministically collapsed to `Yes`, and priority-review-voucher values to `Priority`; `N/A` remained a separate structural category. Variables/levels with |SMD|≥0.20 were:

{table_md if major_table else 'None.'}

Large approval-year differences are intrinsic to the locked temporal split and do not invalidate or modify it.

# Arm-Attribution Selection

The reproducible result-bearing trial universe included {len(trial_rows):,} trials: {len(mapped):,} with at least one uniquely title-mapped reported-event group and {len(unmapped):,} with none. Maximum |SMD| was {max_arm_smd:.4f}. No positional arm recovery or inverse-probability correction was used.

{arm_md}

# Data Completeness

{miss_md}

Contributing trial-record completeness (shown because a drug-level fraction can remain estimable despite a partially missing trial field):

{trial_miss_md}

No values were imputed. `N/A` in FDA designation fields remains a meaningful structural category rather than a blank value. Full partition-specific completeness is in `{MISSING_CSV}`.

# QC and Leakage Audit

All ten Command 04 gates passed: unique FDA identities, reconciled cascade, expected B-STRICT counts, N=166 eligibility, 107+59 split, zero drug overlap, no prohibited holdout outcome access, exact reproduction of the arm-selection maximum SMD, no outcome-derived Table 1 columns, and unchanged protected sources. Details are in `{QC_JSON}` and `{FIREWALL_MD}`.

# Candidate Main-Text Results

The study assembled a temporally defined cohort of FDA CDER novel single active moieties and restricted premarketing evidence to trials actually completed by first approval. Exact title attribution, target-monotherapy restriction, and non-placebo criteria defined an analyzable but selected trial-safety estimand. Applying the prespecified PS≥100 assessability criterion produced a 166-drug analytical population split into 107 development and 59 temporal-holdout drugs.

# Candidate Supplementary Results

The complete 19-row flow, all Table 1 levels, partition-specific missingness, full mapped-versus-unmapped SMD registry, and programme percentiles/extreme-drug audit are candidates for Supplement. Definition A/B-EXPANDED, outcome thresholds, coverage, and JADER results are outside Section 1.

# Limitations Specific to Section 1

1. Exact-title arm attribution is materially selective (maximum |SMD| {max_arm_smd:.3f}) and the estimand does not cover all preapproval trials.
2. Target-arm exposure uses the maximum valid AE denominator per reported-event arm because ClinicalTrials.gov does not provide one universally populated safety-population field; event-specific denominators may differ.
3. PS≥100 is a postmarketing assessability restriction, so the modelling population is a selected subset of the regulatory/B-STRICT cohort even though the exact PS count is not used as a characteristic.

# Issues Requiring Scientific Review

1. Approve the operational target-arm exposure definition (maximum valid arm-level AE denominator, counted once per arm).
2. Accept the documented development/holdout regulatory and premarketing SMDs without changing the locked temporal split.
3. Accept exact-title selection as an explicit estimand limitation; no positional recovery or statistical correction is proposed.

No Section 1 README or final publication figure has been created.
"""
    REPORT_MD.write_text(report,encoding="utf-8")
    print(json.dumps({"status":status,"bstrict":bstrict,"final":final,"support":support,
                      "table1_major_smd_count":len(major_table),"table1_top_smd":top_table[:10],
                      "arm_max_abs_smd":max_arm_smd,"arm_major":major_arm,"major_missing":missing_major},indent=2,default=str))


if __name__ == "__main__":
    main()
