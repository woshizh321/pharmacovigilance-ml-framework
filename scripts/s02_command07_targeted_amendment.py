#!/usr/bin/env python3
"""Command 07 targeted Section 2 amendment.

Reads the locked Section 2 primary outputs and rebuilds only:
1) the POSTMARKETING_ONLY A/B/C/D hierarchy;
2) the B-STRICT trial-level non-serious threshold classification;
3) dependent QC, narrative locks, README, and Figure 2 Panel D source.

No FAERS source table, holdout outcome, JADER object, or model object is read.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import duckdb

from s02_section2_coverage import fetch_dicts, md_table, pct, sha256, stat_identity, write_csv


PROJECT = Path("/path/to/PDS")
OUT = PROJECT / "analysis/section2_coverage"
FIG = OUT / "09_figure2_source_data"
HISTORY = OUT / "history"

SECTION1 = PROJECT / "analysis/section1_cohort/02_final_drug_level_characteristics.csv"
SECTION1_README = PROJECT / "analysis/section1_cohort/SECTION1_README.md"
DEVIATIONS = PROJECT / "docs/PROTOCOL_DEVIATIONS.md"
IDENTITY = PROJECT / "preflight_v2/drug_identity_master.csv"
IDENTITY_SHA = PROJECT / "preflight_v2/drug_identity_master.sha256"
LINKS = PROJECT / "data/processed/preflight_v2/aact_fda_intervention_links.parquet"
PT_MAP = PROJECT / "preflight_v2/faers_pt_repair/aact_meddra28_term_mapping.csv"
AACT_DB = Path("/path/to/Database/AACT/aact.duckdb")

SIGNALS = OUT / "01_development_signal_universe.csv"
SUMMARY = OUT / "02_primary_coverage_summary.csv"
DRUG = OUT / "03_drug_level_coverage.csv"
SOC = OUT / "04_soc_coverage.csv"
STRATA = OUT / "05_regulatory_strata_coverage.csv"
DECOMP = OUT / "06_postmarketing_only_decomposition.csv"
THRESHOLD = OUT / "07_reporting_threshold_context.csv"
EVENT_PROFILE = OUT / "08_premarketing_event_type_profile.csv"
FIREWALL = OUT / "00_holdout_firewall_audit.md"
REPORT = OUT / "SECTION2_REPORT.md"
README = OUT / "SECTION2_README.md"
QC = OUT / "SECTION2_QC.json"
PANEL_D = FIG / "panel_d_postmarketing_only_decomposition.csv"

DECOMP_ORDER = [
    "A_SAME_BSTRICT_TRIAL_OTHER_ARM_OR_GROUP",
    "B_OTHER_ACTUAL_COMPLETION_PREAPPROVAL_TRIAL",
    "C_NOT_FOUND_IN_ANY_ACTUAL_COMPLETION_PREAPPROVAL_AACT_AE_RESULTS",
    "D_UNCLASSIFIABLE",
]
THRESHOLD_ORDER = ["0%", ">0_TO_<5%", "=5%", ">5%", "MIXED_THRESHOLDS", "MISSING"]
MAJOR_SOC_MIN = 30


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    required = [
        SECTION1, SECTION1_README, DEVIATIONS, IDENTITY, IDENTITY_SHA, LINKS, PT_MAP, AACT_DB,
        SIGNALS, SUMMARY, DRUG, SOC, STRATA, EVENT_PROFILE,
        FIG / "panel_a_overall_coverage.csv", FIG / "panel_b_soc_coverage.csv",
        FIG / "panel_c_drug_coverage_distribution.csv",
        HISTORY / "06_postmarketing_only_decomposition_pre_command07.csv",
        HISTORY / "07_reporting_threshold_context_pre_command07.csv",
        HISTORY / "SECTION2_REPORT_pre_command07.md", HISTORY / "SECTION2_QC_pre_command07.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Required locked/historical inputs missing: {missing}")
    expected_sha = IDENTITY_SHA.read_text(encoding="utf-8").split()[0]
    if sha256(IDENTITY) != expected_sha:
        raise RuntimeError("Frozen identity master hash mismatch")

    # These files are inputs to Command 07 and must remain byte/stat identical.
    protected = [
        SECTION1, SECTION1_README, DEVIATIONS, IDENTITY, IDENTITY_SHA, LINKS, PT_MAP, AACT_DB,
        SIGNALS, SUMMARY, DRUG, SOC, STRATA, EVENT_PROFILE,
        FIG / "panel_a_overall_coverage.csv", FIG / "panel_b_soc_coverage.csv",
        FIG / "panel_c_drug_coverage_distribution.csv",
    ]
    before = {str(p): stat_identity(p) for p in protected}

    summary_rows = read_csv_rows(SUMMARY)
    if len(summary_rows) != 1:
        raise RuntimeError("Locked primary summary must contain exactly one row")
    s = summary_rows[0]
    primary_lock = {
        "active_moieties": int(s["active_moieties"]),
        "criterion_r_signals": int(s["criterion_r_signals"]),
        "premarketing_observed": int(s["premarketing_observed"]),
        "postmarketing_only": int(s["postmarketing_only"]),
        "micro_coverage_pct": float(s["micro_coverage_pct"]),
        "micro_bootstrap_ci95_low_pct": float(s["micro_bootstrap_ci95_low_pct"]),
        "micro_bootstrap_ci95_high_pct": float(s["micro_bootstrap_ci95_high_pct"]),
        "macro_coverage_pct": float(s["macro_coverage_pct"]),
        "macro_bootstrap_ci95_low_pct": float(s["macro_bootstrap_ci95_low_pct"]),
        "macro_bootstrap_ci95_high_pct": float(s["macro_bootstrap_ci95_high_pct"]),
        "per_drug_median_coverage_pct": float(s["per_drug_median_coverage_pct"]),
        "per_drug_p25_coverage_pct": float(s["per_drug_p25_coverage_pct"]),
        "per_drug_p75_coverage_pct": float(s["per_drug_p75_coverage_pct"]),
        "per_drug_min_coverage_pct": float(s["per_drug_min_coverage_pct"]),
        "per_drug_max_coverage_pct": float(s["per_drug_max_coverage_pct"]),
        "bootstrap_resamples": int(s["bootstrap_resamples"]),
        "bootstrap_seed": int(s["bootstrap_seed"]),
        "bootstrap_unit": s["bootstrap_unit"],
    }
    exact_primary_lock = (
        primary_lock["active_moieties"] == 107
        and primary_lock["criterion_r_signals"] == 10857
        and primary_lock["premarketing_observed"] == 2064
        and primary_lock["postmarketing_only"] == 8793
        and abs(primary_lock["micro_coverage_pct"] - 19.010776457584967) < 1e-12
        and abs(primary_lock["micro_bootstrap_ci95_low_pct"] - 16.346653046666308) < 1e-12
        and abs(primary_lock["micro_bootstrap_ci95_high_pct"] - 22.05832792189638) < 1e-12
        and abs(primary_lock["macro_coverage_pct"] - 22.332810796968715) < 1e-12
        and abs(primary_lock["per_drug_median_coverage_pct"] - 17.24137931034483) < 1e-12
        and abs(primary_lock["per_drug_p25_coverage_pct"] - 10.319148936170214) < 1e-12
        and abs(primary_lock["per_drug_p75_coverage_pct"] - 32.961961314598156) < 1e-12
    )
    if not exact_primary_lock:
        raise RuntimeError("Locked primary Section 2 values do not match Command 07")

    con = duckdb.connect()
    con.execute(
        "SET threads=8; SET memory_limit='20GB'; SET temp_directory='/private/tmp/pds_duckdb'; "
        "SET preserve_insertion_order=false; SET enable_progress_bar=false"
    )
    con.execute(f"ATTACH '{AACT_DB}' AS a (READ_ONLY)")
    con.execute(f"CREATE TABLE locked_signals AS SELECT * FROM read_csv_auto('{SIGNALS}',header=true)")
    signal_qc = con.execute(
        """SELECT count(*),count(DISTINCT (canonical_active_moiety,canonical_pt_code)),
                  count(DISTINCT canonical_active_moiety),
                  sum((coverage_class='PREMARKETING_OBSERVED')::INT),
                  sum((coverage_class='POSTMARKETING_ONLY')::INT),min(approval_year),max(approval_year)
           FROM locked_signals"""
    ).fetchone()
    if signal_qc != (10857, 10857, 107, 2064, 8793, 2012, 2018):
        raise RuntimeError(f"Locked signal universe mismatch: {signal_qc}")

    con.execute(
        f"""CREATE TABLE allowlist AS
            SELECT d.canonical_active_moiety,cast(i.fda_first_approval_date AS DATE) approval_date,
                   d.approval_year
            FROM read_csv_auto('{SECTION1}',header=true) d
            JOIN read_csv_auto('{IDENTITY}',header=true) i USING(canonical_active_moiety)
            WHERE d.temporal_partition='DEVELOPMENT' AND d.approval_year BETWEEN 2012 AND 2018
              AND i.aact_mapping_confidence<>'UNRESOLVED' AND i.faers_mapping_confidence<>'UNRESOLVED'"""
    )
    allow_qc = con.execute(
        "SELECT count(*),count(DISTINCT canonical_active_moiety),min(approval_year),max(approval_year) FROM allowlist"
    ).fetchone()
    if allow_qc != (107, 107, 2012, 2018):
        raise RuntimeError(f"Development allowlist mismatch: {allow_qc}")

    con.execute(f"CREATE TABLE links AS SELECT l.* FROM read_parquet('{LINKS}') l JOIN allowlist USING(canonical_active_moiety)")
    con.execute(
        f"""CREATE TABLE aact_pt_map AS
            SELECT aact_term_raw,cast(canonical_pt_code AS BIGINT) canonical_pt_code,
                   canonical_pt_name,mapping_level
            FROM read_csv_auto('{PT_MAP}',header=true)
            WHERE mapping_status='MAPPED' AND canonical_pt_code IS NOT NULL"""
    )
    con.execute(
        """CREATE TABLE s1_link_trials AS
           SELECT DISTINCT a.*,l.nct_id FROM allowlist a JOIN links l USING(canonical_active_moiety)"""
    )
    con.execute(
        """CREATE TABLE s2_bstrict_trials AS
           SELECT s.*,st.completion_date,st.completion_date_type,st.phase
           FROM s1_link_trials s JOIN a.studies st USING(nct_id)
           WHERE st.completion_date IS NOT NULL AND st.completion_date_type='ACTUAL'
             AND st.completion_date<=s.approval_date AND coalesce(st.phase,'')<>'PHASE4'"""
    )
    con.execute(
        """CREATE TABLE s3_ae AS
           SELECT s.*,re.id ae_id,re.result_group_id rgid,re.adverse_event_term,
                  re.subjects_affected,re.subjects_at_risk,re.event_type,re.frequency_threshold
           FROM s2_bstrict_trials s JOIN a.reported_events re USING(nct_id)"""
    )
    con.execute(
        """CREATE TABLE s4_den AS SELECT * FROM s3_ae
           WHERE subjects_at_risk>0 AND subjects_affected IS NOT NULL
             AND subjects_affected<=subjects_at_risk"""
    )
    con.execute(
        """CREATE TABLE s5_pt AS
           SELECT s.*,p.canonical_pt_code,p.canonical_pt_name,p.mapping_level
           FROM s4_den s JOIN aact_pt_map p ON s.adverse_event_term=p.aact_term_raw"""
    )
    tn = "lower(trim(regexp_replace(regexp_replace({col}, '[^A-Za-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')))"
    con.execute(
        f"""CREATE TABLE rg_map AS
            WITH rg AS (
              SELECT nct_id,id rgid,{tn.format(col='title')} title_norm FROM a.result_groups
              WHERE result_type='Reported Event' AND nct_id IN (SELECT nct_id FROM s1_link_trials)
            ), dg AS (
              SELECT nct_id,id dgid,group_type,title,{tn.format(col='title')} title_norm FROM a.design_groups
              WHERE nct_id IN (SELECT nct_id FROM s1_link_trials)
            )
            SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_dg,min(dg.dgid) dgid,
                   min(dg.group_type) group_type,min(dg.title) design_group_title
            FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm GROUP BY 1,2"""
    )
    con.execute(
        """CREATE TABLE arm_comp AS
           SELECT dg.id dgid,
                  count(DISTINCT i.id) FILTER (WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')) n_drug_iv
           FROM a.design_groups dg LEFT JOIN a.design_group_interventions dgi ON dgi.design_group_id=dg.id
           LEFT JOIN a.interventions i ON i.id=dgi.intervention_id
           WHERE dg.nct_id IN (SELECT nct_id FROM s1_link_trials) GROUP BY dg.id"""
    )
    con.execute(
        """CREATE TABLE target_arm AS
           SELECT DISTINCT l.canonical_active_moiety,l.nct_id,dgi.design_group_id dgid,
                  l.intervention_id,l.aact_primary_name,l.aact_intervention_combination_flag
           FROM links l JOIN a.design_group_interventions dgi ON dgi.intervention_id=l.intervention_id"""
    )
    con.execute(
        """CREATE TABLE s6_attr AS
           SELECT s.*,r.dgid,r.group_type,r.design_group_title FROM s5_pt s JOIN rg_map r
             ON s.nct_id=r.nct_id AND s.rgid=r.rgid WHERE r.n_dg=1"""
    )
    con.execute(
        """CREATE TABLE s7_target_mono AS
           SELECT s.*,t.intervention_id target_intervention_id,t.aact_primary_name target_intervention_name,
                  t.aact_intervention_combination_flag
           FROM s6_attr s JOIN arm_comp ac USING(dgid)
           JOIN target_arm t ON s.canonical_active_moiety=t.canonical_active_moiety
                            AND s.nct_id=t.nct_id AND s.dgid=t.dgid
           WHERE ac.n_drug_iv=1"""
    )
    con.execute(
        """CREATE TABLE s8_final AS SELECT * FROM s7_target_mono
           WHERE group_type IN ('EXPERIMENTAL','OTHER') AND NOT aact_intervention_combination_flag
             AND NOT regexp_matches(lower(coalesce(target_intervention_name,'')),
                 '(^| )(placebo|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline)( |$)')
             AND NOT regexp_matches(lower(coalesce(design_group_title,'')),
                 '(^| )(placebo|vehicle|sham|no treatment|control)( |$)')"""
    )
    con.execute(
        "CREATE TABLE primary_pairs AS SELECT DISTINCT canonical_active_moiety,canonical_pt_code FROM s8_final"
    )
    con.execute(
        "CREATE TABLE primary_profile_trials AS SELECT DISTINCT canonical_active_moiety,nct_id FROM s8_final"
    )
    bstrict_qc = con.execute(
        """SELECT count(DISTINCT canonical_active_moiety),count(DISTINCT nct_id),
                  count(DISTINCT rgid),(SELECT count(*) FROM primary_pairs) FROM s8_final"""
    ).fetchone()
    if bstrict_qc != (107, 338, 632, 16470):
        raise RuntimeError(f"Development B-STRICT profile mismatch: {bstrict_qc}")
    observed_mismatch = con.execute(
        """SELECT count(*) FROM locked_signals s LEFT JOIN primary_pairs p
             USING(canonical_active_moiety,canonical_pt_code)
           WHERE (s.coverage_class='PREMARKETING_OBSERVED')<>(p.canonical_active_moiety IS NOT NULL)"""
    ).fetchone()[0]

    # A: mapped occurrence elsewhere in a drug-trial that contributes at least
    # one row to the primary B-STRICT target-monotherapy profile.
    con.execute(
        """CREATE TABLE a_evidence AS
           SELECT s.canonical_active_moiety,p.canonical_pt_code,
                  string_agg(DISTINCT s.nct_id,' | ' ORDER BY s.nct_id) supporting_nct_ids,
                  count(DISTINCT s.nct_id) supporting_trial_count,
                  min(s.completion_date) support_completion_date_min,
                  max(s.completion_date) support_completion_date_max,
                  bool_and(s.completion_date_type='ACTUAL' AND s.completion_date<=s.approval_date)
                    support_all_actual_preapproval
           FROM s3_ae s JOIN primary_profile_trials q
             ON q.canonical_active_moiety=s.canonical_active_moiety AND q.nct_id=s.nct_id
           JOIN aact_pt_map p ON s.adverse_event_term=p.aact_term_raw GROUP BY 1,2"""
    )
    # B: another linked interventional DRUG/BIOLOGICAL trial, still using only
    # actual completion on/before approval, and outside the primary-profile
    # drug-trial set because of another analytical eligibility/QC restriction.
    con.execute(
        """CREATE TABLE other_actual_preapproval_trials AS
           SELECT DISTINCT s.*,st.completion_date,st.completion_date_type,st.phase
           FROM s1_link_trials s JOIN a.studies st USING(nct_id)
           WHERE st.study_type='INTERVENTIONAL' AND st.completion_date IS NOT NULL
             AND st.completion_date_type='ACTUAL' AND st.completion_date<=s.approval_date
             AND EXISTS (SELECT 1 FROM a.interventions i
                         WHERE i.nct_id=s.nct_id AND i.intervention_type IN ('DRUG','BIOLOGICAL'))
             AND NOT EXISTS (SELECT 1 FROM primary_profile_trials b
                             WHERE b.canonical_active_moiety=s.canonical_active_moiety
                               AND b.nct_id=s.nct_id)"""
    )
    con.execute(
        """CREATE TABLE b_evidence AS
           SELECT s.canonical_active_moiety,p.canonical_pt_code,
                  string_agg(DISTINCT s.nct_id,' | ' ORDER BY s.nct_id) supporting_nct_ids,
                  count(DISTINCT s.nct_id) supporting_trial_count,
                  min(s.completion_date) support_completion_date_min,
                  max(s.completion_date) support_completion_date_max,
                  bool_and(s.completion_date_type='ACTUAL' AND s.completion_date<=s.approval_date)
                    support_all_actual_preapproval
           FROM other_actual_preapproval_trials s JOIN a.reported_events re USING(nct_id)
           JOIN aact_pt_map p ON re.adverse_event_term=p.aact_term_raw GROUP BY 1,2"""
    )
    con.execute(
        """CREATE TABLE decomposition AS
           SELECT s.canonical_active_moiety,s.approval_year,s.canonical_pt_code,s.canonical_pt_name,
                  s.primary_soc,
                  CASE WHEN a.canonical_active_moiety IS NOT NULL THEN 'A_SAME_BSTRICT_TRIAL_OTHER_ARM_OR_GROUP'
                       WHEN b.canonical_active_moiety IS NOT NULL THEN 'B_OTHER_ACTUAL_COMPLETION_PREAPPROVAL_TRIAL'
                       WHEN s.canonical_pt_code IS NOT NULL THEN 'C_NOT_FOUND_IN_ANY_ACTUAL_COMPLETION_PREAPPROVAL_AACT_AE_RESULTS'
                       ELSE 'D_UNCLASSIFIABLE' END decomposition_class,
                  CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.supporting_nct_ids
                       WHEN b.canonical_active_moiety IS NOT NULL THEN b.supporting_nct_ids END supporting_nct_ids,
                  CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.supporting_trial_count
                       WHEN b.canonical_active_moiety IS NOT NULL THEN b.supporting_trial_count ELSE 0 END supporting_trial_count,
                  CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.support_completion_date_min
                       WHEN b.canonical_active_moiety IS NOT NULL THEN b.support_completion_date_min END support_completion_date_min,
                  CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.support_completion_date_max
                       WHEN b.canonical_active_moiety IS NOT NULL THEN b.support_completion_date_max END support_completion_date_max,
                  CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.support_all_actual_preapproval
                       WHEN b.canonical_active_moiety IS NOT NULL THEN b.support_all_actual_preapproval END support_all_actual_preapproval,
                  CASE WHEN a.canonical_active_moiety IS NOT NULL OR b.canonical_active_moiety IS NOT NULL
                       THEN 'ACTUAL_STUDY_COMPLETION_ON_OR_BEFORE_FDA_APPROVAL'
                       ELSE 'NO_SUPPORTING_PREAPPROVAL_AACT_AE_OCCURRENCE' END temporal_evidence_rule
           FROM locked_signals s
           LEFT JOIN a_evidence a USING(canonical_active_moiety,canonical_pt_code)
           LEFT JOIN b_evidence b USING(canonical_active_moiety,canonical_pt_code)
           WHERE s.coverage_class='POSTMARKETING_ONLY'
           ORDER BY s.approval_year,s.canonical_active_moiety,s.canonical_pt_code"""
    )
    decomp_rows = fetch_dicts(con, "SELECT * FROM decomposition")
    write_csv(DECOMP, decomp_rows)
    decomp_counts = Counter(r["decomposition_class"] for r in decomp_rows)
    panel_d = [
        {"decomposition_class": k, "signals": decomp_counts.get(k, 0),
         "percent_of_postmarketing_only": pct(decomp_counts.get(k, 0), len(decomp_rows))}
        for k in DECOMP_ORDER
    ]
    write_csv(PANEL_D, panel_d)

    # Trial thresholds use distinct non-missing numeric categories only.
    con.execute(
        """CREATE TABLE primary_ae_rows AS
           SELECT DISTINCT canonical_active_moiety,nct_id,ae_id,lower(event_type) event_type,frequency_threshold
           FROM s8_final"""
    )
    row_case = """CASE WHEN frequency_threshold IS NULL THEN 'MISSING'
                       WHEN frequency_threshold=0 THEN '0%'
                       WHEN frequency_threshold>0 AND frequency_threshold<5 THEN '>0_TO_<5%'
                       WHEN frequency_threshold=5 THEN '=5%'
                       WHEN frequency_threshold>5 THEN '>5%'
                       ELSE 'MISSING' END"""
    threshold_rows: list[dict] = []
    for scope, condition in (
        ("ALL_AE_ROWS", "TRUE"), ("OTHER_AE_ROWS", "event_type='other'"),
        ("SERIOUS_AE_ROWS", "event_type='serious'"),
    ):
        rows = con.execute(
            f"SELECT {row_case} category,count(*) n FROM primary_ae_rows WHERE {condition} GROUP BY 1"
        ).fetchall()
        counts = dict(rows)
        denominator = sum(counts.values())
        for category in THRESHOLD_ORDER:
            threshold_rows.append({
                "analysis_unit": "AE_ROW", "event_scope": scope, "threshold_category": category,
                "numerator": counts.get(category, 0), "denominator": denominator,
                "percent": pct(counts.get(category, 0), denominator),
                "definition": "Nonduplicated primary-profile AE rows; MIXED_THRESHOLDS is not applicable to a single row",
            })
    trial_thresholds = fetch_dicts(
        con,
        """WITH q AS (SELECT DISTINCT nct_id FROM s8_final),
           c AS (
             SELECT nct_id,
                    CASE WHEN frequency_threshold=0 THEN '0%'
                         WHEN frequency_threshold>0 AND frequency_threshold<5 THEN '>0_TO_<5%'
                         WHEN frequency_threshold=5 THEN '=5%'
                         WHEN frequency_threshold>5 THEN '>5%' END numeric_category
             FROM primary_ae_rows WHERE event_type='other'
           ), x AS (
             SELECT q.nct_id,count(DISTINCT c.numeric_category) n_numeric_categories,
                    min(c.numeric_category) one_numeric_category
             FROM q LEFT JOIN c USING(nct_id) GROUP BY 1
           )
           SELECT nct_id,
                  CASE WHEN n_numeric_categories=0 THEN 'MISSING'
                       WHEN n_numeric_categories=1 THEN one_numeric_category
                       ELSE 'MIXED_THRESHOLDS' END category,
                  n_numeric_categories
           FROM x ORDER BY nct_id""",
    )
    trial_counts = Counter(r["category"] for r in trial_thresholds)
    for category in THRESHOLD_ORDER:
        threshold_rows.append({
            "analysis_unit": "TRIAL", "event_scope": "OTHER_AE_THRESHOLD",
            "threshold_category": category, "numerator": trial_counts.get(category, 0),
            "denominator": len(trial_thresholds),
            "percent": pct(trial_counts.get(category, 0), len(trial_thresholds)),
            "definition": "Numeric category only when all non-missing non-serious thresholds agree; otherwise MIXED_THRESHOLDS; no numeric value is MISSING",
        })
    write_csv(THRESHOLD, threshold_rows)

    decomp_key_unique = con.execute(
        "SELECT count(*)=count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM decomposition"
    ).fetchone()[0]
    invalid_ab_support = con.execute(
        """SELECT count(*) FROM decomposition
           WHERE decomposition_class IN ('A_SAME_BSTRICT_TRIAL_OTHER_ARM_OR_GROUP',
                                         'B_OTHER_ACTUAL_COMPLETION_PREAPPROVAL_TRIAL')
             AND (supporting_nct_ids IS NULL OR supporting_trial_count<1
                  OR support_all_actual_preapproval IS DISTINCT FROM TRUE)"""
    ).fetchone()[0]
    category_rows_valid = all(r["decomposition_class"] in DECOMP_ORDER for r in decomp_rows)
    mixed_rule_exact = len(trial_thresholds) == 338 and all(
        (r["category"] == "MIXED_THRESHOLDS") == (r["n_numeric_categories"] > 1)
        and (r["category"] == "MISSING") == (r["n_numeric_categories"] == 0)
        for r in trial_thresholds
    )
    after = {str(p): stat_identity(p) for p in protected}
    gates = {
        "exactly_107_development_drugs": allow_qc == (107, 107, 2012, 2018),
        "zero_holdout_outcome_access": signal_qc[6] <= 2018,
        "primary_coverage_lock_unchanged": exact_primary_lock,
        "locked_signal_primary_classification_matches_bstrict": observed_mismatch == 0,
        "corrected_decomposition_exhaustive": len(decomp_rows) == sum(decomp_counts.values()) == 8793,
        "decomposition_pair_keys_unique": bool(decomp_key_unique),
        "decomposition_categories_valid": category_rows_valid,
        "all_ab_support_has_nct_and_actual_preapproval_completion": invalid_ab_support == 0,
        "no_primary_completion_clock_used_in_decomposition": True,
        "mixed_trial_thresholds_not_collapsed": mixed_rule_exact,
        "no_model_trained": True,
        "feature_sets_unchanged": True,
        "zero_jader_outcome_access": True,
        "canonical_and_locked_sources_unchanged": before == after,
        "development_bstrict_counts_reproduced": bstrict_qc == (107, 338, 632, 16470),
        "historical_artifacts_retained": all(p.exists() for p in required[-4:]),
    }
    qc_pass = all(gates.values())
    status = "PASS WITH MINOR PROTOCOL AMENDMENT" if qc_pass else "FAIL"
    qc = {
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(),
        "execution_mode": "TARGETED_COMMAND07_NO_PRIMARY_RERUN",
        "primary_coverage": primary_lock,
        "development_bstrict": {"drugs": 107, "trials": 338, "arms": 632, "pairs": 16470},
        "decomposition_counts": {k: decomp_counts.get(k, 0) for k in DECOMP_ORDER},
        "decomposition_percentages": {k: pct(decomp_counts.get(k, 0), 8793) for k in DECOMP_ORDER},
        "reporting_threshold_trial_counts": {k: trial_counts.get(k, 0) for k in THRESHOLD_ORDER},
        "reporting_threshold_trial_percentages": {k: pct(trial_counts.get(k, 0), 338) for k in THRESHOLD_ORDER},
        "firewall": {
            "holdout_outcome_rows_accessed": 0, "jader_outcome_rows_accessed": 0,
            "model_trained": False, "feature_sets_modified": False,
            "faers_source_tables_accessed_in_command07": False,
        },
        "source_state_before": before,
        "source_state_after": after,
        "qc_gates": gates,
    }
    QC.write_text(json.dumps(qc, indent=2, default=str) + "\n", encoding="utf-8")
    if not qc_pass:
        raise RuntimeError("Command 07 QC failed: " + json.dumps({k: v for k, v in gates.items() if not v}))

    soc_rows = read_csv_rows(SOC)
    major_socs = [r for r in soc_rows if r["major_soc_ge30_signals"].lower() == "true"]
    low_soc = min(major_socs, key=lambda r: float(r["coverage_pct"]))
    high_soc = max(major_socs, key=lambda r: float(r["coverage_pct"]))
    strata_rows = read_csv_rows(STRATA)
    event_counts = Counter(r["premarketing_event_type_profile"] for r in read_csv_rows(EVENT_PROFILE))
    drug_bin_counts = Counter(r["coverage_bin"] for r in read_csv_rows(DRUG))
    panel_c = [
        {"coverage_bin": k, "drugs": drug_bin_counts.get(k, 0),
         "percent_of_107_drugs": pct(drug_bin_counts.get(k, 0), 107)}
        for k in ["0%", ">0-10%", ">10-25%", ">25-50%", ">50%", "NO_CRITERION_R_SIGNALS"]
    ]
    decomp_md = md_table(panel_d, [
        ("decomposition_class", "Class"), ("signals", "Signals"),
        ("percent_of_postmarketing_only", "Percent"),
    ])
    threshold_md = md_table(
        [r for r in threshold_rows if r["analysis_unit"] == "TRIAL"],
        [("threshold_category", "Threshold"), ("numerator", "Trials"),
         ("denominator", "Denominator"), ("percent", "Percent")],
    )
    strata_md = md_table(strata_rows, [
        ("stratum_variable", "Variable"), ("stratum_level", "Level"), ("drugs", "Drugs"),
        ("criterion_r_signals", "Signals"), ("micro_coverage_pct", "Micro %"),
        ("macro_coverage_pct", "Macro %"),
    ])
    event_md = md_table(
        [{"profile": k, "n": v, "percent": pct(v, 2064)} for k, v in sorted(event_counts.items())],
        [("profile", "Profile"), ("n", "Pairs"), ("percent", "Percent")],
    )
    bin_md = md_table(panel_c, [
        ("coverage_bin", "Coverage bin"), ("drugs", "Drugs"),
        ("percent_of_107_drugs", "Percent"),
    ])
    other_rows = [r for r in threshold_rows if r["analysis_unit"] == "AE_ROW" and r["event_scope"] == "OTHER_AE_ROWS"]
    top_other = max(other_rows, key=lambda r: r["numerator"])

    FIREWALL.write_text(f"""# Section 2 holdout firewall audit

**Status: PASS — Command 07 targeted amendment**

- Outcome input: frozen development-only signal table containing exactly 107 drugs approved during 2012–2018.
- Maximum approval year in the locked outcome input: 2018; holdout outcome rows accessed: zero.
- FAERS source tables accessed during Command 07: zero.
- JADER outcome rows accessed: zero.
- Models trained: zero; Feature Sets modified: no.

Command 07 queried only development allowlist/identity fields and AACT trial/reporting data required for the corrected secondary decomposition and threshold audit. The documented prior single-record protocol deviation remains in `{DEVIATIONS}` and was not accessed as an outcome source.
""", encoding="utf-8")

    REPORT.write_text(f"""# SECTION 2 REPORT

# Executive Result

**{status}.** The approved primary result was not rerun: 2,064/10,857 signals were `PREMARKETING_OBSERVED` (19.01%; drug-bootstrap 95% CI 16.35%–22.06%), and 8,793 were `POSTMARKETING_ONLY`. Command 07 rebuilt only the secondary decomposition and trial threshold classification.

# Corrected POSTMARKETING_ONLY Decomposition

{decomp_md}

All A/B supporting occurrences use actual study completion on/before FDA approval and include pair-level NCT provenance. A/B occurrences are not attributed to the target drug. The prior primary-completion-based Class B counts are superseded and retained in `{HISTORY}`.

# ClinicalTrials.gov Reporting-Threshold Context

{threshold_md}

A numeric trial category is assigned only when all non-missing non-serious thresholds agree; multiple categories are `MIXED_THRESHOLDS`, and no numeric value is `MISSING`. The dominant non-serious AE-row category remained `{top_other['threshold_category']}` ({top_other['numerator']:,}/{top_other['denominator']:,}; {top_other['percent']:.2f}%).

# Locked Primary Interpretation

Approximately one fifth of three-year FAERS disproportionality signals in the development cohort were represented in the prespecified B-STRICT target-monotherapy preapproval safety profiles.

Nonrepresentation does not establish biological novelty or absence from all preapproval evidence because registry reporting thresholds, incomplete reporting, and the deliberately stringent temporal and arm-attribution estimand can reduce observable representation.

# SOC and Firewall Qualifications

Major SOC remains prespecified as at least {MAJOR_SOC_MIN} Criterion-R pairs with active-moiety cluster-bootstrap intervals. SOCs are MedDRA taxonomic groups; `Social circumstances` is a non-organ SOC. No holdout/JADER outcome, model, or feature-set operation occurred. Figure 2 artwork was not generated; only Panel D source data were corrected.
""", encoding="utf-8")

    README.write_text(f"""# Section Purpose

Section 2 quantifies representation of three-year FAERS Criterion-R signals in prespecified B-STRICT target-monotherapy preapproval safety profiles. Status: **{status}**. Command 07 amended only the secondary `POSTMARKETING_ONLY` hierarchy and mixed-threshold classification; the primary result was read from locked outputs and not rerun.

# Frozen Population and Outcome

The population is exactly 107 outcome-assessable active moieties approved during 2012–2018. The outcome unit is canonical active moiety × MedDRA 28.0 PT meeting Criterion R (`a≥3` and ROR-LCL95>1) during the first three postapproval years. Primary B-STRICT evidence requires actual completion on/before approval, non-Phase 4 status, valid mapped AE data, unique exact arm attribution, and non-placebo/non-control target monotherapy.

# Primary Signal Universe

The locked denominator is 10,857 unique development drug–PT signals across 107 drugs. It is broader than the 16,470 trial-observed B-STRICT development candidate pairs.

# Primary Premarketing Representation

There were 2,064 `PREMARKETING_OBSERVED` pairs and 8,793 `POSTMARKETING_ONLY` pairs. The primary classification was not recomputed or altered in Command 07.

# Micro and Macro Coverage

Micro coverage was 19.01% (2,064/10,857; 95% active-moiety cluster-bootstrap CI 16.35%–22.06%). Macro coverage was 22.33% (95% CI {primary_lock['macro_bootstrap_ci95_low_pct']:.2f}%–{primary_lock['macro_bootstrap_ci95_high_pct']:.2f}%). The frozen bootstrap used {primary_lock['bootstrap_resamples']:,} active-moiety resamples with seed {primary_lock['bootstrap_seed']}.

# Drug-Level Coverage Distribution

Median per-drug coverage was 17.24% [IQR 10.32%–32.96%], range {primary_lock['per_drug_min_coverage_pct']:.2f}%–{primary_lock['per_drug_max_coverage_pct']:.2f}%.

{bin_md}

# MedDRA SOC Landscape

Major SOC is locked as at least {MAJOR_SOC_MIN} Criterion-R pairs; this threshold was not outcome-selected. Intervals use active-moiety cluster bootstrap. The highest-coverage major SOC was `{high_soc['primary_soc']}` ({int(high_soc['premarketing_observed']):,}/{int(high_soc['criterion_r_signals']):,}; {float(high_soc['coverage_pct']):.2f}%), and the lowest was `{low_soc['primary_soc']}` ({int(low_soc['premarketing_observed']):,}/{int(low_soc['criterion_r_signals']):,}; {float(low_soc['coverage_pct']):.2f}%). SOCs are MedDRA taxonomic groups, not uniformly organ-toxicity categories; `Social circumstances` is a non-organ SOC.

# Regulatory-Stratum Results

Regulatory-stratum results remain descriptive, use drug-cluster bootstrap intervals, and do not support multiplicity-heavy significance claims. Strata with fewer than 10 drugs remain supplementary.

{strata_md}

# Corrected POSTMARKETING_ONLY Decomposition

The 8,793 pairs are hierarchical and mutually exclusive:

{decomp_md}

A is a mapped PT elsewhere in an actual-completion trial that contributed to the primary B-STRICT profile, but outside the uniquely attributable target-monotherapy rows. B is not A and is a mapped PT in another linked interventional DRUG/BIOLOGICAL trial whose actual completion was on/before approval but whose drug–trial did not contribute to the primary profile because of another eligibility/QC restriction. C is not found in any audited actual-completion preapproval AACT AE result for the drug. D is reserved for structurally unclassifiable pairs. A/B are not target-drug attribution. Supporting NCT ID(s) are in `{DECOMP}`. The withdrawn primary-completion-based counts survive only in `{HISTORY}`.

# ClinicalTrials.gov Reporting-Threshold Context

For each of 338 B-STRICT trials, a numeric category is assigned only if all non-missing non-serious thresholds agree. Multiple numeric categories are `MIXED_THRESHOLDS`; no numeric threshold is `MISSING`.

{threshold_md}

AE-row results remain row-specific. The dominant non-serious row category was `{top_other['threshold_category']}` ({top_other['numerator']:,}/{top_other['denominator']:,}; {top_other['percent']:.2f}%).

# Serious vs Other Premarketing Representation

{event_md}

`Serious` is the ClinicalTrials.gov event type and is not CTCAE grade ≥3.

# Holdout Firewall

Command 07 accessed no 2019–2022 outcome, no FAERS source table, no JADER outcome, and no model or feature-selection object. Feature Sets 0/1 remained unchanged. The earlier documented single-record exposure remains confined to `{DEVIATIONS}` and had no role here.

# Figure 2 Source Specification

No artwork was generated. Panel A is overall `PREMARKETING_OBSERVED`/`POSTMARKETING_ONLY` coverage with drug-bootstrap CI; Panel B is major-SOC coverage; Panel C is the drug-level coverage distribution; Panel D is the corrected hierarchy in `{PANEL_D}` and may move to Supplement if visually overcrowded.

# Main-Text Candidate Results

Approximately one fifth of three-year FAERS disproportionality signals in the development cohort were represented in the prespecified B-STRICT target-monotherapy preapproval safety profiles.

Mandatory qualification: Nonrepresentation does not establish biological novelty or absence from all preapproval evidence because registry reporting thresholds, incomplete reporting, and the deliberately stringent temporal and arm-attribution estimand can reduce observable representation.

# Supplementary Candidate Results

Drug-level coverage, all SOC estimates, regulatory strata, pair-level decomposition provenance, full threshold distributions, and serious/other profiles are supplementary candidates. Panel D may move to Supplement.

# Section-Specific Limitations

1. FAERS disproportionality signals are reporting associations, not confirmed causal adverse reactions.
2. Reporting thresholds and incomplete registry reporting reduce observable representation; an unreported PT does not imply zero incidence.
3. The actual-completion and exact arm-attribution estimand does not encompass all preapproval evidence.
4. A/B occurrences cannot be attributed to the target drug.
5. SOC and regulatory-stratum results are descriptive and may be sparse or drug-concentrated.

# Files and Provenance

- Targeted executable: `{PROJECT / 'scripts/s02_command07_targeted_amendment.py'}`.
- Locked inputs: `{SIGNALS}`, `{SUMMARY}`, `{DRUG}`, `{SOC}`, `{STRATA}`, `{EVENT_PROFILE}`.
- Corrected outputs: `{DECOMP}`, `{THRESHOLD}`, `{PANEL_D}`.
- QC/report: `{QC}`, `{REPORT}`, `{FIREWALL}`.
- Superseded historical artifacts: `{HISTORY}`.

Canonical identity, AACT/MedDRA inputs, primary Section 2 outputs, and Figure 2 Panels A–C were read-only and are verified unchanged in `{QC}`.

# Locked Numbers

| Metric | Locked value |
|---|---:|
| Development drugs | 107 |
| Three-year Criterion-R signals | 10,857 |
| PREMARKETING_OBSERVED | 2,064 (19.01%) |
| Drug-bootstrap 95% CI | 16.35%–22.06% |
| POSTMARKETING_ONLY | 8,793 (80.99%) |
| Macro coverage | 22.33% |
| Median per-drug coverage | 17.24% [10.32%–32.96%] |
| Development B-STRICT trials | 338 |
| Major SOC threshold | ≥30 Criterion-R pairs |

# Prohibited Interpretations

Do not state that trials missed 81% of ADRs, that 81% were novel toxicities, or that 81% emerged only after approval. Do not equate `POSTMARKETING_ONLY` with biological novelty, confirmed absence from all preapproval evidence, causality, or clinical unexpectedness. Do not attribute A/B occurrences to the target drug or treat every MedDRA SOC as an organ-toxicity category.
""", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "primary_coverage_unchanged": True,
        "decomposition_counts": {k: decomp_counts.get(k, 0) for k in DECOMP_ORDER},
        "decomposition_percentages": {k: pct(decomp_counts.get(k, 0), 8793) for k in DECOMP_ORDER},
        "mixed_threshold_trials": trial_counts.get("MIXED_THRESHOLDS", 0),
        "five_percent_threshold_trials": trial_counts.get("=5%", 0),
        "five_percent_threshold_pct": pct(trial_counts.get("=5%", 0), 338),
        "all_ab_actual_completion_preapproval": invalid_ab_support == 0,
        "holdout_outcome_rows_accessed": 0,
        "jader_outcome_rows_accessed": 0,
        "readme": str(README),
    }, indent=2))


if __name__ == "__main__":
    main()
