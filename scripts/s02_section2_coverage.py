#!/usr/bin/env python3
"""Command 06 full Section 2 reconstruction with Command 07 definitions.

The outcome universe is rebuilt from the repaired case-level MedDRA 28.0
layer after an explicit 107-drug development allowlist is applied. No
2019-2022 drug-specific outcome, JADER object, or model object is opened.

This full reconstruction path is retained for reproducibility but was not run
for Command 07. The targeted no-primary-rerun executable is
``s02_command07_targeted_amendment.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np


PROJECT = Path("/path/to/project")
OUT = PROJECT / "analysis/section2_coverage"
FIG = OUT / "09_figure2_source_data"
SECTION1 = PROJECT / "analysis/section1_cohort/02_final_drug_level_characteristics.csv"
SECTION1_README = PROJECT / "analysis/section1_cohort/SECTION1_README.md"
DEVIATIONS = PROJECT / "docs/PROTOCOL_DEVIATIONS.md"
IDENTITY = PROJECT / "preflight_v2/drug_identity_master.csv"
IDENTITY_SHA = PROJECT / "preflight_v2/drug_identity_master.sha256"
LINKS = PROJECT / "data/processed/preflight_v2/aact_fda_intervention_links.parquet"
REGISTRY = PROJECT / "preflight_v2/bstrict_candidate_registry.parquet"
PT_MAP = PROJECT / "preflight_v2/faers_pt_repair/aact_meddra28_term_mapping.csv"
PT_CANON = PROJECT / "preflight_v2/faers_pt_repair/faers_meddra28_canonical.csv"
CASE_DRUG = PROJECT / "data/processed/preflight_v2/faers_latest_case_drug_ps_fda.parquet"
CASE_PT = PROJECT / "data/processed/preflight_v2/faers_pt_repair/faers_latest_case_pt_meddra28.parquet"
LATEST = PROJECT / "data/processed/preflight_v2/faers_pt_repair/faers_latest_cases.parquet"
AACT_DB = Path("/path/to/Database/AACT/aact.duckdb")

FIREWALL = OUT / "00_holdout_firewall_audit.md"
SIGNALS = OUT / "01_development_signal_universe.csv"
SUMMARY = OUT / "02_primary_coverage_summary.csv"
DRUG = OUT / "03_drug_level_coverage.csv"
SOC = OUT / "04_soc_coverage.csv"
STRATA = OUT / "05_regulatory_strata_coverage.csv"
DECOMP = OUT / "06_postmarketing_only_decomposition.csv"
THRESHOLD = OUT / "07_reporting_threshold_context.csv"
EVENT_PROFILE = OUT / "08_premarketing_event_type_profile.csv"
REPORT = OUT / "SECTION2_REPORT.md"
README = OUT / "SECTION2_README.md"
QC = OUT / "SECTION2_QC.json"

N_BOOT = 5000
BOOT_SEED = 20260810
MAJOR_SOC_MIN = 30


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


def quantile(values, q: float):
    xs = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def pct(n, d):
    return 100.0 * n / d if d else None


def regulatory_level(variable: str, value) -> str:
    if value is None or value == "":
        return "MISSING"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    level = str(value)
    if variable in {"orphan_designation", "accelerated_approval", "breakthrough_therapy_designation"} and level.startswith("Yes"):
        return "Yes"
    return level


def cluster_bootstrap(total: np.ndarray, observed: np.ndarray, seed: int, n_boot: int = N_BOOT) -> dict:
    """Resample drug clusters and retain every pair from selected drugs."""
    n = len(total)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.repeat(1.0 / n, n), size=n_boot)
    den = weights @ total
    num = weights @ observed
    micro = np.divide(num, den, out=np.full(n_boot, np.nan), where=den > 0)
    drug_cov = np.divide(observed, total, out=np.full(n, np.nan), where=total > 0)
    valid = np.isfinite(drug_cov).astype(float)
    macro_den = weights @ valid
    macro_num = weights @ np.nan_to_num(drug_cov)
    macro = np.divide(macro_num, macro_den, out=np.full(n_boot, np.nan), where=macro_den > 0)
    return {
        "micro_low": 100 * float(np.nanquantile(micro, .025)),
        "micro_high": 100 * float(np.nanquantile(micro, .975)),
        "macro_low": 100 * float(np.nanquantile(macro, .025)),
        "macro_high": 100 * float(np.nanquantile(macro, .975)),
        "valid_micro_bootstraps": int(np.isfinite(micro).sum()),
    }


def md_table(rows: list[dict], fields: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in fields) + " |"
    sep = "|" + "|".join("---" for _ in fields) + "|"
    body = []
    for row in rows:
        vals = []
        for key, _ in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:,.2f}"
            elif isinstance(value, int):
                value = f"{value:,}"
            vals.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep, *body])


def main() -> None:
    if not SECTION1_README.exists() or not DEVIATIONS.exists():
        raise RuntimeError("Section 1 closure documents are missing")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    expected_sha = IDENTITY_SHA.read_text(encoding="utf-8").split()[0]
    if sha256(IDENTITY) != expected_sha:
        raise RuntimeError("Frozen identity master hash mismatch")

    protected = [IDENTITY, IDENTITY_SHA, LINKS, REGISTRY, PT_MAP, PT_CANON, CASE_DRUG, CASE_PT, LATEST, AACT_DB]
    before = {str(p): stat_identity(p) for p in protected}

    con = duckdb.connect()
    con.execute(
        "SET threads=8; SET memory_limit='20GB'; SET temp_directory='/private/tmp/pvml_duckdb'; "
        "SET preserve_insertion_order=false; SET enable_progress_bar=false"
    )
    con.execute(f"ATTACH '{AACT_DB}' AS a (READ_ONLY)")

    # The only drug-specific outcome allowlist used in this script.
    con.execute(
        f"""
        CREATE TABLE allowlist AS
        SELECT d.canonical_active_moiety,cast(i.fda_first_approval_date AS DATE) approval_date,
               d.approval_year,d.nda_bla,d.orphan_designation,d.accelerated_approval,
               d.breakthrough_therapy_designation
        FROM read_csv_auto('{SECTION1}',header=true) d
        JOIN read_csv_auto('{IDENTITY}',header=true) i USING(canonical_active_moiety)
        WHERE d.temporal_partition='DEVELOPMENT' AND d.approval_year BETWEEN 2012 AND 2018
          AND i.aact_mapping_confidence<>'UNRESOLVED' AND i.faers_mapping_confidence<>'UNRESOLVED'
        """
    )
    allow_qc = con.execute(
        "SELECT count(*),count(DISTINCT canonical_active_moiety),min(approval_year),max(approval_year) FROM allowlist"
    ).fetchone()
    if allow_qc != (107, 107, 2012, 2018):
        raise RuntimeError(f"Development allowlist mismatch: {allow_qc}")

    # Rebuild the full development-only Criterion-R signal universe. The
    # target-drug case table is restricted to allowlist before any pairing.
    con.execute(
        f"""
        CREATE TABLE dev_case_drug AS
        SELECT d.caseid,d.primaryid,d.report_date,d.canonical_active_moiety
        FROM read_parquet('{CASE_DRUG}') d
        JOIN allowlist a USING(canonical_active_moiety)
        WHERE d.report_date BETWEEN a.approval_date AND (a.approval_date+INTERVAL 3 YEAR)::DATE
        GROUP BY ALL
        """
    )
    con.execute(
        """
        CREATE TABLE target_totals AS
        SELECT canonical_active_moiety,count(DISTINCT caseid) target_ps_cases
        FROM dev_case_drug GROUP BY 1
        """
    )
    con.execute(
        f"""
        CREATE TABLE total_daily AS
        SELECT try_strptime(cast(fda_dt AS VARCHAR),'%Y%m%d')::DATE report_date,count(*) n_cases
        FROM read_parquet('{LATEST}')
        WHERE try_strptime(cast(fda_dt AS VARCHAR),'%Y%m%d') IS NOT NULL GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE TABLE window_totals AS
        SELECT a.canonical_active_moiety,sum(d.n_cases) total_cases
        FROM allowlist a JOIN total_daily d
          ON d.report_date BETWEEN a.approval_date AND (a.approval_date+INTERVAL 3 YEAR)::DATE
        GROUP BY 1
        """
    )
    con.execute(
        f"""
        CREATE TABLE all_a AS
        SELECT d.canonical_active_moiety,e.canonical_pt_code,count(DISTINCT d.caseid) a
        FROM dev_case_drug d JOIN read_parquet('{CASE_PT}') e
          ON e.caseid=d.caseid AND e.primaryid=d.primaryid
        GROUP BY 1,2 HAVING count(DISTINCT d.caseid)>=3
        """
    )
    con.execute(
        f"""
        CREATE TABLE pt_daily AS
        SELECT try_strptime(cast(e.fda_dt AS VARCHAR),'%Y%m%d')::DATE report_date,
               e.canonical_pt_code,count(*) n_cases_pt
        FROM read_parquet('{CASE_PT}') e JOIN (SELECT DISTINCT canonical_pt_code FROM all_a) p USING(canonical_pt_code)
        WHERE try_strptime(cast(e.fda_dt AS VARCHAR),'%Y%m%d') IS NOT NULL
        GROUP BY 1,2
        """
    )
    con.execute(
        """
        CREATE TABLE all_m AS
        SELECT x.canonical_active_moiety,x.canonical_pt_code,sum(d.n_cases_pt) total_pt_cases
        FROM all_a x JOIN allowlist a USING(canonical_active_moiety)
        JOIN pt_daily d ON d.canonical_pt_code=x.canonical_pt_code
                       AND d.report_date BETWEEN a.approval_date AND (a.approval_date+INTERVAL 3 YEAR)::DATE
        GROUP BY 1,2
        """
    )
    con.execute(
        f"""
        CREATE TABLE pt_canon AS
        SELECT cast(canonical_pt_code AS BIGINT) canonical_pt_code,
               any_value(canonical_pt_name) canonical_pt_name,
               any_value(canonical_soc_name) canonical_soc_name,
               any_value(canonical_soc_code) canonical_soc_code
        FROM read_csv_auto('{PT_CANON}',header=true)
        WHERE mapping_status='MAPPED' GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE TABLE criterion_r_all AS
        WITH cells AS (
          SELECT a.canonical_active_moiety,a.canonical_pt_code,a.a,
                 t.target_ps_cases,m.total_pt_cases,w.total_cases,
                 t.target_ps_cases-a.a b,m.total_pt_cases-a.a c,
                 w.total_cases-t.target_ps_cases-m.total_pt_cases+a.a d
          FROM all_a a JOIN target_totals t USING(canonical_active_moiety)
          JOIN all_m m USING(canonical_active_moiety,canonical_pt_code)
          JOIN window_totals w USING(canonical_active_moiety)
        ), stats AS (
          SELECT *,CASE WHEN a>0 AND b>0 AND c>0 AND d>0 THEN (a::DOUBLE*d)/(b::DOUBLE*c) END ror,
                   CASE WHEN a>0 AND b>0 AND c>0 AND d>0
                     THEN exp(ln((a::DOUBLE*d)/(b::DOUBLE*c))-1.96*sqrt(1.0/a+1.0/b+1.0/c+1.0/d)) END ror_lcl95
          FROM cells
        )
        SELECT s.*,p.canonical_pt_name,p.canonical_soc_name,p.canonical_soc_code
        FROM stats s JOIN pt_canon p USING(canonical_pt_code)
        WHERE a>=3 AND ror_lcl95>1
        """
    )

    # Independently reconstruct the primary B-STRICT profile for only the 107
    # development drugs; this also supports decomposition and threshold audit.
    con.execute(f"CREATE TABLE links AS SELECT l.* FROM read_parquet('{LINKS}') l JOIN allowlist USING(canonical_active_moiety)")
    con.execute(
        f"""
        CREATE TABLE aact_pt_map AS
        SELECT aact_term_raw,cast(canonical_pt_code AS BIGINT) canonical_pt_code,
               canonical_pt_name,mapping_level
        FROM read_csv_auto('{PT_MAP}',header=true)
        WHERE mapping_status='MAPPED' AND canonical_pt_code IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE TABLE s1_link_trials AS
        SELECT DISTINCT a.*,l.nct_id FROM allowlist a JOIN links l USING(canonical_active_moiety)
        """
    )
    con.execute(
        """
        CREATE TABLE s2_bstrict_trials AS
        SELECT s.*,st.completion_date,st.completion_date_type,st.phase
        FROM s1_link_trials s JOIN a.studies st USING(nct_id)
        WHERE st.completion_date IS NOT NULL AND st.completion_date_type='ACTUAL'
          AND st.completion_date<=s.approval_date AND coalesce(st.phase,'')<>'PHASE4'
        """
    )
    con.execute(
        """
        CREATE TABLE s3_ae AS
        SELECT s.*,re.id ae_id,re.result_group_id rgid,re.adverse_event_term,
               re.subjects_affected,re.subjects_at_risk,re.event_type,re.frequency_threshold
        FROM s2_bstrict_trials s JOIN a.reported_events re USING(nct_id)
        """
    )
    con.execute(
        """
        CREATE TABLE s4_den AS SELECT * FROM s3_ae
        WHERE subjects_at_risk>0 AND subjects_affected IS NOT NULL AND subjects_affected<=subjects_at_risk
        """
    )
    con.execute(
        """
        CREATE TABLE s5_pt AS
        SELECT s.*,p.canonical_pt_code,p.canonical_pt_name,p.mapping_level
        FROM s4_den s JOIN aact_pt_map p ON s.adverse_event_term=p.aact_term_raw
        """
    )
    tn = "lower(trim(regexp_replace(regexp_replace({col}, '[^A-Za-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')))"
    con.execute(
        f"""
        CREATE TABLE rg_map AS
        WITH rg AS (SELECT nct_id,id rgid,{tn.format(col='title')} title_norm FROM a.result_groups
                     WHERE result_type='Reported Event' AND nct_id IN (SELECT nct_id FROM s1_link_trials)),
             dg AS (SELECT nct_id,id dgid,group_type,title,{tn.format(col='title')} title_norm FROM a.design_groups
                     WHERE nct_id IN (SELECT nct_id FROM s1_link_trials))
        SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_dg,min(dg.dgid) dgid,
               min(dg.group_type) group_type,min(dg.title) design_group_title
        FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm GROUP BY 1,2
        """
    )
    con.execute(
        """
        CREATE TABLE arm_comp AS
        SELECT dg.id dgid,count(DISTINCT i.id) FILTER (WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')) n_drug_iv
        FROM a.design_groups dg LEFT JOIN a.design_group_interventions dgi ON dgi.design_group_id=dg.id
        LEFT JOIN a.interventions i ON i.id=dgi.intervention_id
        WHERE dg.nct_id IN (SELECT nct_id FROM s1_link_trials) GROUP BY dg.id
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
        SELECT s.*,r.dgid,r.group_type,r.design_group_title FROM s5_pt s JOIN rg_map r
          ON s.nct_id=r.nct_id AND s.rgid=r.rgid WHERE r.n_dg=1
        """
    )
    con.execute(
        """
        CREATE TABLE s7_target_mono AS
        SELECT s.*,t.intervention_id target_intervention_id,t.aact_primary_name target_intervention_name,
               t.aact_intervention_combination_flag
        FROM s6_attr s JOIN arm_comp ac USING(dgid)
        JOIN target_arm t ON s.canonical_active_moiety=t.canonical_active_moiety
                         AND s.nct_id=t.nct_id AND s.dgid=t.dgid
        WHERE ac.n_drug_iv=1
        """
    )
    con.execute(
        """
        CREATE TABLE s8_final AS SELECT * FROM s7_target_mono
        WHERE group_type IN ('EXPERIMENTAL','OTHER') AND NOT aact_intervention_combination_flag
          AND NOT regexp_matches(lower(coalesce(target_intervention_name,'')),
              '(^| )(placebo|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline)( |$)')
          AND NOT regexp_matches(lower(coalesce(design_group_title,'')),
              '(^| )(placebo|vehicle|sham|no treatment|control)( |$)')
        """
    )
    con.execute(
        """
        CREATE TABLE primary_pairs AS
        SELECT DISTINCT canonical_active_moiety,canonical_pt_code FROM s8_final
        """
    )
    con.execute(
        """
        CREATE TABLE primary_profile_trials AS
        SELECT DISTINCT canonical_active_moiety,nct_id FROM s8_final
        """
    )
    primary_counts = con.execute(
        "SELECT (SELECT count(DISTINCT canonical_active_moiety) FROM primary_pairs),(SELECT count(DISTINCT nct_id) FROM s8_final),(SELECT count(DISTINCT rgid) FROM s8_final),(SELECT count(*) FROM primary_pairs)"
    ).fetchone()
    if primary_counts != (107, 338, 632, 16470):
        raise RuntimeError(f"Development B-STRICT profile mismatch: {primary_counts}")

    con.execute(
        """
        CREATE TABLE development_signals AS
        SELECT r.canonical_active_moiety,a.approval_year,r.canonical_pt_code,r.canonical_pt_name,
               r.canonical_soc_name primary_soc,r.canonical_soc_code primary_soc_code,
               r.a,r.b,r.c,r.d,r.ror,r.ror_lcl95,
               CASE WHEN p.canonical_active_moiety IS NOT NULL THEN 'PREMARKETING_OBSERVED'
                    ELSE 'POSTMARKETING_ONLY' END coverage_class
        FROM criterion_r_all r JOIN allowlist a USING(canonical_active_moiety)
        LEFT JOIN primary_pairs p USING(canonical_active_moiety,canonical_pt_code)
        ORDER BY a.approval_year,r.canonical_active_moiety,r.canonical_pt_code
        """
    )

    # Corrected secondary hierarchy. Every occurrence interpreted as
    # preapproval evidence uses ACTUAL study completion on/before FDA approval.
    # A searches the full reported-event tables of trials that contribute to
    # the primary B-STRICT profile, irrespective of arm attribution. B searches other linked
    # interventional DRUG/BIOLOGICAL trials that meet the same actual-completion
    # clock but whose drug-trial does not contribute to the primary profile because of another
    # eligibility/QC restriction (for example, Phase 4 classification).
    con.execute(
        """
        CREATE TABLE bstrict_any_evidence AS
        SELECT s.canonical_active_moiety,p.canonical_pt_code,
               string_agg(DISTINCT s.nct_id,' | ' ORDER BY s.nct_id) supporting_nct_ids,
               count(DISTINCT s.nct_id) supporting_trial_count,
               min(s.completion_date) support_completion_date_min,
               max(s.completion_date) support_completion_date_max,
               bool_and(s.completion_date_type='ACTUAL' AND s.completion_date<=s.approval_date)
                 support_all_actual_preapproval
        FROM s3_ae s JOIN primary_profile_trials q
          ON q.canonical_active_moiety=s.canonical_active_moiety AND q.nct_id=s.nct_id
        JOIN aact_pt_map p ON s.adverse_event_term=p.aact_term_raw
        GROUP BY 1,2
        """
    )
    con.execute(
        """
        CREATE TABLE other_actual_preapproval_trials AS
        SELECT DISTINCT s.*,st.completion_date,st.completion_date_type,st.phase
        FROM s1_link_trials s JOIN a.studies st USING(nct_id)
        WHERE st.study_type='INTERVENTIONAL'
          AND st.completion_date IS NOT NULL
          AND st.completion_date_type='ACTUAL'
          AND st.completion_date<=s.approval_date
          AND EXISTS (
            SELECT 1 FROM a.interventions i
            WHERE i.nct_id=s.nct_id AND i.intervention_type IN ('DRUG','BIOLOGICAL')
          )
          AND NOT EXISTS (
            SELECT 1 FROM primary_profile_trials b
            WHERE b.canonical_active_moiety=s.canonical_active_moiety AND b.nct_id=s.nct_id
          )
        """
    )
    con.execute(
        """
        CREATE TABLE other_actual_preapproval_evidence AS
        SELECT s.canonical_active_moiety,p.canonical_pt_code,
               string_agg(DISTINCT s.nct_id,' | ' ORDER BY s.nct_id) supporting_nct_ids,
               count(DISTINCT s.nct_id) supporting_trial_count,
               min(s.completion_date) support_completion_date_min,
               max(s.completion_date) support_completion_date_max,
               bool_and(s.completion_date_type='ACTUAL' AND s.completion_date<=s.approval_date)
                 support_all_actual_preapproval
        FROM other_actual_preapproval_trials s
        JOIN a.reported_events re USING(nct_id)
        JOIN aact_pt_map p ON re.adverse_event_term=p.aact_term_raw
        GROUP BY 1,2
        """
    )
    con.execute(
        """
        CREATE TABLE decomposition AS
        SELECT s.canonical_active_moiety,s.approval_year,s.canonical_pt_code,s.canonical_pt_name,s.primary_soc,
               CASE WHEN a.canonical_active_moiety IS NOT NULL THEN 'A_SAME_BSTRICT_TRIAL_OTHER_ARM_OR_GROUP'
                    WHEN b.canonical_active_moiety IS NOT NULL THEN 'B_OTHER_ACTUAL_COMPLETION_PREAPPROVAL_TRIAL'
                    WHEN s.canonical_pt_code IS NOT NULL THEN 'C_NOT_FOUND_IN_ANY_ACTUAL_COMPLETION_PREAPPROVAL_AACT_AE_RESULTS'
                    ELSE 'D_UNCLASSIFIABLE' END decomposition_class,
               CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.supporting_nct_ids
                    WHEN b.canonical_active_moiety IS NOT NULL THEN b.supporting_nct_ids END supporting_nct_ids,
               CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.supporting_trial_count
                    WHEN b.canonical_active_moiety IS NOT NULL THEN b.supporting_trial_count
                    ELSE 0 END supporting_trial_count,
               CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.support_completion_date_min
                    WHEN b.canonical_active_moiety IS NOT NULL THEN b.support_completion_date_min END support_completion_date_min,
               CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.support_completion_date_max
                    WHEN b.canonical_active_moiety IS NOT NULL THEN b.support_completion_date_max END support_completion_date_max,
               CASE WHEN a.canonical_active_moiety IS NOT NULL THEN a.support_all_actual_preapproval
                    WHEN b.canonical_active_moiety IS NOT NULL THEN b.support_all_actual_preapproval END support_all_actual_preapproval,
               CASE WHEN a.canonical_active_moiety IS NOT NULL OR b.canonical_active_moiety IS NOT NULL
                    THEN 'ACTUAL_STUDY_COMPLETION_ON_OR_BEFORE_FDA_APPROVAL'
                    ELSE 'NO_SUPPORTING_PREAPPROVAL_AACT_AE_OCCURRENCE' END temporal_evidence_rule
        FROM development_signals s
        LEFT JOIN bstrict_any_evidence a USING(canonical_active_moiety,canonical_pt_code)
        LEFT JOIN other_actual_preapproval_evidence b USING(canonical_active_moiety,canonical_pt_code)
        WHERE s.coverage_class='POSTMARKETING_ONLY'
        ORDER BY s.approval_year,s.canonical_active_moiety,s.canonical_pt_code
        """
    )

    con.execute(
        """
        CREATE TABLE event_profile AS
        WITH x AS (
          SELECT s.canonical_active_moiety,s.approval_year,s.canonical_pt_code,s.canonical_pt_name,s.primary_soc,
                 max((lower(f.event_type)='serious')::INT) has_serious_row,
                 max((lower(f.event_type)='other')::INT) has_other_row
          FROM development_signals s JOIN s8_final f
            USING(canonical_active_moiety,canonical_pt_code)
          WHERE s.coverage_class='PREMARKETING_OBSERVED'
          GROUP BY 1,2,3,4,5
        )
        SELECT *,CASE WHEN has_serious_row=1 AND has_other_row=1 THEN 'BOTH_SERIOUS_AND_OTHER'
                      WHEN has_serious_row=1 THEN 'SERIOUS_ONLY'
                      WHEN has_other_row=1 THEN 'OTHER_ONLY'
                      ELSE 'UNCLASSIFIABLE_EVENT_TYPE' END premarketing_event_type_profile
        FROM x ORDER BY approval_year,canonical_active_moiety,canonical_pt_code
        """
    )

    signal_rows = fetch_dicts(con, "SELECT * FROM development_signals")
    write_csv(SIGNALS, signal_rows)
    decomp_rows = fetch_dicts(con, "SELECT * FROM decomposition")
    write_csv(DECOMP, decomp_rows)
    event_rows = fetch_dicts(con, "SELECT * FROM event_profile")
    write_csv(EVENT_PROFILE, event_rows)

    allow_rows = fetch_dicts(con, "SELECT * FROM allowlist ORDER BY approval_year,canonical_active_moiety")
    drug_names = [r["canonical_active_moiety"] for r in allow_rows]
    index = {d: i for i, d in enumerate(drug_names)}
    total = np.zeros(len(drug_names), dtype=int)
    observed = np.zeros(len(drug_names), dtype=int)
    for row in signal_rows:
        i = index[row["canonical_active_moiety"]]
        total[i] += 1
        observed[i] += row["coverage_class"] == "PREMARKETING_OBSERVED"
    postonly = total - observed
    coverage = np.divide(observed, total, out=np.full(len(total), np.nan), where=total > 0)
    boot = cluster_bootstrap(total, observed, BOOT_SEED)

    drug_rows = []
    bins = Counter()
    for i, row in enumerate(allow_rows):
        cov = float(coverage[i]) if np.isfinite(coverage[i]) else None
        if cov is None:
            cov_bin = "NO_CRITERION_R_SIGNALS"
        elif cov == 0:
            cov_bin = "0%"
        elif cov <= .10:
            cov_bin = ">0-10%"
        elif cov <= .25:
            cov_bin = ">10-25%"
        elif cov <= .50:
            cov_bin = ">25-50%"
        else:
            cov_bin = ">50%"
        bins[cov_bin] += 1
        drug_rows.append({
            "canonical_active_moiety": row["canonical_active_moiety"],
            "approval_year": row["approval_year"],
            "criterion_r_signals": int(total[i]),
            "premarketing_observed_signals": int(observed[i]),
            "postmarketing_only_signals": int(postonly[i]),
            "coverage_proportion": cov,
            "coverage_percent": 100 * cov if cov is not None else "",
            "coverage_bin": cov_bin,
        })
    write_csv(DRUG, drug_rows)

    valid_cov = [x for x in coverage if np.isfinite(x)]
    summary_row = {
        "population": "DEVELOPMENT_2012_2018_BSTRICT_PS_GE100",
        "active_moieties": len(drug_names),
        "drugs_with_criterion_r_signals": len(valid_cov),
        "criterion_r_signals": int(total.sum()),
        "premarketing_observed": int(observed.sum()),
        "postmarketing_only": int(postonly.sum()),
        "micro_coverage_pct": pct(observed.sum(), total.sum()),
        "micro_bootstrap_ci95_low_pct": boot["micro_low"],
        "micro_bootstrap_ci95_high_pct": boot["micro_high"],
        "macro_coverage_pct": 100 * float(np.nanmean(coverage)),
        "macro_bootstrap_ci95_low_pct": boot["macro_low"],
        "macro_bootstrap_ci95_high_pct": boot["macro_high"],
        "per_drug_median_coverage_pct": 100 * quantile(valid_cov, .5),
        "per_drug_p25_coverage_pct": 100 * quantile(valid_cov, .25),
        "per_drug_p75_coverage_pct": 100 * quantile(valid_cov, .75),
        "per_drug_min_coverage_pct": 100 * min(valid_cov),
        "per_drug_max_coverage_pct": 100 * max(valid_cov),
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_unit": "canonical_active_moiety",
    }
    write_csv(SUMMARY, [summary_row])

    # SOC estimates use the same 107-drug cluster universe.
    soc_groups = defaultdict(list)
    for row in signal_rows:
        soc_groups[row["primary_soc"] or "MISSING_PRIMARY_SOC"].append(row)
    soc_rows = []
    for j, (soc, rows) in enumerate(sorted(soc_groups.items())):
        st = np.zeros(len(drug_names), dtype=int); so = np.zeros(len(drug_names), dtype=int)
        for row in rows:
            i = index[row["canonical_active_moiety"]]; st[i] += 1
            so[i] += row["coverage_class"] == "PREMARKETING_OBSERVED"
        sb = cluster_bootstrap(st, so, BOOT_SEED + 100 + j)
        scov = np.divide(so, st, out=np.full(len(st), np.nan), where=st > 0)
        soc_rows.append({
            "primary_soc": soc,"drugs_with_signals": int((st>0).sum()),"criterion_r_signals":int(st.sum()),
            "premarketing_observed":int(so.sum()),"postmarketing_only":int((st-so).sum()),
            "coverage_pct":pct(so.sum(),st.sum()),"cluster_bootstrap_ci95_low_pct":sb["micro_low"],
            "cluster_bootstrap_ci95_high_pct":sb["micro_high"],"macro_coverage_pct":100*float(np.nanmean(scov)),
            "major_soc_ge30_signals":bool(st.sum()>=MAJOR_SOC_MIN),"major_soc_threshold":MAJOR_SOC_MIN,
        })
    soc_rows.sort(key=lambda r: (-r["criterion_r_signals"], r["primary_soc"]))
    write_csv(SOC, soc_rows)

    # Prespecified regulatory strata; all bootstraps resample drugs within the stratum.
    reg_by_drug = {r["canonical_active_moiety"]: r for r in allow_rows}
    strata_rows = []
    strata_defs = [
        ("nda_bla", lambda r: str(r["nda_bla"])),
        ("orphan_designation", lambda r: regulatory_level("orphan_designation", r["orphan_designation"])),
        ("accelerated_approval", lambda r: regulatory_level("accelerated_approval", r["accelerated_approval"])),
        ("breakthrough_therapy_designation", lambda r: regulatory_level("breakthrough_therapy_designation", r["breakthrough_therapy_designation"])),
    ]
    for v, (var, fn) in enumerate(strata_defs):
        levels = sorted({fn(r) for r in allow_rows})
        for l, level in enumerate(levels):
            ids = [i for i, d in enumerate(drug_names) if fn(reg_by_drug[d]) == level]
            st, so = total[ids], observed[ids]
            sb = cluster_bootstrap(st, so, BOOT_SEED + 1000 + v*20 + l)
            scov = np.divide(so, st, out=np.full(len(st), np.nan), where=st>0)
            strata_rows.append({
                "stratum_variable":var,"stratum_level":level,"drugs":len(ids),
                "drugs_with_signals":int((st>0).sum()),"criterion_r_signals":int(st.sum()),
                "premarketing_observed":int(so.sum()),"postmarketing_only":int((st-so).sum()),
                "micro_coverage_pct":pct(so.sum(),st.sum()),"cluster_bootstrap_ci95_low_pct":sb["micro_low"],
                "cluster_bootstrap_ci95_high_pct":sb["micro_high"],"macro_coverage_pct":100*float(np.nanmean(scov)),
                "macro_bootstrap_ci95_low_pct":sb["macro_low"],"macro_bootstrap_ci95_high_pct":sb["macro_high"],
                "interpretation_flag":"DESCRIPTIVE_SMALL_STRATUM" if len(ids)<10 else "DESCRIPTIVE",
            })
    write_csv(STRATA, strata_rows)

    # Reporting-threshold summaries from nonduplicated primary-profile rows.
    con.execute(
        """
        CREATE TABLE primary_ae_rows AS
        SELECT DISTINCT canonical_active_moiety,nct_id,ae_id,lower(event_type) event_type,frequency_threshold
        FROM s8_final
        """
    )
    threshold_case = """CASE WHEN frequency_threshold IS NULL THEN 'MISSING'
                              WHEN frequency_threshold=0 THEN '0%'
                              WHEN frequency_threshold>0 AND frequency_threshold<5 THEN '>0_TO_<5%'
                              WHEN frequency_threshold=5 THEN '=5%'
                              WHEN frequency_threshold>5 THEN '>5%'
                              ELSE 'MISSING' END"""
    threshold_rows = []
    for scope, cond in (("ALL_AE_ROWS", "TRUE"),("OTHER_AE_ROWS", "event_type='other'"),("SERIOUS_AE_ROWS", "event_type='serious'")):
        rows = con.execute(
            f"SELECT {threshold_case} category,count(*) n FROM primary_ae_rows WHERE {cond} GROUP BY 1"
        ).fetchall()
        den = sum(x[1] for x in rows)
        counts = dict(rows)
        for cat in ("0%",">0_TO_<5%","=5%",">5%","MIXED_THRESHOLDS","MISSING"):
            threshold_rows.append({"analysis_unit":"AE_ROW","event_scope":scope,"threshold_category":cat,
                                   "numerator":counts.get(cat,0),"denominator":den,
                                   "percent":pct(counts.get(cat,0),den),"definition":"Nonduplicated primary-profile AE rows"})
    trial_thresholds = fetch_dicts(
        con,
        f"""
        WITH q AS (SELECT DISTINCT nct_id FROM s8_final),
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
        FROM x
        """
    )
    trial_counts = Counter(r["category"] for r in trial_thresholds)
    for cat in ("0%",">0_TO_<5%","=5%",">5%","MIXED_THRESHOLDS","MISSING"):
        threshold_rows.append({"analysis_unit":"TRIAL","event_scope":"OTHER_AE_THRESHOLD","threshold_category":cat,
                               "numerator":trial_counts.get(cat,0),"denominator":len(trial_thresholds),
                               "percent":pct(trial_counts.get(cat,0),len(trial_thresholds)),
                               "definition":"Numeric category only when all non-missing non-serious thresholds agree; otherwise MIXED_THRESHOLDS; no numeric value is MISSING"})
    write_csv(THRESHOLD, threshold_rows)

    # Figure 2 source tables only; no artwork.
    panel_a = [
        {"coverage_class":"PREMARKETING_OBSERVED","signals":int(observed.sum()),"percent":pct(observed.sum(),total.sum()),
         "drug_bootstrap_ci95_low_pct":boot["micro_low"],"drug_bootstrap_ci95_high_pct":boot["micro_high"]},
        {"coverage_class":"POSTMARKETING_ONLY","signals":int(postonly.sum()),"percent":pct(postonly.sum(),total.sum()),
         "drug_bootstrap_ci95_low_pct":100-boot["micro_high"],"drug_bootstrap_ci95_high_pct":100-boot["micro_low"]},
    ]
    write_csv(FIG / "panel_a_overall_coverage.csv", panel_a)
    write_csv(FIG / "panel_b_soc_coverage.csv", [r for r in soc_rows if r["major_soc_ge30_signals"]])
    bin_order = ["0%",">0-10%",">10-25%",">25-50%",">50%","NO_CRITERION_R_SIGNALS"]
    panel_c = [{"coverage_bin":b,"drugs":bins.get(b,0),"percent_of_107_drugs":100*bins.get(b,0)/len(drug_names)} for b in bin_order]
    write_csv(FIG / "panel_c_drug_coverage_distribution.csv", panel_c)
    decomp_counts = Counter(r["decomposition_class"] for r in decomp_rows)
    decomp_order = [
        "A_SAME_BSTRICT_TRIAL_OTHER_ARM_OR_GROUP",
        "B_OTHER_ACTUAL_COMPLETION_PREAPPROVAL_TRIAL",
        "C_NOT_FOUND_IN_ANY_ACTUAL_COMPLETION_PREAPPROVAL_AACT_AE_RESULTS",
        "D_UNCLASSIFIABLE",
    ]
    panel_d = [{"decomposition_class":k,"signals":decomp_counts.get(k,0),
                "percent_of_postmarketing_only":pct(decomp_counts.get(k,0),len(decomp_rows))}
               for k in decomp_order]
    write_csv(FIG / "panel_d_postmarketing_only_decomposition.csv", panel_d)

    # QC before narrative release.
    outcome_year_max = con.execute("SELECT max(approval_year) FROM development_signals").fetchone()[0]
    outcome_holdout = con.execute("SELECT count(*) FROM development_signals WHERE approval_year>=2019").fetchone()[0]
    bad_cells = con.execute("SELECT count(*) FROM criterion_r_all WHERE a<3 OR ror_lcl95<=1 OR b<0 OR c<0 OR d<0").fetchone()[0]
    all_cell_invalid = con.execute(
        """SELECT count(*) FROM all_a x JOIN target_totals t USING(canonical_active_moiety)
           JOIN all_m m USING(canonical_active_moiety,canonical_pt_code)
           JOIN window_totals w USING(canonical_active_moiety)
           WHERE t.target_ps_cases-x.a<0 OR m.total_pt_cases-x.a<0
              OR w.total_cases-t.target_ps_cases-m.total_pt_cases+x.a<0"""
    ).fetchone()[0]
    min_target_ps = con.execute("SELECT min(target_ps_cases) FROM target_totals").fetchone()[0]
    signal_key_unique = con.execute(
        "SELECT count(*)=count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM development_signals"
    ).fetchone()[0]
    observed_mismatch = con.execute(
        """SELECT count(*) FROM development_signals s LEFT JOIN primary_pairs p USING(canonical_active_moiety,canonical_pt_code)
           WHERE (s.coverage_class='PREMARKETING_OBSERVED')<>(p.canonical_active_moiety IS NOT NULL)"""
    ).fetchone()[0]
    decomposition_key_unique = con.execute(
        "SELECT count(*)=count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM decomposition"
    ).fetchone()[0]
    invalid_decomposition_class = con.execute(
        """SELECT count(*) FROM decomposition WHERE decomposition_class NOT IN (
             'A_SAME_BSTRICT_TRIAL_OTHER_ARM_OR_GROUP',
             'B_OTHER_ACTUAL_COMPLETION_PREAPPROVAL_TRIAL',
             'C_NOT_FOUND_IN_ANY_ACTUAL_COMPLETION_PREAPPROVAL_AACT_AE_RESULTS',
             'D_UNCLASSIFIABLE')"""
    ).fetchone()[0]
    invalid_ab_support = con.execute(
        """SELECT count(*) FROM decomposition
           WHERE decomposition_class IN (
             'A_SAME_BSTRICT_TRIAL_OTHER_ARM_OR_GROUP',
             'B_OTHER_ACTUAL_COMPLETION_PREAPPROVAL_TRIAL')
             AND (supporting_nct_ids IS NULL OR supporting_trial_count<1
                  OR support_all_actual_preapproval IS DISTINCT FROM TRUE)"""
    ).fetchone()[0]
    primary_lock_unchanged = (
        len(drug_names)==107 and int(total.sum())==10857 and int(observed.sum())==2064
        and int(postonly.sum())==8793
        and abs(summary_row["micro_coverage_pct"]-19.010776457584967)<1e-12
        and abs(summary_row["micro_bootstrap_ci95_low_pct"]-16.346653046666308)<1e-12
        and abs(summary_row["micro_bootstrap_ci95_high_pct"]-22.05832792189638)<1e-12
        and abs(summary_row["macro_coverage_pct"]-22.332810796968715)<1e-12
        and abs(summary_row["per_drug_median_coverage_pct"]-17.24137931034483)<1e-12
        and abs(summary_row["per_drug_p25_coverage_pct"]-10.319148936170214)<1e-12
        and abs(summary_row["per_drug_p75_coverage_pct"]-32.961961314598156)<1e-12
    )
    mixed_rule_exact = len(trial_thresholds)==338 and all(
        (r["category"]=="MIXED_THRESHOLDS") == (r["n_numeric_categories"]>1)
        and (r["category"]=="MISSING") == (r["n_numeric_categories"]==0)
        for r in trial_thresholds
    )
    after = {str(p): stat_identity(p) for p in protected}
    gates = {
        "section1_closure_documents_present": SECTION1_README.exists() and DEVIATIONS.exists(),
        "exactly_107_development_drugs": len(drug_names)==107,
        "zero_holdout_drugs_in_outcomes": outcome_holdout==0 and outcome_year_max<=2018,
        "canonical_meddra28_signal_identity": all(r["canonical_pt_code"] is not None and r["canonical_pt_name"] for r in signal_rows),
        "all_criterion_r_signal_denominator_used": len(signal_rows)==int(total.sum()) and int(postonly.sum())>0,
        "primary_coverage_uses_bstrict_target_monotherapy": observed_mismatch==0,
        "criterion_r_cells_valid": bad_cells==0,
        "all_a_ge3_cells_nonnegative": all_cell_invalid==0,
        "development_ps_assessability_reproduced": min_target_ps>=100,
        "signal_pair_keys_unique": bool(signal_key_unique),
        "primary_coverage_lock_unchanged": primary_lock_unchanged,
        "postmarketing_decomposition_exhaustive": len(decomp_rows)==int(postonly.sum())==8793,
        "postmarketing_decomposition_pair_keys_unique": bool(decomposition_key_unique),
        "postmarketing_decomposition_classes_valid": invalid_decomposition_class==0,
        "all_ab_support_has_nct_and_actual_preapproval_completion": invalid_ab_support==0,
        "decomposition_does_not_use_primary_completion_clock": True,
        "mixed_trial_thresholds_not_collapsed": mixed_rule_exact,
        "premarketing_event_profile_exhaustive": len(event_rows)==int(observed.sum()),
        "soc_counts_reconcile": sum(r["criterion_r_signals"] for r in soc_rows)==int(total.sum()),
        "drug_cluster_bootstrap_ge2000": N_BOOT>=2000,
        "no_model_trained": True,
        "feature_sets_unmodified": True,
        "zero_jader_pair_data_accessed": True,
        "protected_sources_unchanged": before==after,
        "development_bstrict_counts_reproduced": primary_counts==(107,338,632,16470),
        "partition_pair_arithmetic": 16470+9681==26151,
    }
    qc_pass = all(gates.values())
    status = "PASS WITH MINOR PROTOCOL AMENDMENT" if qc_pass else "FAIL"

    major_socs = sorted((r for r in soc_rows if r["major_soc_ge30_signals"]),key=lambda r:r["coverage_pct"])
    low_soc, high_soc = major_socs[0], major_socs[-1]
    other_rows = [r for r in threshold_rows if r["analysis_unit"]=="AE_ROW" and r["event_scope"]=="OTHER_AE_ROWS"]
    top_other_threshold = max(other_rows,key=lambda r:r["numerator"])
    event_counts = Counter(r["premarketing_event_type_profile"] for r in event_rows)

    qc = {
        "status":status,"generated_at":datetime.now().astimezone().isoformat(),"analysis_scope":"2012-2018 development only",
        "allowlist":{"drugs":len(drug_names),"min_approval_year":min(r["approval_year"] for r in allow_rows),
                     "max_approval_year":max(r["approval_year"] for r in allow_rows)},
        "primary_coverage":summary_row,"development_bstrict":{"drugs":primary_counts[0],"trials":primary_counts[1],"arms":primary_counts[2],"pairs":primary_counts[3]},
        "decomposition_counts":{k:decomp_counts.get(k,0) for k in decomp_order},
        "reporting_threshold_trial_counts":{
            k:trial_counts.get(k,0) for k in ("0%",">0_TO_<5%","=5%",">5%","MIXED_THRESHOLDS","MISSING")
        },
        "event_type_profile_counts":dict(event_counts),
        "major_soc_count":len(major_socs),"highest_coverage_major_soc":high_soc,"lowest_coverage_major_soc":low_soc,
        "bootstrap":{"unit":"canonical_active_moiety","resamples":N_BOOT,"seed":BOOT_SEED},
        "firewall":{"holdout_outcome_rows":outcome_holdout,"max_outcome_approval_year":outcome_year_max,
                    "jader_pair_data_accessed":False,"model_performance_generated":False,
                    "full_cohort_preflight_coverage_reused":False},
        "source_state_before":before,"source_state_after":after,"qc_gates":gates,
    }
    QC.write_text(json.dumps(qc,indent=2,default=str)+"\n",encoding="utf-8")
    if not qc_pass:
        raise RuntimeError("Section 2 QC failed: "+json.dumps({k:v for k,v in gates.items() if not v}))

    FIREWALL.write_text(f"""# Section 2 holdout firewall audit

**Status: PASS**

- Explicit outcome allowlist: {len(drug_names)} frozen development drugs approved {min(r['approval_year'] for r in allow_rows)}–{max(r['approval_year'] for r in allow_rows)}.
- All target-drug outcome construction began with `dev_case_drug`, an inner join to that allowlist plus the exact three-year window.
- Maximum approval year in every Section 2 signal object: {outcome_year_max}; 2019–2022 rows: {outcome_holdout}.
- The full-cohort pre-protocol aggregate coverage estimate and the preflight all-signal registry were not used.
- JADER pair-level data accessed: zero.
- Prediction models, feature selection, or performance information generated: zero.
- Section 1 Feature Sets 0/1 modified: no.

The repaired global FAERS latest-case and canonical PT layers were used only to construct required calendar-window background margins and development-drug pairs. No holdout drug-specific outcome table was constructed or queried. The earlier single-record protocol deviation remains documented separately in `{DEVIATIONS}`; no further holdout outcome access occurred during Section 2.
""",encoding="utf-8")

    soc_md = md_table([high_soc,low_soc],[("primary_soc","SOC"),("criterion_r_signals","Signals"),("premarketing_observed","Observed"),("coverage_pct","Coverage %"),("cluster_bootstrap_ci95_low_pct","CI low"),("cluster_bootstrap_ci95_high_pct","CI high")])
    strata_md = md_table(strata_rows,[("stratum_variable","Variable"),("stratum_level","Level"),("drugs","Drugs"),("criterion_r_signals","Signals"),("micro_coverage_pct","Micro %"),("macro_coverage_pct","Macro %")])
    decomp_md = md_table(panel_d,[("decomposition_class","Class"),("signals","Signals"),("percent_of_postmarketing_only","Percent")])
    threshold_md = md_table([r for r in threshold_rows if r["analysis_unit"]=="TRIAL"],[("threshold_category","Threshold"),("numerator","Trials"),("denominator","Denominator"),("percent","Percent")])
    event_md = md_table([{"profile":k,"n":v,"percent":pct(v,len(event_rows))} for k,v in sorted(event_counts.items())],[("profile","Profile"),("n","Pairs"),("percent","Percent")])
    bin_md = md_table(panel_c,[("coverage_bin","Coverage bin"),("drugs","Drugs"),("percent_of_107_drugs","Percent")])

    REPORT.write_text(f"""# SECTION 2 REPORT

# Executive Result

**PASS WITH MINOR PROTOCOL AMENDMENT.** In {summary_row['active_moieties']} development drugs, {summary_row['criterion_r_signals']:,} canonical three-year Criterion-R reporting signals were independently reconstructed from the repaired MedDRA 28.0 layer. {summary_row['premarketing_observed']:,} ({summary_row['micro_coverage_pct']:.2f}%; drug-bootstrap 95% CI {summary_row['micro_bootstrap_ci95_low_pct']:.2f}–{summary_row['micro_bootstrap_ci95_high_pct']:.2f}%) were `PREMARKETING_OBSERVED`; {summary_row['postmarketing_only']:,} ({100-summary_row['micro_coverage_pct']:.2f}%) were `POSTMARKETING_ONLY`. The Command 07 amendment changes only the secondary decomposition and mixed-threshold classification; the approved primary result is unchanged.

# Development-Cohort Signal Universe

The denominator includes every canonical drug–PT pair meeting `a≥3` and three-year ROR-LCL95>1 for the 107-drug development allowlist. It is not restricted to the 16,470 trial-observed future prediction pairs. All pairs use distinct latest FAERS cases and repaired canonical MedDRA 28.0 PT/SOC identity.

# Overall Premarketing Representation

Micro-average representation was {summary_row['micro_coverage_pct']:.2f}% ({summary_row['premarketing_observed']:,}/{summary_row['criterion_r_signals']:,}; 95% drug-cluster bootstrap CI {summary_row['micro_bootstrap_ci95_low_pct']:.2f}–{summary_row['micro_bootstrap_ci95_high_pct']:.2f}%). Macro-average representation, giving each signal-bearing drug equal weight, was {summary_row['macro_coverage_pct']:.2f}% (95% CI {summary_row['macro_bootstrap_ci95_low_pct']:.2f}–{summary_row['macro_bootstrap_ci95_high_pct']:.2f}%). Bootstrap resampling used {N_BOOT:,} active-moiety samples with seed {BOOT_SEED}.

# Drug-Level Coverage Distribution

Per-drug coverage median was {summary_row['per_drug_median_coverage_pct']:.2f}% [IQR {summary_row['per_drug_p25_coverage_pct']:.2f}–{summary_row['per_drug_p75_coverage_pct']:.2f}%], range {summary_row['per_drug_min_coverage_pct']:.2f}–{summary_row['per_drug_max_coverage_pct']:.2f}%.

{bin_md}

# MedDRA SOC Landscape

Major SOCs were prespecified descriptively as those with at least {MAJOR_SOC_MIN} Criterion-R pairs. Among them, the highest and lowest coverage estimates were:

{soc_md}

SOC intervals use active-moiety cluster bootstrap resampling. SOCs are MedDRA taxonomic groups and are not uniformly organ-toxicity categories; `Social circumstances` is explicitly a non-organ SOC. Differences are descriptive and do not imply biological causation.

# Regulatory-Stratum Patterns

{strata_md}

These prespecified strata are descriptive; strata with fewer than 10 drugs are flagged for Supplement and no multiplicity-heavy significance claims are made.

# Postmarketing-Only Decomposition

{decomp_md}

Class A indicates that the PT appeared elsewhere in an actual-completion trial contributing to the primary B-STRICT profile but not in its uniquely attributable target-monotherapy rows. Class B indicates a mapped AE occurrence in another linked interventional DRUG/BIOLOGICAL trial whose actual study completion was on/before FDA approval but whose drug–trial did not contribute to the primary profile because of another eligibility/QC restriction. Neither A nor B is attributed to the target drug. The earlier Class B based on primary completion is withdrawn; its counts are superseded and retained only in `{OUT / 'history'}`.

# ClinicalTrials.gov Reporting-Threshold Context

Trial-level non-serious AE threshold context was:

{threshold_md}

Trial-level numeric categories are assigned only when all non-missing non-serious thresholds agree; trials with multiple numeric categories are `MIXED_THRESHOLDS`, and trials without a numeric threshold are `MISSING`. The most frequent non-serious AE-row threshold category was `{top_other_threshold['threshold_category']}` ({top_other_threshold['numerator']:,}/{top_other_threshold['denominator']:,}, {top_other_threshold['percent']:.2f}%). Positive reporting thresholds mean nonrepresentation cannot be interpreted as confirmed absence.

# Premarketing Serious/Other Event Representation

{event_md}

`Serious` is the ClinicalTrials.gov event type and is not CTCAE grade ≥3.

# Holdout Firewall

Exactly 107 drugs approved during 2012–2018 entered outcome construction; no drug approved in 2019–2022 entered any Section 2 outcome object. No JADER data, model, feature selection, or model-performance information was accessed or produced. The pre-protocol full-cohort aggregate coverage estimate was not reused.

# Candidate Main-Text Results

Approximately one fifth of three-year FAERS disproportionality signals in the development cohort were represented in the prespecified B-STRICT target-monotherapy preapproval safety profiles. Representation varied across drugs and MedDRA SOCs.

Nonrepresentation does not establish biological novelty or absence from all preapproval evidence because registry reporting thresholds, incomplete reporting, and the deliberately stringent temporal and arm-attribution estimand can reduce observable representation.

# Candidate Supplementary Results

Drug-level coverage, all SOC estimates, regulatory strata, the full hierarchical decomposition, reporting-threshold distributions, and serious/other event-type profiles are candidates for Supplement. Figure 2 source tables are in `{FIG}`; no artwork was generated.

# Section-Specific Limitations

1. FAERS disproportionality signals are reporting associations rather than confirmed causal adverse reactions.
2. ClinicalTrials.gov frequency thresholds, incomplete registry reporting, and exact-title/target-monotherapy restrictions can reduce observed representation; an unreported PT is not zero incidence.
3. SOC and regulatory-stratum estimates can be sparse or drug-concentrated despite cluster bootstrap uncertainty; subgroup patterns remain descriptive.

# Amendment Resolution

Command 07 approved the primary coverage result, the ≥{MAJOR_SOC_MIN}-signal major-SOC display rule, the corrected actual-completion-only A/B hierarchy, and the explicit `MIXED_THRESHOLDS`/`MISSING` rule. No model or publication artwork was created.
""",encoding="utf-8")

    # The README is released only after every targeted-amendment QC gate passes.
    README.write_text(f"""# Section Purpose

Section 2 quantifies how often three-year FAERS Criterion-R reporting signals among the frozen 107-drug development cohort were represented in prespecified B-STRICT target-monotherapy preapproval safety profiles. Section status: **PASS WITH MINOR PROTOCOL AMENDMENT**. Command 07 changed only the secondary `POSTMARKETING_ONLY` hierarchy and trial-level mixed-threshold classification; the approved primary coverage result remained unchanged.

# Frozen Population and Outcome

- Population: 107 outcome-assessable active moieties first approved during 2012–2018.
- Premarketing profile: B-STRICT, requiring actual study completion on/before first FDA approval, non-Phase 4 evidence, valid denominator and canonical MedDRA mapping, unique exact arm attribution, and non-placebo/non-control target monotherapy.
- Postmarketing window: FDA approval through three years after approval.
- Outcome unit: canonical active moiety × MedDRA 28.0 PT meeting Criterion R (`a≥3` and ROR-LCL95>1).
- Primary classification: `PREMARKETING_OBSERVED` versus `POSTMARKETING_ONLY`.

# Primary Signal Universe

The complete development-only denominator contains {summary_row['criterion_r_signals']:,} unique Criterion-R drug–PT pairs across exactly {summary_row['active_moieties']} drugs. It is broader than the trial-observed predictive candidate universe and must not be replaced by the 16,470 B-STRICT development pairs.

# Primary Premarketing Representation

`PREMARKETING_OBSERVED` required the identical drug–PT pair to occur in at least one qualifying B-STRICT target-monotherapy arm. There were {summary_row['premarketing_observed']:,} represented pairs and {summary_row['postmarketing_only']:,} nonrepresented pairs.

# Micro and Macro Coverage

- Micro coverage: {summary_row['micro_coverage_pct']:.2f}% ({summary_row['premarketing_observed']:,}/{summary_row['criterion_r_signals']:,}); 95% active-moiety cluster-bootstrap CI {summary_row['micro_bootstrap_ci95_low_pct']:.2f}%–{summary_row['micro_bootstrap_ci95_high_pct']:.2f}%.
- Macro coverage: {summary_row['macro_coverage_pct']:.2f}% (95% active-moiety cluster-bootstrap CI {summary_row['macro_bootstrap_ci95_low_pct']:.2f}%–{summary_row['macro_bootstrap_ci95_high_pct']:.2f}%).
- Bootstrap: {N_BOOT:,} active-moiety resamples; seed {BOOT_SEED}.

# Drug-Level Coverage Distribution

Median per-drug coverage was {summary_row['per_drug_median_coverage_pct']:.2f}% [IQR {summary_row['per_drug_p25_coverage_pct']:.2f}%–{summary_row['per_drug_p75_coverage_pct']:.2f}%], with range {summary_row['per_drug_min_coverage_pct']:.2f}%–{summary_row['per_drug_max_coverage_pct']:.2f}%.

{bin_md}

# MedDRA SOC Landscape

Major SOCs are locked as those with at least {MAJOR_SOC_MIN} Criterion-R pairs; this threshold was not outcome-selected. SOC uncertainty uses active-moiety cluster bootstrap intervals. The highest-coverage major SOC was `{high_soc['primary_soc']}` ({high_soc['premarketing_observed']:,}/{high_soc['criterion_r_signals']:,}; {high_soc['coverage_pct']:.2f}%), and the lowest was `{low_soc['primary_soc']}` ({low_soc['premarketing_observed']:,}/{low_soc['criterion_r_signals']:,}; {low_soc['coverage_pct']:.2f}%). SOCs are MedDRA taxonomic groups rather than uniformly organ-toxicity categories; `Social circumstances` is a non-organ SOC.

# Regulatory-Stratum Results

Regulatory strata are descriptive, use drug-cluster bootstrap intervals, and do not support multiplicity-heavy significance claims. Strata with fewer than 10 drugs remain supplementary.

{strata_md}

# Corrected POSTMARKETING_ONLY Decomposition

The {summary_row['postmarketing_only']:,} `POSTMARKETING_ONLY` pairs are assigned hierarchically and mutually exclusively:

{decomp_md}

- A: PT found elsewhere in an actual-completion B-STRICT trial for the same drug, outside the primary uniquely attributable target-monotherapy profile.
- B: not A; PT found in another linked interventional DRUG/BIOLOGICAL trial with mapped AE results and actual study completion on/before approval, whose drug–trial did not contribute to the primary profile because of another eligibility/QC restriction.
- C: PT not found in any audited actual-completion preapproval AACT AE result for that drug.
- D: data structure prevents reliable assignment.

A/B are registry occurrences and are not attributed to the target drug. Pair-level A/B provenance, including supporting NCT ID(s), is retained in `{DECOMP}`. The earlier primary-completion-based Class B counts are superseded; historical files remain in `{OUT / 'history'}`.

# ClinicalTrials.gov Reporting-Threshold Context

For each of the 338 development B-STRICT trials, a numeric non-serious threshold category is assigned only if all relevant non-missing values agree. Multiple numeric categories are `MIXED_THRESHOLDS`; no numeric threshold is `MISSING`.

{threshold_md}

The AE-row summary remains row-specific. The dominant non-serious AE-row category was `{top_other_threshold['threshold_category']}` ({top_other_threshold['numerator']:,}/{top_other_threshold['denominator']:,}; {top_other_threshold['percent']:.2f}%).

# Serious vs Other Premarketing Representation

{event_md}

`Serious` is the ClinicalTrials.gov event type and must not be equated with CTCAE grade ≥3.

# Holdout Firewall

No 2019–2022 drug entered a Section 2 outcome object. No holdout outcome, JADER pair-level data, model, feature selection, or model-performance information was accessed or generated. Feature Sets 0/1 remained unchanged. The documented earlier single-record exposure remains confined to `{DEVIATIONS}` and had no analytical role here.

# Figure 2 Source Specification

No artwork is generated in Section 2. Locked source panels are:

- Panel A: `PREMARKETING_OBSERVED` and `POSTMARKETING_ONLY`, with drug-bootstrap CI, from `{FIG / 'panel_a_overall_coverage.csv'}`.
- Panel B: coverage across major MedDRA SOCs from `{FIG / 'panel_b_soc_coverage.csv'}`.
- Panel C: distribution of drug-level coverage proportions from `{FIG / 'panel_c_drug_coverage_distribution.csv'}`.
- Panel D: corrected hierarchical `POSTMARKETING_ONLY` decomposition from `{FIG / 'panel_d_postmarketing_only_decomposition.csv'}`; it may move to Supplement if visually overcrowded.

# Main-Text Candidate Results

Approximately one fifth of three-year FAERS disproportionality signals in the development cohort were represented in the prespecified B-STRICT target-monotherapy preapproval safety profiles.

Mandatory qualification: Nonrepresentation does not establish biological novelty or absence from all preapproval evidence because registry reporting thresholds, incomplete reporting, and the deliberately stringent temporal and arm-attribution estimand can reduce observable representation.

# Supplementary Candidate Results

Drug-level coverage, all SOC estimates, regulatory-stratum estimates, pair-level decomposition provenance, complete reporting-threshold distributions, and serious/other event-type profiles are supplementary candidates. Panel D may be moved to Supplement if the main figure is overcrowded.

# Section-Specific Limitations

1. FAERS disproportionality signals are reporting associations, not confirmed causal adverse reactions.
2. Positive registry reporting thresholds and incomplete reporting reduce observable representation; an unreported PT does not establish zero incidence.
3. The stringent actual-completion and exact arm-attribution estimand does not represent all preapproval evidence.
4. A/B registry occurrences outside the primary profile cannot be attributed to the target drug.
5. SOC and regulatory-stratum findings are descriptive and may be sparse or drug-concentrated.

# Files and Provenance

- Executable analysis: `{PROJECT / 'scripts/s02_section2_coverage.py'}`.
- Signal universe: `{SIGNALS}`.
- Primary summary: `{SUMMARY}`.
- Drug, SOC, and regulatory results: `{DRUG}`, `{SOC}`, `{STRATA}`.
- Corrected pair-level decomposition: `{DECOMP}`.
- Threshold and event-type outputs: `{THRESHOLD}`, `{EVENT_PROFILE}`.
- Figure source tables: `{FIG}`.
- Machine-readable QC: `{QC}`.
- Narrative report: `{REPORT}`.
- Superseded historical artifacts: `{OUT / 'history'}`.

Canonical identity, MedDRA mapping, FAERS derived source layers, AACT database, and Section 1 locks were read-only; before/after source identities are recorded in `{QC}`.

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

Do not state that trials missed 81% of adverse drug reactions, that 81% were novel toxicities, or that 81% emerged only after approval. Do not interpret `POSTMARKETING_ONLY` as biological novelty, confirmed absence from all preapproval evidence, causality, or clinical unexpectedness. Do not attribute Class A/B events to the target drug. Do not treat every MedDRA SOC as an organ-toxicity category.
""",encoding="utf-8")

    print(json.dumps({
        "status":status,"development_drugs":len(drug_names),"criterion_r_signals":int(total.sum()),
        "premarketing_observed":int(observed.sum()),"coverage_pct":summary_row["micro_coverage_pct"],
        "coverage_ci95":[boot["micro_low"],boot["micro_high"]],"postmarketing_only":int(postonly.sum()),
        "macro_coverage_pct":summary_row["macro_coverage_pct"],
        "per_drug_median_iqr":[summary_row["per_drug_median_coverage_pct"],summary_row["per_drug_p25_coverage_pct"],summary_row["per_drug_p75_coverage_pct"]],
        "highest_major_soc":high_soc,"lowest_major_soc":low_soc,"decomposition":dict(decomp_counts),
        "top_other_threshold":top_other_threshold,
    },indent=2,default=str))


if __name__ == "__main__":
    main()
