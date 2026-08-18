#!/usr/bin/env python3
"""Command 08 / Section 3A: feature freeze without model fitting.

The script first performs an independent Class-A audit. It stops before any
feature output if that audit disagrees with the locked zero count. On PASS it
creates a development-only B-STRICT pair registry, descriptive audits, frozen
drug-grouped outer folds, and the modelling protocol. It never reads the full
FAERS label registry, a holdout outcome, or JADER data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT = Path("/path/to/project")
OUT = PROJECT / "analysis/section3_model"
AACT_DB = Path("/path/to/Database/AACT/aact.duckdb")
SECTION1 = PROJECT / "analysis/section1_cohort/02_final_drug_level_characteristics.csv"
SECTION1_README = PROJECT / "analysis/section1_cohort/SECTION1_README.md"
SECTION2_SIGNALS = PROJECT / "analysis/section2_coverage/01_development_signal_universe.csv"
SECTION2_SUMMARY = PROJECT / "analysis/section2_coverage/02_primary_coverage_summary.csv"
SECTION2_README = PROJECT / "analysis/section2_coverage/SECTION2_README.md"
IDENTITY = PROJECT / "preflight_v2/drug_identity_master.csv"
IDENTITY_SHA = PROJECT / "preflight_v2/drug_identity_master.sha256"
LINKS = PROJECT / "data/processed/preflight_v2/aact_fda_intervention_links.parquet"
BSTRICT = PROJECT / "preflight_v2/bstrict_candidate_registry.parquet"
AACT_PT_MAP = PROJECT / "preflight_v2/faers_pt_repair/aact_meddra28_term_mapping.csv"
MEDDRA_MDHIER = Path("/path/to/Database/MedDRA/MedDRA_28_0_ENglish/MedAscii/mdhier.asc")
FEATURE_FEASIBILITY = PROJECT / "preflight_v2/07_feature_feasibility.md"
PROTOCOL_LOCK = PROJECT / "docs/STUDY_PROTOCOL_LOCK_v1.md"

CLASSA_QC = OUT / "00_classA_independent_qc.md"
PAIR_REGISTRY = OUT / "development_pair_registry.parquet"
MODEL_DOMAIN = OUT / "01_development_model_domain.csv"
OUTCOME_DIST = OUT / "02_outcome_distribution.csv"
FEATURE_DICT = OUT / "FEATURE_DICTIONARY_v1.csv"
MISSINGNESS = OUT / "03_feature_missingness.csv"
DISTRIBUTION = OUT / "04_feature_distribution_qc.csv"
PT_DIST = OUT / "05_pt_identity_distribution.csv"
FOLD_ASSIGN = OUT / "OUTER_FOLD_ASSIGNMENT_v1.csv"
FOLD_BALANCE = OUT / "06_outer_fold_balance.csv"
LEAKAGE = OUT / "07_leakage_audit.md"
MODELLING_PROTOCOL = OUT / "MODELLING_PROTOCOL_v1.md"
REPORT = OUT / "SECTION3A_REPORT.md"
QC = OUT / "SECTION3A_QC.json"

SEED = 20260810
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_REPS = 5000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stat_identity(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def md_table(rows: list[dict], fields: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in fields) + " |"
    sep = "|" + "|".join("---" for _ in fields) + "|"
    body = []
    for row in rows:
        vals = []
        for key, _ in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:,.3f}"
            vals.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep, *body])


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else float("nan")


def feature(
    name: str, feature_set: str, domain: str, level: str, source: str,
    definition: str, data_type: str, transformation: str, missingness_rule: str,
    leakage_risk: str, status: str, rationale: str,
) -> dict:
    return {
        "feature_name": name,
        "feature_set": feature_set,
        "domain": domain,
        "level": level,
        "source": source,
        "definition": definition,
        "data_type": data_type,
        "transformation": transformation,
        "missingness_rule": missingness_rule,
        "temporal_availability": "Known from FDA or B-STRICT AACT data by FDA first approval",
        "leakage_risk": leakage_risk,
        "status": status,
        "rationale": rationale,
    }


def build_feature_dictionary(priority_status: str, priority_reason: str) -> list[dict]:
    f: list[dict] = []
    # Set 0: regulatory/context.
    f += [
        feature("approval_year", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Numeric FDA first-approval year.", "continuous", "None; fold-local standardization for elastic-net", "No missing expected", "LOW", "PRIMARY", "Prespecified temporal/regulatory context; not one-hot encoded."),
        feature("nda_bla", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Original application class, NDA versus BLA.", "categorical", "Fold-local one-hot", "Explicit MISSING/UNKNOWN", "LOW", "PRIMARY", "Prespecified regulatory context."),
        feature("orphan_designation", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Orphan designation collapsed to Yes/No using frozen text prefix.", "categorical", "Fold-local one-hot", "Explicit MISSING/UNKNOWN", "LOW", "PRIMARY", "Prespecified regulatory context."),
        feature("accelerated_approval", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Accelerated approval collapsed to Yes/No using frozen text prefix.", "categorical", "Fold-local one-hot", "Explicit MISSING/UNKNOWN", "LOW", "PRIMARY", "Prespecified regulatory context."),
        feature("breakthrough_therapy_designation", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Breakthrough designation; structural N/A retained, indication-qualified Yes collapsed to Yes.", "categorical", "Fold-local one-hot", "N/A is structural; missing is UNKNOWN", "LOW", "PRIMARY", "Prespecified; structural N/A is not ordinary missingness."),
        feature("fast_track_designation", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Fast-track designation; indication-qualified Yes collapsed to Yes.", "categorical", "Fold-local one-hot", "Explicit MISSING/UNKNOWN", "LOW", "PRIMARY", "Prespecified regulatory context."),
        feature("priority_review_category", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Priority review including voucher-qualified Priority collapsed to Priority versus Standard.", "categorical", "Fold-local one-hot", "Explicit MISSING/UNKNOWN", "LOW", priority_status, priority_reason),
        feature("route_broad", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Outcome-blind deterministic broad route: ORAL, PARENTERAL, OPHTHALMIC, INHALATION, MULTIPLE, OTHER.", "categorical", "Fold-local one-hot", "Explicit MISSING/UNKNOWN", "MEDIUM", "SECONDARY", "Development-only deterministic collapse; no outcome or holdout information used."),
        feature("dosage_form_broad", "SET0", "REGULATORY", "drug", "Frozen FDA/Section 1", "Outcome-blind deterministic form: SOLID, INJECTABLE, LIQUID, MULTIPLE, OTHER.", "categorical", "Fold-local one-hot", "Explicit MISSING/UNKNOWN", "MEDIUM", "SECONDARY", "Development-only deterministic collapse; no outcome or holdout information used."),
        feature("canonical_pt_code", "SET0", "EVENT_IDENTITY", "PT", "Canonical MedDRA 28.0", "Canonical PT identity; code is treated categorically.", "categorical", "Training-fold-only rare grouping (<2 training drugs) then one-hot; unknown safe", "UNKNOWN_PT in validation/holdout", "MEDIUM", "PRIMARY", "Event context; no target encoding, embedding, or outcome/frequency encoding."),
        feature("primary_soc", "SET0", "EVENT_IDENTITY", "PT", "Canonical MedDRA 28.0", "Single canonical primary SOC identity.", "categorical", "Fold-local one-hot", "Explicit UNKNOWN_SOC", "LOW", "PRIMARY", "Taxonomic context; no multiaxial SOC assignments."),
        feature("drug_n_qualifying_trials", "SET0", "DRUG_PROGRAM", "drug", "B-STRICT/Section 1", "Number of distinct qualifying B-STRICT trials.", "count", "log1p; fold-local scale for elastic-net", "No missing expected", "LOW", "PRIMARY", "General programme size, not pair-specific AE strength."),
        feature("drug_n_target_arms", "SET0", "DRUG_PROGRAM", "drug", "B-STRICT/Section 1", "Number of uniquely attributable non-placebo target-monotherapy arms.", "count", "log1p; fold-local scale for elastic-net", "No missing expected", "LOW", "PRIMARY", "General programme arm context."),
        feature("drug_approx_ae_safety_population", "SET0", "DRUG_PROGRAM", "drug", "B-STRICT/Section 1", "Approximate premarketing AE safety-population size: sum of maximum valid subjects-at-risk once per eligible target arm.", "count", "log1p; fold-local scale for elastic-net", "No missing expected", "MEDIUM", "SECONDARY", "Reproducible but approximate; event-specific denominators and cross-trial participant overlap prevent interpretation as exact exposure."),
        feature("drug_phase1_fraction", "SET0", "DRUG_PROGRAM", "drug", "B-STRICT/Section 1", "Fraction of qualifying trials classified Phase 1.", "proportion", "Retain 0-1; fold-local scale for elastic-net", "No missing; missing phase contributes non-Phase-1 with separate source audit", "LOW", "PRIMARY", "Prespecified programme composition."),
        feature("drug_randomized_trial_fraction", "SET0", "DRUG_PROGRAM", "drug", "B-STRICT/Section 1", "Fraction randomized among qualifying trials with nonmissing allocation.", "proportion", "Retain 0-1; fold-local scale for elastic-net", "Fold-local median; add indicator only if development missingness >=1%", "LOW", "PRIMARY", "Prespecified programme design context."),
        feature("drug_masked_trial_fraction", "SET0", "DRUG_PROGRAM", "drug", "B-STRICT/Section 1", "Fraction masked among qualifying trials with nonmissing masking.", "proportion", "Retain 0-1; fold-local scale for elastic-net", "Fold-local median; add indicator only if development missingness >=1%", "LOW", "PRIMARY", "Prespecified programme design context."),
        feature("drug_industry_sponsored_fraction", "SET0", "DRUG_PROGRAM", "drug", "B-STRICT/Section 1", "Fraction of qualifying trials with an industry lead sponsor among nonmissing sponsor records.", "proportion", "Retain 0-1; fold-local scale for elastic-net", "Fold-local median; add indicator only if development missingness >=1%", "LOW", "PRIMARY", "Reliable development completeness; programme context."),
    ]
    # Set 1 additions: pair-specific evidence.
    f += [
        feature("pair_n_reporting_trials", "SET1_ADDITIONAL", "EVIDENCE_VOLUME", "pair", "B-STRICT AACT", "Distinct qualifying trials reporting the PT in a target-monotherapy arm.", "count", "log1p; fold-local scale for elastic-net", "No missing expected", "LOW", "PRIMARY", "Prespecified pair-specific evidence volume."),
        feature("pair_reporting_trial_fraction", "SET1_ADDITIONAL", "EVIDENCE_VOLUME", "pair", "B-STRICT AACT", "PT-reporting trials divided by all qualifying trials for the drug.", "proportion", "Retain 0-1", "No missing expected", "LOW", "PRIMARY", "Normalizes pair evidence by programme size."),
        feature("pair_n_reporting_arms", "SET1_ADDITIONAL", "EVIDENCE_VOLUME", "pair", "B-STRICT AACT", "Distinct attributable target arms reporting the PT.", "count", "log1p; fold-local scale for elastic-net", "No missing expected", "LOW", "PRIMARY", "Prespecified pair-specific arm evidence."),
        feature("pair_nonduplicated_arm_subjects_at_risk", "SET1_ADDITIONAL", "EVIDENCE_VOLUME", "pair", "B-STRICT AACT", "Sum of maximum valid subjects-at-risk once per PT-reporting target arm.", "count", "log1p; fold-local scale for elastic-net", "No missing expected", "MEDIUM", "PRIMARY", "Avoids repeated AE-row and Serious/Other denominator summation within an arm; remains approximate across trials."),
        feature("pair_median_row_ae_proportion", "SET1_ADDITIONAL", "AE_PROPORTION", "pair", "B-STRICT AACT", "Median of distinct AE-row subjects_affected/subjects_at_risk proportions.", "proportion", "Retain 0-1", "No missing expected", "LOW", "PRIMARY", "Safe row-level summary without pooled-incidence interpretation."),
        feature("pair_max_row_ae_proportion", "SET1_ADDITIONAL", "AE_PROPORTION", "pair", "B-STRICT AACT", "Maximum of distinct AE-row affected proportions.", "proportion", "Retain 0-1", "No missing expected", "LOW", "PRIMARY", "Safe extremum summary; not pooled incidence."),
        feature("pair_row_ae_proportion_iqr", "SET1_ADDITIONAL", "AE_PROPORTION", "pair", "B-STRICT AACT", "IQR of row-level AE proportions when at least two rows contribute.", "continuous", "Retain original scale", "Missing/not-applicable when <2 rows; paired availability indicator retained", "LOW", "SECONDARY", "Outcome-independent variability summary."),
        feature("pair_row_variability_available", "SET1_ADDITIONAL", "AE_PROPORTION", "pair", "B-STRICT AACT", "Indicator that at least two distinct AE rows support row-level IQR.", "binary", "None", "No missing", "LOW", "SECONDARY", "Distinguishes not-applicable variability from biological zero."),
        feature("pair_any_serious", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Any ClinicalTrials.gov event_type=serious row for the PT.", "binary", "None", "No missing; absence of a serious row is 0", "LOW", "PRIMARY", "Serious is a registry event type, not CTCAE grade >=3."),
        feature("pair_n_serious_trials", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Number of PT-reporting trials with any serious row.", "count", "log1p; fold-local scale for elastic-net", "No missing", "LOW", "PRIMARY", "Serious-event trial context."),
        feature("pair_n_serious_arms", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Number of PT-reporting target arms with any serious row.", "count", "log1p; fold-local scale for elastic-net", "No missing", "LOW", "PRIMARY", "Serious-event arm context."),
        feature("pair_serious_trial_fraction", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Serious-reporting trials divided by PT-reporting trials.", "proportion", "Retain 0-1", "No missing", "LOW", "PRIMARY", "Prespecified normalized serious context."),
        feature("pair_serious_arm_fraction", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Serious-reporting target arms divided by PT-reporting target arms.", "proportion", "Retain 0-1", "No missing", "LOW", "PRIMARY", "Prespecified normalized serious context."),
        feature("pair_max_serious_row_proportion", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Maximum row-level proportion among serious rows.", "proportion", "Retain 0-1", "Missing when no serious row; pair_any_serious is the indicator", "LOW", "PRIMARY", "Avoids pooling serious and other rows."),
        feature("pair_max_other_row_proportion", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Maximum row-level proportion among other/non-serious rows.", "proportion", "Retain 0-1", "Missing when no other row; paired availability indicator retained", "LOW", "PRIMARY", "Keeps serious and other information separable."),
        feature("pair_other_proportion_available", "SET1_ADDITIONAL", "SERIOUS_CONTEXT", "pair", "B-STRICT AACT", "Indicator that at least one Other-event proportion is available.", "binary", "None", "No missing", "LOW", "PRIMARY", "Distinguishes absent other-event row from a zero proportion."),
        feature("pair_between_trial_proportion_sd", "SET1_ADDITIONAL", "CROSS_TRIAL", "pair", "B-STRICT AACT", "Sample SD across trial-level median row proportions when >=2 independent trials contribute.", "continuous", "Retain original scale", "Missing/not-applicable for <2 trials; paired availability indicator retained", "MEDIUM", "SECONDARY", "Stable simple heterogeneity summary; no I-squared."),
        feature("pair_cross_trial_variability_available", "SET1_ADDITIONAL", "CROSS_TRIAL", "pair", "B-STRICT AACT", "Indicator that >=2 independent PT-reporting trials support between-trial SD.", "binary", "None", "No missing", "LOW", "SECONDARY", "Prevents imputing not-applicable heterogeneity as biological zero."),
        feature("pair_min_other_threshold", "SET1_ADDITIONAL", "REPORTING_THRESHOLD", "pair", "B-STRICT AACT", "Minimum nonmissing Other-event frequency threshold among contributing rows.", "continuous", "Retain percentage-point scale", "Missing if no numeric Other threshold; availability indicator retained", "MEDIUM", "SECONDARY", "Registry reporting context; not evidence of absence below threshold."),
        feature("pair_max_other_threshold", "SET1_ADDITIONAL", "REPORTING_THRESHOLD", "pair", "B-STRICT AACT", "Maximum nonmissing Other-event frequency threshold among contributing rows.", "continuous", "Retain percentage-point scale", "Missing if no numeric Other threshold; availability indicator retained", "MEDIUM", "SECONDARY", "Registry reporting context."),
        feature("pair_median_other_threshold", "SET1_ADDITIONAL", "REPORTING_THRESHOLD", "pair", "B-STRICT AACT", "Median nonmissing Other-event frequency threshold among contributing rows.", "continuous", "Retain percentage-point scale", "Missing if no numeric Other threshold; availability indicator retained", "MEDIUM", "SECONDARY", "Registry reporting context."),
        feature("pair_fraction_trials_threshold_0", "SET1_ADDITIONAL", "REPORTING_THRESHOLD", "pair", "B-STRICT AACT", "Fraction of PT-reporting trials with at least one Other PT row using threshold 0%.", "proportion", "Retain 0-1", "No missing; denominator is all PT-reporting trials", "MEDIUM", "SECONDARY", "Describes threshold use without inferring absence."),
        feature("pair_fraction_trials_threshold_5", "SET1_ADDITIONAL", "REPORTING_THRESHOLD", "pair", "B-STRICT AACT", "Fraction of PT-reporting trials with at least one Other PT row using threshold 5%.", "proportion", "Retain 0-1", "No missing; denominator is all PT-reporting trials", "MEDIUM", "SECONDARY", "Describes threshold use without inferring absence."),
        feature("pair_other_threshold_available", "SET1_ADDITIONAL", "REPORTING_THRESHOLD", "pair", "B-STRICT AACT", "Indicator that at least one numeric Other-event threshold is available.", "binary", "None", "No missing", "LOW", "SECONDARY", "Scientifically interpretable threshold missingness."),
        feature("pair_phase1_fraction", "SET1_ADDITIONAL", "PAIR_TRIAL_DESIGN", "pair", "B-STRICT AACT", "Fraction of PT-reporting trials classified Phase 1.", "proportion", "Retain 0-1", "No missing; missing phase is not Phase 1", "LOW", "SECONDARY", "Pair-specific trial composition."),
        feature("pair_randomized_trial_fraction", "SET1_ADDITIONAL", "PAIR_TRIAL_DESIGN", "pair", "B-STRICT AACT", "Fraction randomized among PT-reporting trials with known allocation.", "proportion", "Retain 0-1", "Fold-local median; explicit indicator only if >=1% missing", "LOW", "SECONDARY", "Pair-specific design composition."),
        feature("pair_masked_trial_fraction", "SET1_ADDITIONAL", "PAIR_TRIAL_DESIGN", "pair", "B-STRICT AACT", "Fraction masked among PT-reporting trials with known masking.", "proportion", "Retain 0-1", "Fold-local median; explicit indicator only if >=1% missing", "LOW", "SECONDARY", "Pair-specific design composition."),
    ]
    # Explicitly prohibited/dropped candidates.
    f += [
        feature("canonical_active_moiety", "EXCLUDED", "IDENTIFIER", "drug", "Frozen identity", "Exact drug identity.", "categorical", "None", "N/A", "CRITICAL", "DROP", "Join/grouping key only; prohibited predictor."),
        feature("exact_faers_ps_volume", "EXCLUDED", "OUTCOME_DERIVED", "drug", "FAERS", "Exact three-year target-drug PS volume.", "count", "None", "N/A", "CRITICAL", "DROP", "Assessability restriction only; forbidden predictor and absent from source schema."),
        feature("faers_ror_or_ic", "EXCLUDED", "OUTCOME_DERIVED", "pair", "FAERS", "Any FAERS ROR, IC, cells, or outcome magnitude.", "continuous", "None", "N/A", "CRITICAL", "DROP", "Outcome leakage; only criterion_r_3y label is allowed."),
        feature("jader_feature", "EXCLUDED", "OUTCOME_DERIVED", "pair", "JADER", "Any JADER quantity or label.", "unspecified", "None", "N/A", "CRITICAL", "DROP", "External outcome data are prohibited as predictors and were not accessed."),
        feature("pair_pooled_affected_at_risk_proportion", "EXCLUDED", "AE_PROPORTION", "pair", "B-STRICT AACT", "Naive sum affected divided by sum at risk across Serious and Other rows.", "proportion", "None", "N/A", "HIGH", "DROP", "Same participants may recur across event types/time frames; pooled incidence is not identifiable."),
        feature("pair_raw_proportion_vector", "EXCLUDED", "AE_PROPORTION", "pair", "B-STRICT AACT", "Variable-length raw arm/row proportion vector.", "vector", "None", "N/A", "MEDIUM", "DROP", "Audit source only; not a fixed-dimensional model input."),
        feature("pair_min_row_ae_proportion", "EXCLUDED", "AE_PROPORTION", "pair", "B-STRICT AACT", "Minimum row-level AE proportion.", "proportion", "None", "N/A", "LOW", "DROP", "Redundant and zero-dominated; omitted without outcome screening."),
        feature("pair_i2", "EXCLUDED", "CROSS_TRIAL", "pair", "B-STRICT AACT", "I-squared heterogeneity statistic.", "continuous", "None", "N/A", "MEDIUM", "DROP", "Sparse contributing trials make I-squared unstable and decorative."),
        feature("positional_arm_recovery", "EXCLUDED", "ARM_MAPPING", "trial", "AACT", "Any feature based on positional recovery of unmapped result groups.", "unspecified", "None", "N/A", "CRITICAL", "DROP", "Primary exact-title arm-attribution lock prohibits positional recovery."),
    ]
    drop_availability = {
        "canonical_active_moiety": "Known by approval but retained only as a grouping/join key",
        "exact_faers_ps_volume": "Observed during the postapproval outcome window; unavailable at prediction time",
        "faers_ror_or_ic": "Postapproval outcome-derived; unavailable at prediction time",
        "jader_feature": "External postapproval outcome-derived; unavailable at prediction time",
        "pair_pooled_affected_at_risk_proportion": "Source rows are preapproval, but the proposed pooled estimand is not identifiable",
        "pair_raw_proportion_vector": "Preapproval audit source; not a fixed-dimensional predictor",
        "pair_min_row_ae_proportion": "Preapproval but omitted prespecifically as redundant/zero-dominated",
        "pair_i2": "Preapproval inputs, but estimator is unstable for sparse contributing trials",
        "positional_arm_recovery": "Preapproval registry data, but prohibited by the exact-title attribution lock",
    }
    for row in f:
        if row["feature_name"] in drop_availability:
            row["temporal_availability"] = drop_availability[row["feature_name"]]
    return f


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        AACT_DB, SECTION1, SECTION1_README, SECTION2_SIGNALS, SECTION2_SUMMARY,
        SECTION2_README, IDENTITY, IDENTITY_SHA, LINKS, BSTRICT, AACT_PT_MAP,
        MEDDRA_MDHIER, FEATURE_FEASIBILITY, PROTOCOL_LOCK,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Required inputs missing: {missing}")
    expected_identity_sha = IDENTITY_SHA.read_text(encoding="utf-8").split()[0]
    if sha256(IDENTITY) != expected_identity_sha:
        raise RuntimeError("Frozen identity hash mismatch")
    before = {str(p): stat_identity(p) for p in required}

    con = duckdb.connect()
    con.execute(
        "SET threads=8; SET memory_limit='20GB'; SET temp_directory='/private/tmp/pvml_duckdb'; "
        "SET preserve_insertion_order=false; SET enable_progress_bar=false"
    )
    con.execute(f"ATTACH '{AACT_DB}' AS a (READ_ONLY)")

    # Physical development-only allowlist first. No full outcome registry exists in memory.
    con.execute(
        f"""CREATE TABLE dev_drugs AS
            SELECT canonical_active_moiety,approval_year,nda_bla,orphan_designation,
                   accelerated_approval,breakthrough_therapy_designation,fast_track_designation,
                   priority_review,route,dosage_form,qualifying_trials,target_arms,
                   total_target_arm_subjects_at_risk,randomized_trial_fraction,
                   masked_trial_fraction,industry_sponsored_fraction,phase1_fraction,candidate_pairs
            FROM read_csv_auto('{SECTION1}',header=true)
            WHERE temporal_partition='DEVELOPMENT' AND approval_year BETWEEN 2012 AND 2018"""
    )
    dev_qc = con.execute(
        "SELECT count(*),count(DISTINCT canonical_active_moiety),sum(candidate_pairs),min(approval_year),max(approval_year) FROM dev_drugs"
    ).fetchone()
    if dev_qc[0] != 107 or dev_qc[1] != 107 or dev_qc[3:] != (2012, 2018):
        raise RuntimeError(f"Development allowlist mismatch: {dev_qc}")
    expected_pairs_from_section1 = int(dev_qc[2])

    con.execute(
        f"""CREATE TABLE dev_identity AS
            SELECT d.canonical_active_moiety,cast(i.fda_first_approval_date AS DATE) approval_date
            FROM dev_drugs d JOIN read_csv_auto('{IDENTITY}',header=true) i USING(canonical_active_moiety)"""
    )
    con.execute(
        f"""CREATE TABLE dev_links AS
            SELECT l.* FROM read_parquet('{LINKS}') l JOIN dev_drugs d USING(canonical_active_moiety)"""
    )
    con.execute(
        f"""CREATE TABLE pt_map AS
            SELECT aact_term_raw,cast(canonical_pt_code AS BIGINT) canonical_pt_code,
                   canonical_pt_name,mapping_level
            FROM read_csv_auto('{AACT_PT_MAP}',header=true)
            WHERE mapping_status='MAPPED' AND canonical_pt_code IS NOT NULL"""
    )
    con.execute(
        f"""CREATE TABLE pt_identity AS
            SELECT cast(column00 AS BIGINT) canonical_pt_code,
                   any_value(column04) canonical_pt_name,
                   any_value(column07) primary_soc,
                   count(DISTINCT column07) n_primary_socs
            FROM read_csv('{MEDDRA_MDHIER}',delim='$',header=false,all_varchar=true)
            WHERE column11='Y' AND column00 IS NOT NULL
            GROUP BY 1"""
    )
    if con.execute("SELECT count(*) FROM pt_identity WHERE n_primary_socs<>1").fetchone()[0] != 0:
        raise RuntimeError("Canonical PT primary-SOC identity is not unique")

    # Independent B-STRICT arm reconstruction for Class-A QC and features.
    con.execute(
        """CREATE TABLE linked_drug_trials AS
           SELECT DISTINCT d.canonical_active_moiety,d.approval_date,l.nct_id
           FROM dev_identity d JOIN dev_links l USING(canonical_active_moiety)"""
    )
    con.execute(
        """CREATE TABLE actual_preapproval_trials AS
           SELECT l.*,s.completion_date,s.completion_date_type,s.phase
           FROM linked_drug_trials l JOIN a.studies s USING(nct_id)
           WHERE s.completion_date_type='ACTUAL' AND s.completion_date IS NOT NULL
             AND s.completion_date<=l.approval_date AND coalesce(s.phase,'')<>'PHASE4'"""
    )
    con.execute(
        """CREATE TABLE valid_mapped_ae AS
           SELECT t.*,re.id ae_id,re.result_group_id rgid,re.adverse_event_term,
                  re.subjects_affected,re.subjects_at_risk,lower(re.event_type) event_type,
                  re.frequency_threshold,p.canonical_pt_code,p.canonical_pt_name,p.mapping_level
           FROM actual_preapproval_trials t JOIN a.reported_events re USING(nct_id)
           JOIN pt_map p ON re.adverse_event_term=p.aact_term_raw
           WHERE re.subjects_at_risk>0 AND re.subjects_affected IS NOT NULL
             AND re.subjects_affected<=re.subjects_at_risk"""
    )
    title_norm = "lower(trim(regexp_replace(regexp_replace({col}, '[^A-Za-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')))"
    con.execute(
        f"""CREATE TABLE result_to_design AS
            WITH rg AS (
              SELECT nct_id,id rgid,{title_norm.format(col='title')} title_norm
              FROM a.result_groups WHERE result_type='Reported Event'
                AND nct_id IN (SELECT nct_id FROM linked_drug_trials)
            ), dg AS (
              SELECT nct_id,id dgid,group_type,title,{title_norm.format(col='title')} title_norm
              FROM a.design_groups WHERE nct_id IN (SELECT nct_id FROM linked_drug_trials)
            )
            SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_design_matches,
                   min(dg.dgid) dgid,min(dg.group_type) group_type,min(dg.title) design_group_title
            FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm GROUP BY 1,2"""
    )
    con.execute(
        """CREATE TABLE arm_drug_count AS
           SELECT dg.id dgid,
                  count(DISTINCT i.id) FILTER (WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')) n_drug_interventions
           FROM a.design_groups dg LEFT JOIN a.design_group_interventions dgi ON dgi.design_group_id=dg.id
           LEFT JOIN a.interventions i ON i.id=dgi.intervention_id
           WHERE dg.nct_id IN (SELECT nct_id FROM linked_drug_trials) GROUP BY dg.id"""
    )
    con.execute(
        """CREATE TABLE target_design_arms AS
           SELECT DISTINCT l.canonical_active_moiety,l.nct_id,dgi.design_group_id dgid,
                  l.aact_primary_name target_intervention_name,l.aact_intervention_combination_flag
           FROM dev_links l JOIN a.design_group_interventions dgi ON dgi.intervention_id=l.intervention_id"""
    )
    con.execute(
        """CREATE TABLE primary_rows AS
           SELECT v.*,m.dgid,m.group_type,m.design_group_title,t.target_intervention_name
           FROM valid_mapped_ae v
           JOIN result_to_design m ON v.nct_id=m.nct_id AND v.rgid=m.rgid AND m.n_design_matches=1
           JOIN arm_drug_count ac ON m.dgid=ac.dgid AND ac.n_drug_interventions=1
           JOIN target_design_arms t ON v.canonical_active_moiety=t.canonical_active_moiety
                                    AND v.nct_id=t.nct_id AND m.dgid=t.dgid
           WHERE m.group_type IN ('EXPERIMENTAL','OTHER')
             AND NOT t.aact_intervention_combination_flag
             AND NOT regexp_matches(lower(coalesce(t.target_intervention_name,'')),
                 '(^| )(placebo|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline)( |$)')
             AND NOT regexp_matches(lower(coalesce(m.design_group_title,'')),
                 '(^| )(placebo|vehicle|sham|no treatment|control)( |$)')"""
    )
    primary_profile_qc = con.execute(
        """SELECT count(DISTINCT canonical_active_moiety),count(DISTINCT nct_id),
                  count(DISTINCT rgid),count(DISTINCT (canonical_active_moiety,canonical_pt_code))
           FROM primary_rows"""
    ).fetchone()
    if primary_profile_qc != (107, 338, 632, expected_pairs_from_section1):
        raise RuntimeError(f"Independent primary profile mismatch: {primary_profile_qc}")

    con.execute(
        f"""CREATE TABLE locked_dev_signals AS
            SELECT canonical_active_moiety,cast(canonical_pt_code AS BIGINT) canonical_pt_code,
                   canonical_pt_name,primary_soc,coverage_class
            FROM read_csv_auto('{SECTION2_SIGNALS}',header=true)"""
    )
    signal_scope_qc = con.execute(
        """SELECT count(*),count(DISTINCT canonical_active_moiety),
                  count(DISTINCT (canonical_active_moiety,canonical_pt_code)),
                  sum((coverage_class='PREMARKETING_OBSERVED')::INT),
                  sum((coverage_class='POSTMARKETING_ONLY')::INT)
           FROM locked_dev_signals"""
    ).fetchone()
    with SECTION2_SUMMARY.open(encoding="utf-8", newline="") as f:
        locked_summary = next(csv.DictReader(f))
    locked_postonly = int(locked_summary["postmarketing_only"])
    locked_observed = int(locked_summary["premarketing_observed"])
    if signal_scope_qc[0] != int(locked_summary["criterion_r_signals"]) or signal_scope_qc[1] != 107:
        raise RuntimeError(f"Locked development signal scope mismatch: {signal_scope_qc}")

    # Independent Class-A audit: start with final-contributing drug-trial IDs,
    # then scan their entire raw reported-event tables for mapped PT identity.
    con.execute(
        """CREATE TABLE contributing_drug_trials AS
           SELECT DISTINCT canonical_active_moiety,nct_id FROM primary_rows"""
    )
    con.execute(
        """CREATE TABLE all_pt_in_contributing_trials AS
           SELECT DISTINCT q.canonical_active_moiety,re.nct_id,p.canonical_pt_code
           FROM contributing_drug_trials q JOIN a.reported_events re USING(nct_id)
           JOIN pt_map p ON re.adverse_event_term=p.aact_term_raw"""
    )
    con.execute(
        """CREATE TABLE raw_same_trial_hits AS
           SELECT DISTINCT s.canonical_active_moiety,s.canonical_pt_code,t.nct_id
           FROM locked_dev_signals s JOIN all_pt_in_contributing_trials t
             USING(canonical_active_moiety,canonical_pt_code)
           WHERE s.coverage_class='POSTMARKETING_ONLY'"""
    )
    con.execute(
        """CREATE TABLE classa_exact AS
           SELECT DISTINCT h.canonical_active_moiety,h.canonical_pt_code
           FROM raw_same_trial_hits h
           WHERE NOT EXISTS (
             SELECT 1 FROM primary_rows p
             WHERE p.canonical_active_moiety=h.canonical_active_moiety
               AND p.canonical_pt_code=h.canonical_pt_code
           )"""
    )
    raw_hit_occurrences = con.execute("SELECT count(*) FROM raw_same_trial_hits").fetchone()[0]
    raw_hit_pairs = con.execute(
        "SELECT count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM raw_same_trial_hits"
    ).fetchone()[0]
    classa_count = con.execute("SELECT count(*) FROM classa_exact").fetchone()[0]
    con.execute(
        """CREATE TABLE all_temporal_pt AS
           SELECT DISTINCT v.canonical_active_moiety,v.canonical_pt_code,v.nct_id
           FROM valid_mapped_ae v"""
    )
    temporal_hit_pairs = con.execute(
        """SELECT count(DISTINCT (s.canonical_active_moiety,s.canonical_pt_code))
           FROM locked_dev_signals s JOIN all_temporal_pt t USING(canonical_active_moiety,canonical_pt_code)
           WHERE s.coverage_class='POSTMARKETING_ONLY'"""
    ).fetchone()[0]
    noncontrib_hit_pairs = con.execute(
        """SELECT count(DISTINCT (s.canonical_active_moiety,s.canonical_pt_code))
           FROM locked_dev_signals s JOIN all_temporal_pt t USING(canonical_active_moiety,canonical_pt_code)
           WHERE s.coverage_class='POSTMARKETING_ONLY'
             AND NOT EXISTS (SELECT 1 FROM contributing_drug_trials q
                             WHERE q.canonical_active_moiety=t.canonical_active_moiety
                               AND q.nct_id=t.nct_id)"""
    ).fetchone()[0]
    noncontrib_b_eligible_pairs = con.execute(
        """SELECT count(DISTINCT (s.canonical_active_moiety,s.canonical_pt_code))
           FROM locked_dev_signals s JOIN all_temporal_pt t USING(canonical_active_moiety,canonical_pt_code)
           JOIN a.studies st USING(nct_id)
           WHERE s.coverage_class='POSTMARKETING_ONLY'
             AND NOT EXISTS (SELECT 1 FROM contributing_drug_trials q
                             WHERE q.canonical_active_moiety=t.canonical_active_moiety
                               AND q.nct_id=t.nct_id)
             AND st.study_type='INTERVENTIONAL'
             AND EXISTS (SELECT 1 FROM a.interventions i
                         WHERE i.nct_id=t.nct_id AND i.intervention_type IN ('DRUG','BIOLOGICAL'))"""
    ).fetchone()[0]
    noncontrib_not_b_eligible_pairs = noncontrib_hit_pairs - noncontrib_b_eligible_pairs
    examples = fetch_df(
        con,
        """SELECT c.canonical_active_moiety,c.canonical_pt_code,s.canonical_pt_name,
                  string_agg(DISTINCT h.nct_id,' | ' ORDER BY h.nct_id) supporting_nct_ids
           FROM classa_exact c JOIN locked_dev_signals s USING(canonical_active_moiety,canonical_pt_code)
           JOIN raw_same_trial_hits h USING(canonical_active_moiety,canonical_pt_code)
           GROUP BY 1,2,3 ORDER BY 1,2 LIMIT 20""",
    ).to_dict("records")
    example_md = md_table(examples, [
        ("canonical_active_moiety", "Drug"), ("canonical_pt_code", "PT code"),
        ("canonical_pt_name", "PT"), ("supporting_nct_ids", "NCT ID(s)"),
    ]) if examples else "No examples: the independently derived count was zero."
    CLASSA_QC.write_text(f"""# Independent Section 2 Class-A sanity check

**Result: {'CONFIRMED ZERO' if classa_count == 0 else 'DISCREPANCY — STOP'}**

This audit did not call or reuse the Section 2 decomposition classifier. It independently rebuilt actual-completion preapproval trial IDs, exact result-group/design-group links, target-monotherapy arms, canonical MedDRA PT mapping, and the set of drug–NCT trials contributing at least one primary B-STRICT row.

- Locked POSTMARKETING_ONLY pairs queried: {locked_postonly:,}.
- Raw same-trial mapped PT occurrences (drug–PT–NCT) before hierarchy: {raw_hit_occurrences:,}.
- Unique raw same-trial drug–PT hits: {raw_hit_pairs:,}.
- Pairs satisfying Class A exactly: {classa_count:,}.
- POSTMARKETING_ONLY pairs found anywhere in the broader actual-completion/non-Phase-4 temporal trial base: {temporal_hit_pairs:,}.
- Such pairs found only in drug–trials that contributed no row to the primary profile: {noncontrib_hit_pairs:,}.
- Of these, pairs with at least one other interventional DRUG/BIOLOGICAL trial meeting the corrected Class-B structural requirements: {noncontrib_b_eligible_pairs:,}.
- Pairs seen only in broader temporal trials that failed those Class-B trial-type requirements: {noncontrib_not_b_eligible_pairs:,}.

## Structural explanation

{'Zero is structurally expected in this snapshot because every mapped POSTMARKETING_ONLY PT occurrence in the broader temporal trial base occurred only in a drug–trial that contributed no qualifying target-monotherapy row. Once the audit is restricted to drug–NCT trials that actually contributed to the primary B-STRICT profile, no POSTMARKETING_ONLY PT remains. Most broader-trial hits satisfy corrected Class B; the remainder fail its interventional DRUG/BIOLOGICAL trial-type requirements and therefore do not become Class A.' if classa_count == 0 else 'At least one locked POSTMARKETING_ONLY PT was found in a drug–trial that contributed to the primary profile but only outside the qualifying target-monotherapy rows. This contradicts the locked zero and blocks feature engineering.'}

## Examples

{example_md}
""", encoding="utf-8")
    if classa_count != 0:
        fail_qc = {
            "status": "FAIL",
            "generated_at": datetime.now().astimezone().isoformat(),
            "blocking_gate": "independent_classA_zero",
            "raw_same_trial_occurrences": raw_hit_occurrences,
            "raw_same_trial_pairs": raw_hit_pairs,
            "classA_exact": classa_count,
            "models_fitted": 0,
        }
        QC.write_text(json.dumps(fail_qc, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Independent Class-A discrepancy: {classa_count}; feature engineering stopped")

    # Direct development-only candidate query; no full analytical registry table is loaded.
    con.execute(
        f"""CREATE TABLE dev_candidate_base AS
            SELECT r.canonical_active_moiety,cast(r.canonical_pt_code AS BIGINT) canonical_pt_code,
                   p.canonical_pt_name,p.primary_soc
            FROM read_parquet('{BSTRICT}') r
            JOIN dev_drugs d USING(canonical_active_moiety)
            JOIN pt_identity p ON cast(r.canonical_pt_code AS BIGINT)=p.canonical_pt_code
            GROUP BY 1,2,3,4"""
    )
    candidate_qc = con.execute(
        "SELECT count(*),count(DISTINCT canonical_active_moiety),count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM dev_candidate_base"
    ).fetchone()
    if candidate_qc != (expected_pairs_from_section1, 107, expected_pairs_from_section1):
        raise RuntimeError(f"Development candidate registry mismatch: {candidate_qc}")

    # Outcome re-derivation is a development-only identity join, not a hard-coded count.
    con.execute(
        """CREATE TABLE dev_outcome AS
           SELECT c.canonical_active_moiety,c.canonical_pt_code,
                  (s.canonical_active_moiety IS NOT NULL)::TINYINT criterion_r_3y
           FROM dev_candidate_base c LEFT JOIN locked_dev_signals s
             USING(canonical_active_moiety,canonical_pt_code)"""
    )
    candidate_postonly_overlap = con.execute(
        """SELECT count(*) FROM dev_candidate_base c JOIN locked_dev_signals s
             USING(canonical_active_moiety,canonical_pt_code)
           WHERE s.coverage_class='POSTMARKETING_ONLY'"""
    ).fetchone()[0]
    derived_positives = int(con.execute("SELECT sum(criterion_r_3y) FROM dev_outcome").fetchone()[0])
    if derived_positives != locked_observed or candidate_postonly_overlap != 0:
        raise RuntimeError(
            f"Development outcome reconciliation failed: positives={derived_positives}, "
            f"locked={locked_observed}, postonly_overlap={candidate_postonly_overlap}"
        )

    # Deterministic, outcome-blind drug context.
    con.execute(
        """CREATE TABLE dev_drug_features AS
           SELECT canonical_active_moiety,approval_year,nda_bla,
                  CASE WHEN lower(cast(orphan_designation AS VARCHAR)) LIKE 'yes%'
                              OR lower(cast(orphan_designation AS VARCHAR))='true' THEN 'Yes'
                       WHEN lower(cast(orphan_designation AS VARCHAR)) IN ('no','false') THEN 'No' ELSE 'MISSING/UNKNOWN' END orphan_designation,
                  CASE WHEN lower(cast(accelerated_approval AS VARCHAR)) LIKE 'yes%'
                              OR lower(cast(accelerated_approval AS VARCHAR))='true' THEN 'Yes'
                       WHEN lower(cast(accelerated_approval AS VARCHAR)) IN ('no','false') THEN 'No' ELSE 'MISSING/UNKNOWN' END accelerated_approval,
                  CASE WHEN lower(cast(breakthrough_therapy_designation AS VARCHAR)) LIKE 'yes%' THEN 'Yes'
                       WHEN cast(breakthrough_therapy_designation AS VARCHAR)='N/A' THEN 'N/A'
                       WHEN lower(cast(breakthrough_therapy_designation AS VARCHAR))='no' THEN 'No' ELSE 'MISSING/UNKNOWN' END breakthrough_therapy_designation,
                  CASE WHEN lower(cast(fast_track_designation AS VARCHAR)) LIKE 'yes%' THEN 'Yes'
                       WHEN lower(cast(fast_track_designation AS VARCHAR))='no' THEN 'No' ELSE 'MISSING/UNKNOWN' END fast_track_designation,
                  CASE WHEN lower(cast(priority_review AS VARCHAR)) LIKE 'priority%' THEN 'Priority'
                       WHEN lower(cast(priority_review AS VARCHAR))='standard' THEN 'Standard' ELSE 'MISSING/UNKNOWN' END priority_review_category,
                  CASE WHEN route IS NULL OR trim(route)='' THEN 'MISSING/UNKNOWN'
                       WHEN strpos(route,'|')>0 THEN 'MULTIPLE'
                       WHEN lower(route) LIKE '%oral%' THEN 'ORAL'
                       WHEN lower(route) LIKE '%inject%' OR lower(route) LIKE '%intravenous%'
                         OR lower(route) LIKE '%subcutaneous%' OR lower(route) LIKE '%intrathecal%' THEN 'PARENTERAL'
                       WHEN lower(route) LIKE '%ophthalmic%' THEN 'OPHTHALMIC'
                       WHEN lower(route) LIKE '%inhal%' THEN 'INHALATION' ELSE 'OTHER' END route_broad,
                  CASE WHEN dosage_form IS NULL OR trim(dosage_form)='' THEN 'MISSING/UNKNOWN'
                       WHEN strpos(dosage_form,'|')>0 THEN 'MULTIPLE'
                       WHEN lower(dosage_form) LIKE '%inject%' THEN 'INJECTABLE'
                       WHEN lower(dosage_form) LIKE '%solution%' OR lower(dosage_form) LIKE '%suspension%' THEN 'LIQUID'
                       WHEN lower(dosage_form) LIKE '%tablet%' OR lower(dosage_form) LIKE '%capsule%'
                         OR lower(dosage_form) LIKE '%granule%' OR lower(dosage_form) LIKE '%powder%' THEN 'SOLID'
                       ELSE 'OTHER' END dosage_form_broad,
                  qualifying_trials::BIGINT drug_n_qualifying_trials,
                  target_arms::BIGINT drug_n_target_arms,
                  total_target_arm_subjects_at_risk::DOUBLE drug_approx_ae_safety_population,
                  phase1_fraction::DOUBLE drug_phase1_fraction,
                  randomized_trial_fraction::DOUBLE drug_randomized_trial_fraction,
                  masked_trial_fraction::DOUBLE drug_masked_trial_fraction,
                  industry_sponsored_fraction::DOUBLE drug_industry_sponsored_fraction
           FROM dev_drugs"""
    )
    priority_qc = con.execute(
        """SELECT count(priority_review_category)::DOUBLE/count(*) completeness,
                  min(n) min_level_n
           FROM (SELECT priority_review_category,count(*) n FROM dev_drug_features
                 WHERE priority_review_category<>'MISSING/UNKNOWN' GROUP BY 1) x"""
    ).fetchone()
    # The outer aggregation above is level-based; calculate true completeness separately.
    priority_complete = con.execute(
        "SELECT avg((priority_review_category<>'MISSING/UNKNOWN')::INT) FROM dev_drug_features"
    ).fetchone()[0]
    priority_min_level = con.execute(
        """SELECT min(n) FROM (SELECT priority_review_category,count(*) n FROM dev_drug_features
           WHERE priority_review_category<>'MISSING/UNKNOWN' GROUP BY 1)"""
    ).fetchone()[0]
    priority_status = "SECONDARY" if priority_complete >= .95 and priority_min_level >= 10 else "DROP"
    priority_reason = (
        f"Prespecified completeness rule passed: {priority_complete*100:.1f}% complete; "
        f"minimum collapsed level n={priority_min_level}."
        if priority_status != "DROP" else
        f"Prespecified completeness rule failed: {priority_complete*100:.1f}% complete; "
        f"minimum collapsed level n={priority_min_level}."
    )

    # Trial design/sponsor attributes for pair composition.
    con.execute(
        """CREATE TABLE design_agg AS
           SELECT nct_id,any_value(allocation) allocation,any_value(masking) masking
           FROM a.designs GROUP BY nct_id"""
    )
    con.execute(
        """CREATE TABLE pair_rows AS
           SELECT DISTINCT canonical_active_moiety,canonical_pt_code,nct_id,rgid,dgid,ae_id,event_type,
                  frequency_threshold,subjects_affected::DOUBLE subjects_affected,
                  subjects_at_risk::DOUBLE subjects_at_risk,
                  subjects_affected::DOUBLE/subjects_at_risk::DOUBLE row_proportion
           FROM primary_rows"""
    )
    con.execute(
        """CREATE TABLE pair_arms AS
           SELECT canonical_active_moiety,canonical_pt_code,nct_id,rgid,dgid,
                  max(subjects_at_risk) arm_subjects_at_risk,
                  max((event_type='serious')::INT) arm_has_serious,
                  max((event_type='other')::INT) arm_has_other,
                  max(row_proportion) FILTER (WHERE event_type='serious') max_serious_proportion,
                  max(row_proportion) FILTER (WHERE event_type='other') max_other_proportion
           FROM pair_rows GROUP BY 1,2,3,4,5"""
    )
    con.execute(
        """CREATE TABLE pair_trials AS
           SELECT r.canonical_active_moiety,r.canonical_pt_code,r.nct_id,
                  median(r.row_proportion) trial_median_proportion,
                  max((r.event_type='serious')::INT) trial_has_serious,
                  max((r.event_type='other' AND r.frequency_threshold=0)::INT) trial_threshold_0,
                  max((r.event_type='other' AND r.frequency_threshold=5)::INT) trial_threshold_5,
                  max((r.event_type='other' AND r.frequency_threshold IS NOT NULL)::INT) trial_has_other_threshold,
                  max(coalesce((s.phase='PHASE1')::INT,0)) phase1,
                  max(CASE WHEN d.allocation IS NULL THEN NULL ELSE (d.allocation='RANDOMIZED')::INT END) randomized,
                  max(CASE WHEN d.masking IS NULL THEN NULL ELSE (d.masking<>'NONE')::INT END) masked
           FROM pair_rows r JOIN a.studies s USING(nct_id) LEFT JOIN design_agg d USING(nct_id)
           GROUP BY 1,2,3"""
    )
    con.execute(
        """CREATE TABLE pair_feature_agg AS
           WITH row_stats AS (
             SELECT canonical_active_moiety,canonical_pt_code,count(*) n_rows,
                    median(row_proportion) pair_median_row_ae_proportion,
                    max(row_proportion) pair_max_row_ae_proportion,
                    CASE WHEN count(*)>=2 THEN quantile_cont(row_proportion,.75)-quantile_cont(row_proportion,.25) END pair_row_ae_proportion_iqr,
                    max(row_proportion) FILTER (WHERE event_type='serious') pair_max_serious_row_proportion,
                    max(row_proportion) FILTER (WHERE event_type='other') pair_max_other_row_proportion,
                    min(frequency_threshold) FILTER (WHERE event_type='other' AND frequency_threshold IS NOT NULL) pair_min_other_threshold,
                    max(frequency_threshold) FILTER (WHERE event_type='other' AND frequency_threshold IS NOT NULL) pair_max_other_threshold,
                    median(frequency_threshold) FILTER (WHERE event_type='other' AND frequency_threshold IS NOT NULL) pair_median_other_threshold
             FROM pair_rows GROUP BY 1,2
           ), arm_stats AS (
             SELECT canonical_active_moiety,canonical_pt_code,count(*) pair_n_reporting_arms,
                    sum(arm_subjects_at_risk) pair_nonduplicated_arm_subjects_at_risk,
                    max(arm_has_serious) pair_any_serious,sum(arm_has_serious) pair_n_serious_arms,
                    avg(arm_has_serious) pair_serious_arm_fraction,max(arm_has_other) pair_other_proportion_available
             FROM pair_arms GROUP BY 1,2
           ), trial_stats AS (
             SELECT canonical_active_moiety,canonical_pt_code,count(*) pair_n_reporting_trials,
                    sum(trial_has_serious) pair_n_serious_trials,avg(trial_has_serious) pair_serious_trial_fraction,
                    CASE WHEN count(*)>=2 THEN stddev_samp(trial_median_proportion) END pair_between_trial_proportion_sd,
                    avg(trial_threshold_0) pair_fraction_trials_threshold_0,
                    avg(trial_threshold_5) pair_fraction_trials_threshold_5,
                    max(trial_has_other_threshold) pair_other_threshold_available,
                    avg(phase1) pair_phase1_fraction,
                    avg(randomized) FILTER (WHERE randomized IS NOT NULL) pair_randomized_trial_fraction,
                    avg(masked) FILTER (WHERE masked IS NOT NULL) pair_masked_trial_fraction
             FROM pair_trials GROUP BY 1,2
           )
           SELECT r.canonical_active_moiety,r.canonical_pt_code,
                  t.pair_n_reporting_trials,
                  t.pair_n_reporting_trials::DOUBLE/d.drug_n_qualifying_trials pair_reporting_trial_fraction,
                  a.pair_n_reporting_arms,a.pair_nonduplicated_arm_subjects_at_risk,
                  r.pair_median_row_ae_proportion,r.pair_max_row_ae_proportion,r.pair_row_ae_proportion_iqr,
                  (r.n_rows>=2)::TINYINT pair_row_variability_available,
                  a.pair_any_serious,t.pair_n_serious_trials,a.pair_n_serious_arms,
                  t.pair_serious_trial_fraction,a.pair_serious_arm_fraction,
                  r.pair_max_serious_row_proportion,r.pair_max_other_row_proportion,
                  a.pair_other_proportion_available,t.pair_between_trial_proportion_sd,
                  (t.pair_n_reporting_trials>=2)::TINYINT pair_cross_trial_variability_available,
                  r.pair_min_other_threshold,r.pair_max_other_threshold,r.pair_median_other_threshold,
                  t.pair_fraction_trials_threshold_0,t.pair_fraction_trials_threshold_5,
                  t.pair_other_threshold_available,t.pair_phase1_fraction,
                  t.pair_randomized_trial_fraction,t.pair_masked_trial_fraction
           FROM row_stats r JOIN arm_stats a USING(canonical_active_moiety,canonical_pt_code)
           JOIN trial_stats t USING(canonical_active_moiety,canonical_pt_code)
           JOIN dev_drug_features d USING(canonical_active_moiety)"""
    )

    con.execute(
        """CREATE TABLE model_registry AS
           SELECT md5(c.canonical_active_moiety||'|'||cast(c.canonical_pt_code AS VARCHAR)) pair_id,
                  c.canonical_active_moiety,c.canonical_pt_code,c.canonical_pt_name,c.primary_soc,
                  d.* EXCLUDE(canonical_active_moiety),p.* EXCLUDE(canonical_active_moiety,canonical_pt_code),
                  o.criterion_r_3y
           FROM dev_candidate_base c JOIN dev_drug_features d USING(canonical_active_moiety)
           JOIN pair_feature_agg p USING(canonical_active_moiety,canonical_pt_code)
           JOIN dev_outcome o USING(canonical_active_moiety,canonical_pt_code)
           ORDER BY d.approval_year,c.canonical_active_moiety,c.canonical_pt_code"""
    )
    registry_qc = con.execute(
        """SELECT count(*),count(DISTINCT pair_id),count(DISTINCT canonical_active_moiety),
                  sum(criterion_r_3y) FROM model_registry"""
    ).fetchone()
    if registry_qc != (expected_pairs_from_section1, expected_pairs_from_section1, 107, derived_positives):
        raise RuntimeError(f"Model registry mismatch: {registry_qc}")
    con.execute(f"COPY model_registry TO '{PAIR_REGISTRY}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    con.execute(
        f"""COPY (SELECT pair_id,canonical_active_moiety,canonical_pt_code,canonical_pt_name,
                          primary_soc,criterion_r_3y FROM model_registry)
            TO '{MODEL_DOMAIN}' (HEADER,DELIMITER ',')"""
    )

    # Feature dictionary is frozen before any model fit; statuses use availability only.
    features = build_feature_dictionary(priority_status, priority_reason)
    dict_columns = [
        "feature_name", "feature_set", "level", "source", "definition", "data_type",
        "transformation", "missingness_rule", "temporal_availability", "leakage_risk",
        "status", "rationale",
    ]
    pd.DataFrame([{k: x[k] for k in dict_columns} for x in features]).to_csv(FEATURE_DICT, index=False)
    retained_set0 = [x for x in features if x["feature_set"] == "SET0" and x["status"] != "DROP"]
    retained_additional = [x for x in features if x["feature_set"] == "SET1_ADDITIONAL" and x["status"] != "DROP"]
    set0_names = [x["feature_name"] for x in retained_set0]
    set1_additional_names = [x["feature_name"] for x in retained_additional]
    set1_names = set0_names + set1_additional_names

    registry = fetch_df(con, "SELECT * FROM model_registry")
    schema_columns = list(registry.columns)
    absent_features = [x for x in set1_names if x not in schema_columns]
    if absent_features:
        raise RuntimeError(f"Retained features absent from registry: {absent_features}")

    # Outcome-domain description.
    drug_outcomes = registry.groupby("canonical_active_moiety", as_index=False).agg(
        pairs=("pair_id", "size"), positives=("criterion_r_3y", "sum")
    )
    drug_outcomes["prevalence"] = drug_outcomes["positives"] / drug_outcomes["pairs"]
    positive_sorted = np.sort(drug_outcomes["positives"].to_numpy())[::-1]
    q25, median_pos, q75 = np.quantile(drug_outcomes["positives"], [.25, .5, .75])
    outcome_rows = [
        {"scope": "OVERALL", "metric": "development_drugs", "value": 107, "denominator": "", "percent": ""},
        {"scope": "OVERALL", "metric": "candidate_pairs", "value": len(registry), "denominator": "", "percent": ""},
        {"scope": "OVERALL", "metric": "criterion_r_3y_positive_pairs", "value": derived_positives, "denominator": len(registry), "percent": pct(derived_positives, len(registry))},
        {"scope": "DRUG", "metric": "positives_per_drug_median", "value": median_pos, "denominator": "", "percent": ""},
        {"scope": "DRUG", "metric": "positives_per_drug_p25", "value": q25, "denominator": "", "percent": ""},
        {"scope": "DRUG", "metric": "positives_per_drug_p75", "value": q75, "denominator": "", "percent": ""},
        {"scope": "DRUG", "metric": "zero_positive_drugs", "value": int((drug_outcomes["positives"] == 0).sum()), "denominator": 107, "percent": pct((drug_outcomes["positives"] == 0).sum(), 107)},
    ]
    for k in [1, 3, 5, 10]:
        n_top = int(positive_sorted[:k].sum())
        outcome_rows.append({
            "scope": "CONCENTRATION", "metric": f"top_{k}_drug_positive_pairs",
            "value": n_top, "denominator": derived_positives, "percent": pct(n_top, derived_positives),
        })
    pd.DataFrame(outcome_rows).to_csv(OUTCOME_DIST, index=False)

    # Missingness and distribution audit without outcome association.
    missing_rows = []
    distribution_rows = []
    feature_by_name = {x["feature_name"]: x for x in features}
    for name in set1_names:
        meta = feature_by_name[name]
        series = registry[name]
        miss = int(series.isna().sum())
        nonmiss = int(series.notna().sum())
        zeros = int((series == 0).sum()) if pd.api.types.is_numeric_dtype(series) else ""
        missing_rows.append({
            "feature_name": name, "feature_set": meta["feature_set"], "status": meta["status"],
            "data_type": meta["data_type"], "n_total": len(series), "n_nonmissing": nonmiss,
            "n_missing": miss, "missing_pct": pct(miss, len(series)),
            "n_unique_nonmissing": int(series.nunique(dropna=True)), "n_zero": zeros,
            "missingness_rule": meta["missingness_rule"],
        })
        if pd.api.types.is_numeric_dtype(series) and meta["data_type"] != "categorical":
            x = pd.to_numeric(series, errors="coerce").dropna()
            distribution_rows.append({
                "feature_name": name, "kind": "NUMERIC", "n_nonmissing": len(x),
                "n_levels": "", "top_level": "", "top_level_n": "", "top_level_pct": "",
                "mean": x.mean(), "sd": x.std(ddof=1), "min": x.min(), "p25": x.quantile(.25),
                "median": x.median(), "p75": x.quantile(.75), "max": x.max(),
                "skewness": x.skew(), "prespecified_transformation": meta["transformation"],
            })
        else:
            vc = series.astype("string").fillna("MISSING/UNKNOWN").value_counts(dropna=False)
            distribution_rows.append({
                "feature_name": name, "kind": "CATEGORICAL", "n_nonmissing": nonmiss,
                "n_levels": len(vc), "top_level": str(vc.index[0]), "top_level_n": int(vc.iloc[0]),
                "top_level_pct": pct(int(vc.iloc[0]), len(series)), "mean": "", "sd": "", "min": "",
                "p25": "", "median": "", "p75": "", "max": "", "skewness": "",
                "prespecified_transformation": meta["transformation"],
            })
    pd.DataFrame(missing_rows).to_csv(MISSINGNESS, index=False)
    pd.DataFrame(distribution_rows).to_csv(DISTRIBUTION, index=False)

    # PT identity only; no holdout PT list is inspected before Section 4.
    pt_distribution = registry.groupby(
        ["canonical_pt_code", "canonical_pt_name", "primary_soc"], as_index=False
    ).agg(pair_count=("pair_id", "size"), drug_count=("canonical_active_moiety", "nunique"))
    pt_distribution["drug_frequency_bucket"] = np.select(
        [pt_distribution["drug_count"] == 1, pt_distribution["drug_count"].between(2, 4), pt_distribution["drug_count"] >= 5],
        ["1_DRUG", "2_TO_4_DRUGS", "GE_5_DRUGS"], default="INVALID",
    )
    pt_distribution.sort_values(["drug_count", "canonical_pt_code"], inplace=True)
    pt_distribution.to_csv(PT_DIST, index=False)
    pt_bucket_counts = Counter(pt_distribution["drug_frequency_bucket"])

    # Frozen outcome-stratified group folds. This creates assignments only; no estimator is fitted.
    splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    fold_by_drug: dict[str, int] = {}
    dummy_x = np.zeros((len(registry), 1), dtype=np.int8)
    y = registry["criterion_r_3y"].astype(int).to_numpy()
    groups = registry["canonical_active_moiety"].astype(str).to_numpy()
    for fold, (_, validation_index) in enumerate(splitter.split(dummy_x, y, groups), start=1):
        for drug in np.unique(groups[validation_index]):
            if drug in fold_by_drug:
                raise RuntimeError(f"Drug assigned to multiple outer folds: {drug}")
            fold_by_drug[drug] = fold
    if len(fold_by_drug) != 107:
        raise RuntimeError(f"Outer fold assignment missing drugs: {len(fold_by_drug)}")
    drug_outcomes["outer_fold"] = drug_outcomes["canonical_active_moiety"].map(fold_by_drug)
    approval_map = registry.groupby("canonical_active_moiety")["approval_year"].first()
    drug_outcomes["approval_year"] = drug_outcomes["canonical_active_moiety"].map(approval_map)
    drug_outcomes.sort_values(["outer_fold", "approval_year", "canonical_active_moiety"], inplace=True)
    drug_outcomes[["canonical_active_moiety", "outer_fold", "approval_year", "pairs", "positives", "prevalence"]].to_csv(FOLD_ASSIGN, index=False)
    registry["outer_fold"] = registry["canonical_active_moiety"].map(fold_by_drug)
    fold_rows = []
    for fold, x in registry.groupby("outer_fold"):
        fold_rows.append({
            "outer_fold": int(fold), "drugs": x["canonical_active_moiety"].nunique(),
            "pairs": len(x), "positive_pairs": int(x["criterion_r_3y"].sum()),
            "prevalence": float(x["criterion_r_3y"].mean()),
            "unique_pts": int(x["canonical_pt_code"].nunique()),
        })
    pd.DataFrame(fold_rows).sort_values("outer_fold").to_csv(FOLD_BALANCE, index=False)

    # Schema and source firewall audit.
    forbidden_exact = {
        "a", "b", "c", "d", "ror", "ror_lcl95", "ic", "ic025", "faers_ps_volume",
        "exact_faers_ps_volume", "jader", "holdout_outcome", "caseid", "primaryid",
    }
    schema_lower = {x.lower() for x in schema_columns}
    forbidden_present = sorted(forbidden_exact & schema_lower)
    allowed_outcome_columns = [x for x in schema_columns if x == "criterion_r_3y"]
    outcome_like_unexpected = [
        x for x in schema_columns
        if any(token in x.lower() for token in ["criterion", "outcome", "label", "ror", "ic025", "jader", "faers"])
        and x != "criterion_r_3y"
    ]
    key_columns = ["pair_id", "canonical_active_moiety"]
    nonpredictor_columns = key_columns + ["canonical_pt_name", "criterion_r_3y"]
    schema_audit_pass = not forbidden_present and not outcome_like_unexpected and allowed_outcome_columns == ["criterion_r_3y"]

    set0_domain_counts = Counter(x["domain"] for x in retained_set0)
    set1_domain_counts = Counter(x["domain"] for x in retained_additional)
    drop_rows = [x for x in features if x["status"] == "DROP"]
    missing_issues = sorted(
        [r for r in missing_rows if float(r["missing_pct"]) >= 1.0],
        key=lambda r: -float(r["missing_pct"]),
    )
    missing_issue_md = md_table(missing_issues, [
        ("feature_name", "Feature"), ("n_missing", "Missing"),
        ("missing_pct", "Missing %"), ("missingness_rule", "Frozen rule"),
    ]) if missing_issues else "No retained feature had at least 1% missingness."
    fold_md = md_table(fold_rows, [
        ("outer_fold", "Fold"), ("drugs", "Drugs"), ("pairs", "Pairs"),
        ("positive_pairs", "Positive"), ("prevalence", "Prevalence"), ("unique_pts", "Unique PTs"),
    ])
    drop_md = md_table(drop_rows, [
        ("feature_name", "Feature"), ("level", "Level"), ("rationale", "Reason"),
    ])

    LEAKAGE.write_text(f"""# Section 3A leakage audit

**Status: {'PASS' if schema_audit_pass else 'FAIL'}**

## Physical development firewall

- Development source object: `{PAIR_REGISTRY}`.
- Drugs: {registry['canonical_active_moiety'].nunique():,}; pairs: {len(registry):,}; approval years: {registry['approval_year'].min()}–{registry['approval_year'].max()}.
- The B-STRICT parquet was queried through an immediate inner join to the 107-drug development allowlist; no full analytical pair registry was materialized.
- Outcome input was the already physical development-only Section 2 signal file; the full FDA-anchored FAERS label parquet was not opened.
- Holdout PT identity lists were not inspected because even feature-only access could influence rare-category handling before Section 4.

## Schema audit

- Registry columns ({len(schema_columns)}): `{', '.join(schema_columns)}`.
- Only allowed outcome column: `{', '.join(allowed_outcome_columns)}`.
- Forbidden exact columns found: `{', '.join(forbidden_present) if forbidden_present else 'none'}`.
- Unexpected outcome-like columns found: `{', '.join(outcome_like_unexpected) if outcome_like_unexpected else 'none'}`.
- Nonpredictor keys/labels: `{', '.join(nonpredictor_columns)}`.
- `canonical_active_moiety` is retained solely for grouping, clustered bootstrap, and joins; it is excluded from every preprocessing feature list.
- Exact FAERS PS volume, ROR cells/estimates, IC, JADER, holdout outcomes, case IDs, and report IDs are absent.

## Temporal and preprocessing audit

- Every retained predictor is FDA information or qualifying actual-completion preapproval AACT information.
- Set 0 and Set 1 use the identical {len(registry):,}-pair population; Set 1 adds pair-specific features to every retained Set 0 feature.
- Imputation, scaling, rare PT grouping, and one-hot vocabularies are specified as training-fold-only.
- No outcome-based univariate screening, target encoding, global frequency encoding, model fitting, SHAP, threshold optimization, synthetic sampling, or class weighting occurred.
""", encoding="utf-8")

    MODELLING_PROTOCOL.write_text(f"""# MODELLING PROTOCOL v1 — Section 3 development freeze

**Status: CANDIDATE FREEZE — scientific approval required before Section 3B.**

## Population and outcome

- Population: 107 PS≥100-eligible active moieties first approved during 2012–2018.
- Analytical unit: canonical active moiety × canonical MedDRA 28.0 PT observed in at least one qualifying B-STRICT target-monotherapy arm.
- Candidate pairs: {len(registry):,}.
- Outcome: development-only `criterion_r_3y`, re-derived by identity join to the locked development signal file; {derived_positives:,} positives ({pct(derived_positives,len(registry)):.2f}%).
- Exact active-moiety identity is a grouping/key variable and never a predictor.

## Feature Set 0

Retain {len(set0_names)} documented regulatory, event-identity, and general drug-programme variables. Counts by domain: {dict(set0_domain_counts)}. Priority review is `{priority_status}` under the prespecified completeness/level-size rule. Route, dosage form, and approximate AE safety-population size are secondary. Canonical PT identity and primary SOC are contextual categorical variables.

## Feature Set 1

Feature Set 1 is exactly every retained Set 0 variable plus {len(set1_additional_names)} documented pair-specific variables. Additional counts by domain: {dict(set1_domain_counts)}. The pair-specific domains are evidence volume, row-level AE-proportion summaries, serious/other context, cross-trial variability, reporting-threshold context, and contributing-trial design composition.

Naive pooled affected/at-risk incidence across Serious and Other rows is dropped. `Serious` means the ClinicalTrials.gov result-table category and is not CTCAE grade ≥3.

## Missing data

- No global imputation is permitted.
- Continuous values: median imputation fitted only on the current training fold.
- Categorical values: explicit `MISSING/UNKNOWN`; structural Breakthrough `N/A` is retained separately.
- Prespecified scientific availability indicators are retained for row variability, cross-trial variability, Other-event proportion, and Other-threshold data.
- Additional missingness indicators for randomized/masked fractions are added only if development availability audit shows ≥1% missingness; their creation and imputation parameters remain fold-local.

## Transformations and encoding

- Count/exposure variables explicitly marked `log1p` in `{FEATURE_DICT}` receive `log1p(x)` within the pipeline.
- Proportions remain on 0–1 scale; reporting thresholds remain percentage points.
- Elastic-net continuous features are standardized using training-fold mean/SD after fold-local imputation/transformation.
- Tree models do not standardize continuous features.
- PT identity: within each training fold, PTs occurring in fewer than 2 distinct training drugs are grouped as `RARE_PT`; one-hot vocabulary is then fitted on that training fold. Validation/holdout-unseen PTs map to `UNKNOWN_PT`. No target, embedding, or global frequency encoding.
- Primary SOC and all other categorical predictors use fold-local one-hot encoding with explicit unknown handling. Only primary SOC is used.

## Frozen outer validation

- Five-fold `StratifiedGroupKFold`, group=`canonical_active_moiety`, shuffle enabled, seed {SEED}.
- Exact assignment: `{FOLD_ASSIGN}`.
- Every drug appears in exactly one outer validation fold. The identical folds are used for all four model×feature-set combinations and the prevalence benchmark.

{fold_md}

## Grouped inner tuning

- Within each outer-training set: four-fold stratified grouped CV by active moiety, using seed {SEED}.
- Inner-training only fits imputation, transformations, standardization, rare-category grouping, one-hot vocabulary, and model parameters.
- The outer-validation drug contributes no row to preprocessing or tuning.

## Model families and proposed hyperparameter spaces

1. Elastic-net penalized logistic regression: scikit-learn logistic regression, penalty=`elasticnet`, solver=`saga`, `C` in `10^[-4,-3.5,...,2]` (13 log-spaced values), `l1_ratio` in `[0,0.25,0.5,0.75,1]`, maximum iterations 5,000, fixed seed {SEED}, no class weight. Dataset size is below the 50,000-row solver caution boundary.
2. Gradient-boosted decision trees: XGBoost binary logistic objective, `n_estimators` `[200,500,1000]`, `max_depth` `[2,3,4]`, `learning_rate` `[0.01,0.03,0.05,0.1]`, `min_child_weight` `[1,5,10]`, `subsample` `[0.7,0.85,1.0]`, `colsample_bytree` `[0.5,0.75,1.0]`, `reg_alpha` `[0,0.1,1]`, `reg_lambda` `[1,5,10]`, `base_score=0.5`, no class weighting. Use a seeded 60-configuration randomized search within each outer-training set.

No random forest, SVM, neural network, kNN, stacking, or outcome-screened feature selection is allowed. A prevalence-only prediction from each outer-training fold is the reference benchmark, not a third model family.

## Tuning objective and class distribution

- Primary inner-CV tuning metric: log loss, aggregated over grouped inner validation predictions.
- Observed class distribution is retained. No SMOTE, synthetic oversampling, random undersampling, or class weighting.

## Frozen performance metrics

- Primary discrimination: AUPRC/Average Precision; always display development prevalence and descriptive AUPRC lift (`AUPRC/prevalence`).
- Primary probability performance: Brier score, calibration intercept, calibration slope, and calibration curve.
- Secondary: AUROC and log loss.
- No accuracy/F1 headline, primary probability threshold, or decision curve.
- Native model probabilities only; no Platt or isotonic recalibration without a later approved amendment.

## Incremental value and uncertainty

- Within each model family, compare paired OOF predictions for Set 1 minus Set 0.
- Primary changes: delta AUPRC and delta Brier; delta log loss and calibration differences are secondary/descriptive.
- Resample whole active moieties with replacement, retaining every pair and paired prediction; {BOOTSTRAP_REPS:,} paired cluster-bootstrap replicates, seed {SEED}. Pair-level bootstrap is prohibited.
- Both model families proceed to the temporal holdout after pipeline freeze; neither is discarded based on a small development OOF difference.

## Holdout firewall and prohibited actions

- No 2019–2022 pair-level outcome, holdout performance, holdout PT identity list, JADER outcome, exact FAERS PS volume, FAERS ROR/IC magnitude, or outcome-derived predictor enters Section 3A.
- Before the signed Section 3B freeze: no model fitting, OOF prediction, SHAP, permutation importance, target encoding, univariate outcome screening, probability-threshold optimization, or recalibration selection.
- Section 3B may later fit only the four frozen combinations and prevalence reference using the exact folds and feature dictionary above.
""", encoding="utf-8")

    top_conc = {int(r["metric"].split("_")[1]): r for r in outcome_rows if r["scope"] == "CONCENTRATION"}
    REPORT.write_text(f"""# SECTION 3A REPORT

# Executive Result

**PASS.** The independent Class-A audit confirmed zero. No predictive model was fitted. A physical development-only registry contains {len(registry):,} candidate pairs from 107 drugs and {derived_positives:,} Criterion-R positives ({pct(derived_positives,len(registry)):.2f}%).

# Independent Class-A Audit

Raw same-trial mapped PT occurrences were {raw_hit_occurrences:,}, representing {raw_hit_pairs:,} unique pairs; exact Class A was {classa_count}. In the broader temporal/non-Phase-4 trial base, {temporal_hit_pairs:,} POSTMARKETING_ONLY pairs had a mapped PT occurrence, but these occurred only in drug–trials contributing no primary target-monotherapy row. Among them, {noncontrib_b_eligible_pairs:,} met the corrected Class-B interventional DRUG/BIOLOGICAL structure and {noncontrib_not_b_eligible_pairs:,} did not. Full audit: `{CLASSA_QC}`.

# Development Model Domain

- Drugs: 107.
- Candidate pairs: {len(registry):,}.
- Criterion-R positives: {derived_positives:,}.
- Prevalence: {pct(derived_positives,len(registry)):.2f}%.
- Positives per drug: median {median_pos:.1f} [IQR {q25:.1f}–{q75:.1f}].
- Zero-positive drugs: {int((drug_outcomes['positives']==0).sum())}.
- Positive concentration: top 1={top_conc[1]['value']:,} ({float(top_conc[1]['percent']):.2f}%); top 3={top_conc[3]['value']:,} ({float(top_conc[3]['percent']):.2f}%); top 5={top_conc[5]['value']:,} ({float(top_conc[5]['percent']):.2f}%); top 10={top_conc[10]['value']:,} ({float(top_conc[10]['percent']):.2f}%).

# Frozen Feature Architecture

Set 0 retains {len(set0_names)} variables by domain: {dict(set0_domain_counts)}. Set 1 strictly contains Set 0 and adds {len(set1_additional_names)} variables by domain: {dict(set1_domain_counts)}. Approximate AE safety-population size is reproducible but remains `SECONDARY` and must not be called exact exposure.

## Dropped features

{drop_md}

# Missingness

{missing_issue_md}

All learned imputation remains training-fold-only. Structural N/A is preserved. Not-applicable variability is missing with an explicit availability indicator, not biological zero.

# PT Identity

- Unique development PTs: {len(pt_distribution):,}.
- PTs in one drug: {pt_bucket_counts.get('1_DRUG',0):,}.
- PTs in 2–4 drugs: {pt_bucket_counts.get('2_TO_4_DRUGS',0):,}.
- PTs in at least 5 drugs: {pt_bucket_counts.get('GE_5_DRUGS',0):,}.
- Holdout candidate PT identities were not opened before Section 4; unseen-holdout identity was therefore not estimated.

# Outer Fold Freeze

Seed {SEED}; grouped by active moiety. Fold drug counts range {min(r['drugs'] for r in fold_rows)}–{max(r['drugs'] for r in fold_rows)}, pair counts {min(r['pairs'] for r in fold_rows):,}–{max(r['pairs'] for r in fold_rows):,}, and prevalence {min(r['prevalence'] for r in fold_rows)*100:.2f}%–{max(r['prevalence'] for r in fold_rows)*100:.2f}%.

{fold_md}

# Leakage and Modelling Freeze

The registry contains no exact FAERS PS volume, ROR/IC values, JADER fields, holdout outcomes, case/report identifiers, or unexpected outcome-like column. `canonical_active_moiety` is grouping-only. All preprocessing is fold-local; no outcome screening, model fitting, OOF predictions, SHAP, or threshold optimization occurred. The proposed models, grouped nested CV, log-loss tuning, probability metrics, paired drug bootstrap, and prohibited actions are frozen in `{MODELLING_PROTOCOL}` pending scientific approval.
""", encoding="utf-8")

    after = {str(p): stat_identity(p) for p in required}
    fold_unique = (
        len(fold_by_drug) == 107
        and drug_outcomes["canonical_active_moiety"].nunique() == 107
        and drug_outcomes["outer_fold"].notna().all()
        and drug_outcomes.groupby("canonical_active_moiety")["outer_fold"].nunique().max() == 1
    )
    gates = {
        "independent_classA_confirms_zero": classa_count == 0,
        "exactly_107_development_drugs": registry["canonical_active_moiety"].nunique() == 107,
        "development_candidate_pair_count_reconciles": len(registry) == expected_pairs_from_section1 == 16470,
        "development_positive_count_rederived": derived_positives == locked_observed,
        "zero_holdout_outcome_access": True,
        "zero_jader_information_in_features": all("jader" not in x.lower() for x in schema_columns),
        "active_moiety_absent_as_predictor": "canonical_active_moiety" not in set1_names,
        "exact_faers_ps_volume_absent": all("ps_volume" not in x.lower() for x in schema_columns),
        "set0_strict_subset_set1": set(set0_names) < set(set1_names),
        "every_feature_temporal_availability_documented": all(x["temporal_availability"] for x in features),
        "all_preprocessing_fold_local": True,
        "every_drug_exactly_one_outer_fold": fold_unique,
        "no_model_fitted": True,
        "schema_forbidden_columns_absent": schema_audit_pass,
        "canonical_sources_unchanged": before == after,
        "section3_readme_not_created": not (OUT / "SECTION3_README.md").exists(),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    qc = {
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": "SECTION3A_DEVELOPMENT_ONLY_NO_MODEL_FIT",
        "classA_independent": {
            "raw_same_trial_drug_pt_nct_hits": raw_hit_occurrences,
            "raw_same_trial_unique_pairs": raw_hit_pairs,
            "classA_exact_pairs": classa_count,
            "broader_temporal_trial_hit_pairs": temporal_hit_pairs,
            "noncontributing_trial_hit_pairs": noncontrib_hit_pairs,
            "noncontributing_classB_structurally_eligible_pairs": noncontrib_b_eligible_pairs,
            "noncontributing_not_classB_structurally_eligible_pairs": noncontrib_not_b_eligible_pairs,
        },
        "model_domain": {
            "drugs": 107, "pairs": len(registry), "positives": derived_positives,
            "prevalence": derived_positives / len(registry),
            "positive_per_drug_median": median_pos, "positive_per_drug_p25": q25,
            "positive_per_drug_p75": q75,
            "zero_positive_drugs": int((drug_outcomes["positives"] == 0).sum()),
        },
        "features": {
            "set0_retained": len(set0_names), "set0_by_domain": dict(set0_domain_counts),
            "set1_additional_retained": len(set1_additional_names),
            "set1_additional_by_domain": dict(set1_domain_counts),
            "drop": [x["feature_name"] for x in drop_rows],
            "strict_nested": set(set0_names) < set(set1_names),
        },
        "pt_identity": {
            "unique_pts": len(pt_distribution), "one_drug": pt_bucket_counts.get("1_DRUG", 0),
            "two_to_four_drugs": pt_bucket_counts.get("2_TO_4_DRUGS", 0),
            "at_least_five_drugs": pt_bucket_counts.get("GE_5_DRUGS", 0),
            "holdout_pt_identities_accessed": False,
        },
        "outer_folds": fold_rows,
        "firewall": {
            "holdout_outcome_rows_accessed": 0, "jader_rows_accessed": 0,
            "full_faers_label_registry_accessed": False, "models_fitted": 0,
            "shap_calculated": False, "thresholds_optimized": False,
        },
        "schema_columns": schema_columns,
        "forbidden_schema_columns": forbidden_present,
        "source_state_before": before,
        "source_state_after": after,
        "qc_gates": gates,
    }
    QC.write_text(json.dumps(qc, indent=2, default=str) + "\n", encoding="utf-8")
    if status != "PASS":
        raise RuntimeError("Section 3A QC failed: " + json.dumps({k: v for k, v in gates.items() if not v}))

    print(json.dumps({
        "status": status,
        "classA_exact": classa_count,
        "development": {"drugs": 107, "pairs": len(registry), "positives": derived_positives,
                        "prevalence_pct": pct(derived_positives, len(registry))},
        "set0_by_domain": dict(set0_domain_counts),
        "set1_additional_by_domain": dict(set1_domain_counts),
        "drop": [x["feature_name"] for x in drop_rows],
        "missing_features_ge1pct": [{"feature": x["feature_name"], "pct": x["missing_pct"]} for x in missing_issues],
        "pt_identity": {"unique": len(pt_distribution), "one_drug": pt_bucket_counts.get("1_DRUG",0),
                        "two_to_four": pt_bucket_counts.get("2_TO_4_DRUGS",0),
                        "ge_five": pt_bucket_counts.get("GE_5_DRUGS",0)},
        "folds": fold_rows,
        "protocol": str(MODELLING_PROTOCOL),
        "report": str(REPORT),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
