#!/usr/bin/env python3
"""Command 11 Phase A: outcome-free temporal feature construction and scoring.

No FAERS label/signal/JADER path is defined in this program. Predictions are
hashed and logged before Phase B is permitted to run.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

import s03c_preholdout_lock as prelock


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "section4_holdout"
AACT_DB = Path("/path/to/Database/AACT/aact.duckdb")
SECTION1 = ROOT / "analysis" / "section1_cohort" / "02_final_drug_level_characteristics.csv"
IDENTITY = ROOT / "preflight_v2" / "drug_identity_master.csv"
LINKS = ROOT / "data" / "processed" / "preflight_v2" / "aact_fda_intervention_links.parquet"
BSTRICT = ROOT / "preflight_v2" / "bstrict_candidate_registry.parquet"
AACT_PT_MAP = ROOT / "preflight_v2" / "faers_pt_repair" / "aact_meddra28_term_mapping.csv"
MEDDRA_MDHIER = Path("/path/to/Database/MedDRA/MedDRA_28_0_ENglish/MedAscii/mdhier.asc")
DEV_REGISTRY = ROOT / "analysis" / "section3_model" / "development_pair_registry.parquet"
FEATURE_DICTIONARY = ROOT / "analysis" / "section3_model" / "FEATURE_DICTIONARY_v1.csv"
MANIFEST = ROOT / "analysis" / "section3_model" / "PREHOLDOUT_LOCK_MANIFEST.json"

VERIFY_MD = OUT / "00_prehholdout_manifest_verification.md"
FEATURE_REGISTRY = OUT / "01_holdout_feature_registry.parquet"
PT_SUPPORT = OUT / "02_holdout_pt_support_preoutcome.csv"
PREDICTIONS = OUT / "03_holdout_predictions_PREOUTCOME.parquet"
PREDICTIONS_SHA = OUT / "03_holdout_predictions_PREOUTCOME.sha256"
OPENING_LOG = OUT / "HOLDOUT_OPENING_LOG.md"

GROUP = "canonical_active_moiety"
PT = "canonical_pt_code"
PIPELINES = ["elasticnet_set0", "elasticnet_set1", "xgboost_set0", "xgboost_set1"]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def apply_frozen_pt_mapper(frame: pd.DataFrame, retained: set[str]) -> pd.DataFrame:
    out = frame.copy()
    code = out[PT].astype(str)
    out[PT] = np.where(code.isin(retained), code, "PT_RARE")
    return out


def main() -> None:
    # One official Phase-A prediction freeze. Never overwrite partial or prior opening artefacts.
    protected = [FEATURE_REGISTRY, PT_SUPPORT, PREDICTIONS, PREDICTIONS_SHA, OPENING_LOG]
    if any(path.exists() for path in protected):
        raise FileExistsError("Section 4 Phase-A artefact already exists; official predictions will not be regenerated")
    OUT.mkdir(parents=True, exist_ok=True)

    for path in [AACT_DB, SECTION1, IDENTITY, LINKS, BSTRICT, AACT_PT_MAP, MEDDRA_MDHIER, DEV_REGISTRY, FEATURE_DICTIONARY, MANIFEST]:
        if not path.exists():
            raise FileNotFoundError(path)

    # Full pre-holdout verification occurs before any holdout feature identity is opened.
    prelock.verify_lock()
    source = prelock.load_sources()
    prelock.validate_locked_science(source)
    locks = prelock.build_pipeline_locks(source)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["temporal_holdout_scored"] or manifest["jader_outcome_accessed"]:
        raise RuntimeError("Pre-holdout manifest firewall is not sealed")
    for name in PIPELINES:
        frozen_versions = locks[name]["package_versions"]
        runtime = {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        }
        for key in ["python", "scikit_learn", "xgboost", "joblib", "numpy", "pandas"]:
            if str(runtime[key]) != str(frozen_versions[key]):
                raise RuntimeError(f"Package mismatch for {name}: {key} {runtime[key]} != {frozen_versions[key]}")

    verify_lines = [
        "# Pre-holdout manifest verification", "", f"**Verified:** {now()}", "",
        "**Result: PASS — all four frozen model, feature dictionary, PT mapper, and preprocessing hashes matched before holdout feature construction.**", "",
        "| Pipeline | Model SHA256 | PT mapper SHA256 | Preprocessing SHA256 |", "|---|---|---|---|",
    ]
    for name in PIPELINES:
        lock = locks[name]
        verify_lines.append(f"| {name} | `{lock['model_sha256']}` | `{lock['pt_mapper']['sha256']}` | `{lock['preprocessing']['sha256']}` |")
    verify_lines += ["", f"Feature dictionary SHA256: `{locks[PIPELINES[0]]['feature_dictionary_sha256']}`.", "", "No FAERS outcome or JADER source was opened during this verification."]
    VERIFY_MD.write_text("\n".join(verify_lines) + "\n", encoding="utf-8")

    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='20GB'; SET temp_directory='/private/tmp/pds_s04a_duckdb'; SET preserve_insertion_order=false; SET enable_progress_bar=false")
    con.execute(f"ATTACH '{AACT_DB}' AS a (READ_ONLY)")

    # Structural reconciliation is outcome-free and must complete before any model prediction.
    con.execute(
        f"""CREATE TABLE hold_drugs AS
            SELECT canonical_active_moiety,approval_year,nda_bla,orphan_designation,
                   accelerated_approval,breakthrough_therapy_designation,fast_track_designation,
                   priority_review,route,dosage_form,qualifying_trials,target_arms,
                   total_target_arm_subjects_at_risk,randomized_trial_fraction,
                   masked_trial_fraction,industry_sponsored_fraction,phase1_fraction,candidate_pairs
            FROM read_csv_auto('{SECTION1}',header=true)
            WHERE temporal_partition='TEMPORAL_HOLDOUT' AND approval_year BETWEEN 2019 AND 2022"""
    )
    structural = con.execute("SELECT count(*),count(DISTINCT canonical_active_moiety),sum(candidate_pairs),min(approval_year),max(approval_year) FROM hold_drugs").fetchone()
    if structural != (59, 59, 9681, 2019, 2022):
        raise RuntimeError(f"STOP BEFORE OUTCOME OPENING — holdout structure mismatch: {structural}")

    con.execute(
        f"""CREATE TABLE hold_identity AS
            SELECT d.canonical_active_moiety,cast(i.fda_first_approval_date AS DATE) approval_date
            FROM hold_drugs d JOIN read_csv_auto('{IDENTITY}',header=true) i USING(canonical_active_moiety)"""
    )
    con.execute(f"CREATE TABLE hold_links AS SELECT l.* FROM read_parquet('{LINKS}') l JOIN hold_drugs d USING(canonical_active_moiety)")
    con.execute(
        f"""CREATE TABLE pt_map AS
            SELECT aact_term_raw,cast(canonical_pt_code AS BIGINT) canonical_pt_code,
                   canonical_pt_name,mapping_level
            FROM read_csv_auto('{AACT_PT_MAP}',header=true)
            WHERE mapping_status='MAPPED' AND canonical_pt_code IS NOT NULL"""
    )
    con.execute(
        f"""CREATE TABLE pt_identity AS
            SELECT cast(column00 AS BIGINT) canonical_pt_code,any_value(column04) canonical_pt_name,
                   any_value(column07) primary_soc,count(DISTINCT column07) n_primary_socs
            FROM read_csv('{MEDDRA_MDHIER}',delim='$',header=false,all_varchar=true)
            WHERE column11='Y' AND column00 IS NOT NULL GROUP BY 1"""
    )
    if con.execute("SELECT count(*) FROM pt_identity WHERE n_primary_socs<>1").fetchone()[0]:
        raise RuntimeError("MedDRA primary SOC identity is not unique")

    con.execute("CREATE TABLE linked_drug_trials AS SELECT DISTINCT d.canonical_active_moiety,d.approval_date,l.nct_id FROM hold_identity d JOIN hold_links l USING(canonical_active_moiety)")
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
            WITH rg AS (SELECT nct_id,id rgid,{title_norm.format(col='title')} title_norm FROM a.result_groups WHERE result_type='Reported Event' AND nct_id IN (SELECT nct_id FROM linked_drug_trials)),
                 dg AS (SELECT nct_id,id dgid,group_type,title,{title_norm.format(col='title')} title_norm FROM a.design_groups WHERE nct_id IN (SELECT nct_id FROM linked_drug_trials))
            SELECT rg.nct_id,rg.rgid,count(DISTINCT dg.dgid) n_design_matches,min(dg.dgid) dgid,
                   min(dg.group_type) group_type,min(dg.title) design_group_title
            FROM rg LEFT JOIN dg ON rg.nct_id=dg.nct_id AND rg.title_norm=dg.title_norm GROUP BY 1,2"""
    )
    con.execute(
        """CREATE TABLE arm_drug_count AS
           SELECT dg.id dgid,count(DISTINCT i.id) FILTER (WHERE i.intervention_type IN ('DRUG','BIOLOGICAL')) n_drug_interventions
           FROM a.design_groups dg LEFT JOIN a.design_group_interventions dgi ON dgi.design_group_id=dg.id
           LEFT JOIN a.interventions i ON i.id=dgi.intervention_id
           WHERE dg.nct_id IN (SELECT nct_id FROM linked_drug_trials) GROUP BY dg.id"""
    )
    con.execute(
        """CREATE TABLE target_design_arms AS
           SELECT DISTINCT l.canonical_active_moiety,l.nct_id,dgi.design_group_id dgid,
                  l.aact_primary_name target_intervention_name,l.aact_intervention_combination_flag
           FROM hold_links l JOIN a.design_group_interventions dgi ON dgi.intervention_id=l.intervention_id"""
    )
    con.execute(
        """CREATE TABLE primary_rows AS
           SELECT v.*,m.dgid,m.group_type,m.design_group_title,t.target_intervention_name
           FROM valid_mapped_ae v
           JOIN result_to_design m ON v.nct_id=m.nct_id AND v.rgid=m.rgid AND m.n_design_matches=1
           JOIN arm_drug_count ac ON m.dgid=ac.dgid AND ac.n_drug_interventions=1
           JOIN target_design_arms t ON v.canonical_active_moiety=t.canonical_active_moiety AND v.nct_id=t.nct_id AND m.dgid=t.dgid
           WHERE m.group_type IN ('EXPERIMENTAL','OTHER') AND NOT t.aact_intervention_combination_flag
             AND NOT regexp_matches(lower(coalesce(t.target_intervention_name,'')), '(^| )(placebo|inactive placebo|vehicle|sham|dummy|sugar pill|no treatment|control|comparator|saline)( |$)')
             AND NOT regexp_matches(lower(coalesce(m.design_group_title,'')), '(^| )(placebo|vehicle|sham|no treatment|control)( |$)')"""
    )
    profile = con.execute("SELECT count(DISTINCT canonical_active_moiety),count(DISTINCT nct_id),count(DISTINCT rgid),count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM primary_rows").fetchone()
    if profile != (59, 173, 343, 9681):
        raise RuntimeError(f"STOP BEFORE OUTCOME OPENING — B-STRICT reconstruction mismatch: {profile}")

    con.execute(
        f"""CREATE TABLE hold_candidate_base AS
            SELECT r.canonical_active_moiety,cast(r.canonical_pt_code AS BIGINT) canonical_pt_code,p.canonical_pt_name,p.primary_soc
            FROM read_parquet('{BSTRICT}') r JOIN hold_drugs d USING(canonical_active_moiety)
            JOIN pt_identity p ON cast(r.canonical_pt_code AS BIGINT)=p.canonical_pt_code GROUP BY 1,2,3,4"""
    )
    candidate_qc = con.execute("SELECT count(*),count(DISTINCT canonical_active_moiety),count(DISTINCT (canonical_active_moiety,canonical_pt_code)) FROM hold_candidate_base").fetchone()
    if candidate_qc != (9681, 59, 9681):
        raise RuntimeError(f"STOP BEFORE OUTCOME OPENING — candidate mismatch: {candidate_qc}")

    con.execute(
        """CREATE TABLE hold_drug_features AS
           SELECT canonical_active_moiety,approval_year,nda_bla,
             CASE WHEN lower(cast(orphan_designation AS VARCHAR)) LIKE 'yes%' OR lower(cast(orphan_designation AS VARCHAR))='true' THEN 'Yes' WHEN lower(cast(orphan_designation AS VARCHAR)) IN ('no','false') THEN 'No' ELSE 'MISSING/UNKNOWN' END orphan_designation,
             CASE WHEN lower(cast(accelerated_approval AS VARCHAR)) LIKE 'yes%' OR lower(cast(accelerated_approval AS VARCHAR))='true' THEN 'Yes' WHEN lower(cast(accelerated_approval AS VARCHAR)) IN ('no','false') THEN 'No' ELSE 'MISSING/UNKNOWN' END accelerated_approval,
             CASE WHEN lower(cast(breakthrough_therapy_designation AS VARCHAR)) LIKE 'yes%' THEN 'Yes' WHEN cast(breakthrough_therapy_designation AS VARCHAR)='N/A' THEN 'N/A' WHEN lower(cast(breakthrough_therapy_designation AS VARCHAR))='no' THEN 'No' ELSE 'MISSING/UNKNOWN' END breakthrough_therapy_designation,
             CASE WHEN lower(cast(fast_track_designation AS VARCHAR)) LIKE 'yes%' THEN 'Yes' WHEN lower(cast(fast_track_designation AS VARCHAR))='no' THEN 'No' ELSE 'MISSING/UNKNOWN' END fast_track_designation,
             CASE WHEN lower(cast(priority_review AS VARCHAR)) LIKE 'priority%' THEN 'Priority' WHEN lower(cast(priority_review AS VARCHAR))='standard' THEN 'Standard' ELSE 'MISSING/UNKNOWN' END priority_review_category,
             CASE WHEN route IS NULL OR trim(route)='' THEN 'MISSING/UNKNOWN' WHEN strpos(route,'|')>0 THEN 'MULTIPLE' WHEN lower(route) LIKE '%oral%' THEN 'ORAL' WHEN lower(route) LIKE '%inject%' OR lower(route) LIKE '%intravenous%' OR lower(route) LIKE '%subcutaneous%' OR lower(route) LIKE '%intrathecal%' THEN 'PARENTERAL' WHEN lower(route) LIKE '%ophthalmic%' THEN 'OPHTHALMIC' WHEN lower(route) LIKE '%inhal%' THEN 'INHALATION' ELSE 'OTHER' END route_broad,
             CASE WHEN dosage_form IS NULL OR trim(dosage_form)='' THEN 'MISSING/UNKNOWN' WHEN strpos(dosage_form,'|')>0 THEN 'MULTIPLE' WHEN lower(dosage_form) LIKE '%inject%' THEN 'INJECTABLE' WHEN lower(dosage_form) LIKE '%solution%' OR lower(dosage_form) LIKE '%suspension%' THEN 'LIQUID' WHEN lower(dosage_form) LIKE '%tablet%' OR lower(dosage_form) LIKE '%capsule%' OR lower(dosage_form) LIKE '%granule%' OR lower(dosage_form) LIKE '%powder%' THEN 'SOLID' ELSE 'OTHER' END dosage_form_broad,
             qualifying_trials::BIGINT drug_n_qualifying_trials,target_arms::BIGINT drug_n_target_arms,
             total_target_arm_subjects_at_risk::DOUBLE drug_approx_ae_safety_population,
             phase1_fraction::DOUBLE drug_phase1_fraction,randomized_trial_fraction::DOUBLE drug_randomized_trial_fraction,
             masked_trial_fraction::DOUBLE drug_masked_trial_fraction,industry_sponsored_fraction::DOUBLE drug_industry_sponsored_fraction
           FROM hold_drugs"""
    )
    con.execute("CREATE TABLE design_agg AS SELECT nct_id,any_value(allocation) allocation,any_value(masking) masking FROM a.designs GROUP BY nct_id")
    con.execute(
        """CREATE TABLE pair_rows AS
           SELECT DISTINCT canonical_active_moiety,canonical_pt_code,nct_id,rgid,dgid,ae_id,event_type,frequency_threshold,
                  subjects_affected::DOUBLE subjects_affected,subjects_at_risk::DOUBLE subjects_at_risk,
                  subjects_affected::DOUBLE/subjects_at_risk::DOUBLE row_proportion FROM primary_rows"""
    )
    con.execute(
        """CREATE TABLE pair_arms AS
           SELECT canonical_active_moiety,canonical_pt_code,nct_id,rgid,dgid,max(subjects_at_risk) arm_subjects_at_risk,
                  max((event_type='serious')::INT) arm_has_serious,max((event_type='other')::INT) arm_has_other,
                  max(row_proportion) FILTER (WHERE event_type='serious') max_serious_proportion,
                  max(row_proportion) FILTER (WHERE event_type='other') max_other_proportion
           FROM pair_rows GROUP BY 1,2,3,4,5"""
    )
    con.execute(
        """CREATE TABLE pair_trials AS
           SELECT r.canonical_active_moiety,r.canonical_pt_code,r.nct_id,median(r.row_proportion) trial_median_proportion,
                  max((r.event_type='serious')::INT) trial_has_serious,
                  max((r.event_type='other' AND r.frequency_threshold=0)::INT) trial_threshold_0,
                  max((r.event_type='other' AND r.frequency_threshold=5)::INT) trial_threshold_5,
                  max((r.event_type='other' AND r.frequency_threshold IS NOT NULL)::INT) trial_has_other_threshold,
                  max(coalesce((s.phase='PHASE1')::INT,0)) phase1,
                  max(CASE WHEN d.allocation IS NULL THEN NULL ELSE (d.allocation='RANDOMIZED')::INT END) randomized,
                  max(CASE WHEN d.masking IS NULL THEN NULL ELSE (d.masking<>'NONE')::INT END) masked
           FROM pair_rows r JOIN a.studies s USING(nct_id) LEFT JOIN design_agg d USING(nct_id) GROUP BY 1,2,3"""
    )
    con.execute(
        """CREATE TABLE pair_feature_agg AS
           WITH row_stats AS (
             SELECT canonical_active_moiety,canonical_pt_code,count(*) n_rows,median(row_proportion) pair_median_row_ae_proportion,
                    max(row_proportion) pair_max_row_ae_proportion,
                    CASE WHEN count(*)>=2 THEN quantile_cont(row_proportion,.75)-quantile_cont(row_proportion,.25) END pair_row_ae_proportion_iqr,
                    max(row_proportion) FILTER (WHERE event_type='serious') pair_max_serious_row_proportion,
                    max(row_proportion) FILTER (WHERE event_type='other') pair_max_other_row_proportion,
                    min(frequency_threshold) FILTER (WHERE event_type='other' AND frequency_threshold IS NOT NULL) pair_min_other_threshold,
                    max(frequency_threshold) FILTER (WHERE event_type='other' AND frequency_threshold IS NOT NULL) pair_max_other_threshold,
                    median(frequency_threshold) FILTER (WHERE event_type='other' AND frequency_threshold IS NOT NULL) pair_median_other_threshold
             FROM pair_rows GROUP BY 1,2),
           arm_stats AS (
             SELECT canonical_active_moiety,canonical_pt_code,count(*) pair_n_reporting_arms,sum(arm_subjects_at_risk) pair_nonduplicated_arm_subjects_at_risk,
                    max(arm_has_serious) pair_any_serious,sum(arm_has_serious) pair_n_serious_arms,avg(arm_has_serious) pair_serious_arm_fraction,
                    max(arm_has_other) pair_other_proportion_available FROM pair_arms GROUP BY 1,2),
           trial_stats AS (
             SELECT canonical_active_moiety,canonical_pt_code,count(*) pair_n_reporting_trials,sum(trial_has_serious) pair_n_serious_trials,
                    avg(trial_has_serious) pair_serious_trial_fraction,
                    CASE WHEN count(*)>=2 THEN stddev_samp(trial_median_proportion) END pair_between_trial_proportion_sd,
                    avg(trial_threshold_0) pair_fraction_trials_threshold_0,avg(trial_threshold_5) pair_fraction_trials_threshold_5,
                    max(trial_has_other_threshold) pair_other_threshold_available,avg(phase1) pair_phase1_fraction,
                    avg(randomized) FILTER (WHERE randomized IS NOT NULL) pair_randomized_trial_fraction,
                    avg(masked) FILTER (WHERE masked IS NOT NULL) pair_masked_trial_fraction FROM pair_trials GROUP BY 1,2)
           SELECT r.canonical_active_moiety,r.canonical_pt_code,t.pair_n_reporting_trials,
                  t.pair_n_reporting_trials::DOUBLE/d.drug_n_qualifying_trials pair_reporting_trial_fraction,
                  a.pair_n_reporting_arms,a.pair_nonduplicated_arm_subjects_at_risk,
                  r.pair_median_row_ae_proportion,r.pair_max_row_ae_proportion,r.pair_row_ae_proportion_iqr,
                  (r.n_rows>=2)::TINYINT pair_row_variability_available,a.pair_any_serious,t.pair_n_serious_trials,a.pair_n_serious_arms,
                  t.pair_serious_trial_fraction,a.pair_serious_arm_fraction,r.pair_max_serious_row_proportion,r.pair_max_other_row_proportion,
                  a.pair_other_proportion_available,t.pair_between_trial_proportion_sd,
                  (t.pair_n_reporting_trials>=2)::TINYINT pair_cross_trial_variability_available,
                  r.pair_min_other_threshold,r.pair_max_other_threshold,r.pair_median_other_threshold,
                  t.pair_fraction_trials_threshold_0,t.pair_fraction_trials_threshold_5,t.pair_other_threshold_available,
                  t.pair_phase1_fraction,t.pair_randomized_trial_fraction,t.pair_masked_trial_fraction
           FROM row_stats r JOIN arm_stats a USING(canonical_active_moiety,canonical_pt_code)
           JOIN trial_stats t USING(canonical_active_moiety,canonical_pt_code)
           JOIN hold_drug_features d USING(canonical_active_moiety)"""
    )
    con.execute(
        """CREATE TABLE feature_registry AS
           SELECT md5(c.canonical_active_moiety||'|'||cast(c.canonical_pt_code AS VARCHAR)) pair_id,
                  c.canonical_active_moiety,c.canonical_pt_code,c.canonical_pt_name,c.primary_soc,
                  d.* EXCLUDE(canonical_active_moiety),p.* EXCLUDE(canonical_active_moiety,canonical_pt_code)
           FROM hold_candidate_base c JOIN hold_drug_features d USING(canonical_active_moiety)
           JOIN pair_feature_agg p USING(canonical_active_moiety,canonical_pt_code)
           ORDER BY d.approval_year,c.canonical_active_moiety,c.canonical_pt_code"""
    )
    registry = con.execute("SELECT * FROM feature_registry").fetchdf()
    con.close()
    if (len(registry), registry[GROUP].nunique(), registry["pair_id"].nunique()) != (9681, 59, 9681):
        raise RuntimeError("Final outcome-free registry reconciliation failed")
    forbidden = [c for c in registry.columns if any(x in c.lower() for x in ["criterion", "ror", "faers", "consensus", "exact_ps", "jader", "outcome", "signal"])]
    if forbidden:
        raise RuntimeError(f"Outcome-free registry contains forbidden columns: {forbidden}")
    expected_features = pd.read_csv(FEATURE_DICTIONARY)
    expected_features = expected_features.loc[expected_features.status.isin(["PRIMARY", "SECONDARY"]) & expected_features.feature_set.isin(["SET0", "SET1_ADDITIONAL"]), "feature_name"].tolist()
    if not set(expected_features).issubset(registry.columns):
        raise RuntimeError(f"Missing frozen features: {set(expected_features)-set(registry.columns)}")
    registry.to_parquet(FEATURE_REGISTRY, index=False)

    # Frozen 884-category mapper and pre-outcome support classification.
    bundle0 = joblib.load(locks["elasticnet_set0"]["model_path"])
    retained = set(str(x) for x in bundle0["retained_pt_categories"])
    if len(retained) != 884:
        raise RuntimeError("Frozen PT mapper does not retain 884 categories")
    dev = pd.read_parquet(DEV_REGISTRY, columns=[GROUP, PT])
    dev_codes = set(dev[PT].astype(str))
    hold_codes = registry[PT].astype(str)
    supported = hold_codes.isin(retained)
    unseen = ~hold_codes.isin(dev_codes)
    registry_support = pd.DataFrame({
        "pair_id": registry["pair_id"], GROUP: registry[GROUP], PT: registry[PT],
        "pt_support_stratum": np.where(supported, "PT_SUPPORTED", "PT_RARE"),
        "pt_rare_subtype": np.where(supported, "NOT_APPLICABLE", np.where(unseen, "UNSEEN_IN_DEVELOPMENT", "SEEN_BUT_LOW_SUPPORT")),
    })
    support_rows = []
    definitions = [
        ("PT_SUPPORTED", supported), ("PT_RARE", ~supported),
        ("UNSEEN_IN_DEVELOPMENT", (~supported) & unseen),
        ("SEEN_BUT_LOW_SUPPORT", (~supported) & (~unseen)),
    ]
    for label, mask in definitions:
        support_rows.append({
            "support_category": label, "pair_count": int(mask.sum()), "pair_percent": float(100 * mask.mean()),
            "distinct_holdout_pts": int(registry.loc[mask, PT].nunique()), "drug_count": int(registry.loc[mask, GROUP].nunique()),
            "outcome_fields_present": False,
        })
    support_df = pd.DataFrame(support_rows)
    support_df["all_holdout_distinct_pts"] = registry[PT].nunique()
    support_df["completely_unseen_holdout_distinct_pts"] = registry.loc[unseen, PT].nunique()
    support_df.to_csv(PT_SUPPORT, index=False)

    # Native predictions from each frozen bundle; model artefacts are never modified.
    pred = registry[["pair_id", GROUP, PT, "canonical_pt_name"]].merge(registry_support, on=["pair_id", GROUP, PT], validate="one_to_one")
    prediction_ranges = {}
    for name in PIPELINES:
        bundle = joblib.load(locks[name]["model_path"])
        if set(str(x) for x in bundle["retained_pt_categories"]) != retained:
            raise RuntimeError(f"PT mapper differs across pipelines: {name}")
        mapped = apply_frozen_pt_mapper(registry, retained)
        X = bundle["preprocessor"].transform(mapped[bundle["features"]])
        values = bundle["model"].predict_proba(X)[:, 1]
        if len(values) != len(registry) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise RuntimeError(f"Invalid predictions: {name}")
        pred[name] = values
        prediction_ranges[name] = {"min": float(values.min()), "max": float(values.max()), "missing": int(pd.isna(values).sum())}
    if pred.duplicated([GROUP, PT]).any() or pred[list(PIPELINES)].isna().any().any():
        raise RuntimeError("Duplicate or missing pre-outcome predictions")
    pred.to_parquet(PREDICTIONS, index=False)
    pred_hash = sha256(PREDICTIONS)
    PREDICTIONS_SHA.write_text(f"{pred_hash}  {PREDICTIONS.name}\n", encoding="utf-8")
    freeze_time = now()
    OPENING_LOG.write_text(
        "# Holdout opening log\n\n"
        f"- Phase A manifest verification completed: {freeze_time}\n"
        f"- Outcome-free structure reconciled: 59 drugs / 9,681 pairs.\n"
        f"- Feature registry frozen: {FEATURE_REGISTRY}.\n"
        f"- PT support classification frozen before outcome access: {PT_SUPPORT}.\n"
        f"- Native prediction file frozen: {freeze_time}.\n"
        f"- Prediction SHA256: `{pred_hash}`.\n"
        f"- Rows: {len(pred):,}; drugs: {pred[GROUP].nunique():,}; duplicate keys: {int(pred.duplicated([GROUP,PT]).sum())}.\n"
        f"- Prediction diagnostics: `{json.dumps(prediction_ranges, sort_keys=True)}`.\n"
        "- Missing predictions across all four pipelines: 0.\n\n"
        "> **PREDICTIONS FROZEN BEFORE OUTCOME OPENING**\n\n"
        "No FAERS outcome or JADER object was loaded in Phase A.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PHASE_A_PASS_PREDICTIONS_FROZEN", "drugs": 59, "pairs": 9681,
        "pt_support": support_rows, "prediction_sha256": pred_hash,
        "prediction_ranges": prediction_ranges, "freeze_timestamp": freeze_time,
    }, indent=2))


if __name__ == "__main__":
    main()
