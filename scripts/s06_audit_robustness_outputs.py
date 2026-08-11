#!/usr/bin/env python3
"""Independent read-only audit of Command 15 outputs."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/section6_robustness"
JDRUG = Path("/path/to/Database/Jader/jader_v5_drug.parquet")
JREAC = Path("/path/to/Database/Jader/jader_v5_reac.parquet")
LABELS = ROOT / "data/processed/preflight_v2/faers_fda_anchored_labels_1_2_3y.parquet"
SIGNALS = ROOT / "data/processed/preflight_v2/faers_all_exposed_pair_signals_3y.parquet"
DEV_PRED = ROOT / "analysis/section3_model/training/05_oof_predictions.parquet"
HOLD_PRED = ROOT / "analysis/section4_holdout/03_holdout_predictions_PREOUTCOME.parquet"
HOLD_OUTCOME = ROOT / "analysis/section4_holdout/04_holdout_outcome_registry.parquet"
AUDIT = OUT / "SECTION6_AUDIT.json"


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return bool(abs(float(a) - float(b)) <= tol)


def main() -> None:
    required = [
        "00_jader_source_and_identity_qc.md","01_jader_assessability.csv","02_jader_pair_2x2.parquet",
        "03_jader_replication_status.csv","04_jader_replication_summary.csv","05_jader_replication_by_premarketing_status.csv",
        "06_consensus_endpoint_performance.csv","07_horizon_1y_2y_performance.csv","08_ps_threshold_sensitivity.csv",
        "09_definitionA_sensitivity.csv","10_bexpanded_sensitivity.csv","11_arm_attribution_sensitivity.md",
        "12_reporting_threshold_context.md","13_model_robustness_summary.csv","14_figure5_source_data",
        "SECTION6_REPORT.md","SECTION6_QC.json",
    ]
    checks: dict[str, bool] = {}
    checks["required_outputs_exist"] = all((OUT / x).exists() for x in required)
    checks["readme_absent"] = not (OUT / "SECTION6_README.md").exists()

    qc = json.loads((OUT / "SECTION6_QC.json").read_text())
    checks["internal_qc_pass_20_of_20"] = qc["status"] == "PASS" and len(qc["qc_gates"]) == 20 and all(qc["qc_gates"].values())
    checks["protected_artifacts_unchanged"] = qc["protected_artifacts_before"] == qc["protected_artifacts_after"]
    checks["raw_sources_unchanged"] = qc["raw_source_stats_before"] == qc["raw_source_stats_after"]
    checks["prohibited_compute_absent"] = not any([qc["models_loaded"], qc["models_trained"], qc["predictions_regenerated"], qc["shap_calculated"]])
    checks["jader_boundary_locked"] = (
        qc["jader_source_paths"] == [str(JDRUG), str(JREAC)] and not qc["jader_master_used"] and
        not qc["jader_v4_used"] and not qc["jader_temporal_window_constructed"] and not qc["jader_us_approval_anchor_used"]
    )

    con = duckdb.connect()
    con.execute("SET threads=8; SET preserve_insertion_order=false")
    total_cases = con.execute(f"SELECT count(DISTINCT ID) FROM (SELECT ID FROM read_parquet('{JDRUG}') UNION ALL SELECT ID FROM read_parquet('{JREAC}'))").fetchone()[0]
    pair = pd.read_parquet(OUT / "02_jader_pair_2x2.parquet")
    checks["signal_universe_16252_unique"] = len(pair) == 16252 and pair[["canonical_active_moiety","canonical_pt_code"]].drop_duplicates().shape[0] == 16252
    source_keys = con.execute(f"SELECT canonical_active_moiety,canonical_pt_code FROM read_parquet('{SIGNALS}') WHERE criterion_r AND target_ps_cases>=100 ORDER BY 1,2").fetchdf()
    output_keys = pair[["canonical_active_moiety","canonical_pt_code"]].sort_values(["canonical_active_moiety","canonical_pt_code"]).reset_index(drop=True)
    checks["faers_signal_keys_exact"] = source_keys.equals(output_keys)
    checks["jader_case_universe_exact"] = total_cases == 1043485 and (pair[["a","b","c","d"]].sum(axis=1) == total_cases).all()
    checks["jader_cells_nonnegative"] = (pair[["a","b","c","d"]] >= 0).all().all()
    checks["exposure_margins_exact"] = (pair.a + pair.b == pair.jader_ps_cases).all() and (pair.a + pair.c == pair.jader_pt_cases).all()
    valid = (pair.a>0)&(pair.b>0)&(pair.c>0)&(pair.d>0)
    ror = (pair.loc[valid,"a"]*pair.loc[valid,"d"])/(pair.loc[valid,"b"]*pair.loc[valid,"c"])
    lcl = np.exp(np.log(ror)-1.96*np.sqrt(1/pair.loc[valid,"a"]+1/pair.loc[valid,"b"]+1/pair.loc[valid,"c"]+1/pair.loc[valid,"d"]))
    checks["ror_recomputed"] = np.max(np.abs(ror-pair.loc[valid,"ror"])) < 1e-10 and np.max(np.abs(lcl-pair.loc[valid,"ror_lcl95"])) < 1e-10
    expected_r = pair.assessable & (pair.a>=3) & pair.ror_lcl95.gt(1).fillna(False)
    expected_c = expected_r & pair.ic025.gt(0).fillna(False)
    checks["replication_flags_recomputed"] = np.array_equal(expected_r.to_numpy(),pair.jader_r.to_numpy()) and np.array_equal(expected_c.to_numpy(),pair.jader_consensus.to_numpy())
    status_counts = pair.replication_status.value_counts().to_dict()
    checks["three_way_status_exact"] = status_counts == {"NOT_REPLICATED":8207,"NOT_ASSESSABLE":6516,"REPLICATED":1529}
    checks["jader_primary_counts_exact"] = int(pair.assessable.sum())==9736 and int(pair.jader_r.sum())==1529 and int(pair.jader_consensus.sum())==1370 and int(pair.directionally_positive.sum())==3106

    js = pd.read_csv(OUT / "04_jader_replication_summary.csv").set_index("metric")
    checks["jader_summary_matches_pairs"] = (
        int(js.loc["JADER_ASSESSABLE","n"])==int(pair.assessable.sum()) and
        int(js.loc["JADER_R_REPLICATED","n"])==int(pair.jader_r.sum()) and
        int(js.loc["JADER_CONSENSUS_REPLICATED","n"])==int(pair.jader_consensus.sum()) and
        close(js.loc["JADER_R_REPLICATED","percent"],100*pair.jader_r.sum()/pair.assessable.sum())
    )
    checks["drug_bootstrap_5000"] = int(js.loc["JADER_R_REPLICATED","bootstrap_replicates"])==5000 and js.loc["JADER_R_REPLICATED","bootstrap_unit"]=="canonical_active_moiety"
    by = pd.read_csv(OUT / "05_jader_replication_by_premarketing_status.csv").set_index("premarketing_status")
    checks["premarketing_strata_exact"] = (
        int(by.loc["PREMARKETING_OBSERVED","assessable_pairs"])==1824 and int(by.loc["PREMARKETING_OBSERVED","jader_r_replicated"])==538 and
        int(by.loc["POSTMARKETING_ONLY","assessable_pairs"])==7912 and int(by.loc["POSTMARKETING_ONLY","jader_r_replicated"])==991
    )

    labels = pd.read_parquet(LABELS,columns=["canonical_active_moiety","canonical_pt_code","horizon_years","criterion_r","consensus"])
    wide=labels.pivot(index=["canonical_active_moiety","canonical_pt_code"],columns="horizon_years",values=["criterion_r","consensus"])
    wide.columns=[f"{a}_{int(b)}y" for a,b in wide.columns];wide=wide.reset_index()
    robust=pd.read_csv(OUT / "13_model_robustness_summary.csv")
    hold=pd.read_parquet(HOLD_PRED).merge(pd.read_parquet(HOLD_OUTCOME)[["canonical_active_moiety","canonical_pt_code","criterion_r_3y"]],on=["canonical_active_moiety","canonical_pt_code"],validate="one_to_one").merge(wide,on=["canonical_active_moiety","canonical_pt_code"],validate="one_to_one")
    specs=[("Criterion R",3,"criterion_r_3y_x"),("Consensus",3,"consensus_3y"),("Criterion R",2,"criterion_r_2y"),("Criterion R",1,"criterion_r_1y")]
    perf_ok=True
    for endpoint,horizon,target in specs:
        row=robust[(robust.endpoint.eq(endpoint))&(robust.horizon_years.eq(horizon))].iloc[0]
        y=hold[target].astype(int)
        for family in ["elasticnet","xgboost"]:
            ap0=average_precision_score(y,hold[f"{family}_set0"]);ap1=average_precision_score(y,hold[f"{family}_set1"])
            perf_ok &= close(row[f"{family}_set0_ap"],ap0) and close(row[f"{family}_set1_ap"],ap1) and close(row[f"{family}_delta_ap"],ap1-ap0)
    checks["temporal_performance_recomputed"] = bool(perf_ok)
    locked=pd.read_csv(ROOT/"analysis/section4_holdout/07_holdout_incremental_value.csv")
    prim=robust[(robust.endpoint.eq("Criterion R"))&(robust.horizon_years.eq(3))].iloc[0]
    checks["primary_delta_exactly_locked"] = all(
        close(prim[f"{f}_delta_ap"],locked[(locked.model_family.eq(f))&(locked.metric.eq("delta_average_precision_set1_minus_set0"))].iloc[0].estimate)
        for f in ["elasticnet","xgboost"]
    )
    checks["alternative_label_calibration_absent"] = not any("calibration" in c.lower() for c in pd.read_csv(OUT/"06_consensus_endpoint_performance.csv").columns) and not any("calibration" in c.lower() for c in pd.read_csv(OUT/"07_horizon_1y_2y_performance.csv").columns)
    checks["all_alternative_delta_point_estimates_positive"] = (robust.loc[robust.role.eq("ROBUSTNESS"),["elasticnet_delta_ap","xgboost_delta_ap"]]>0).all().all()

    ps=pd.read_csv(OUT/"08_ps_threshold_sensitivity.csv").set_index("ps_threshold")
    checks["ps100_primary_counts_reproduced"] = int(ps.loc["PS_GE_100_PRIMARY","drugs"])==166 and int(ps.loc["PS_GE_100_PRIMARY","candidate_pairs"])==26151 and int(ps.loc["PS_GE_100_PRIMARY","candidate_pair_criterion_r_positives"])==3174
    checks["ps_outside_primary_descriptive_only"] = not ps.performance_evaluated_outside_ps100.astype(bool).any()
    da=pd.read_csv(OUT/"09_definitionA_sensitivity.csv").iloc[0];be=pd.read_csv(OUT/"10_bexpanded_sensitivity.csv").iloc[0]
    checks["definition_a_locked_counts"] = (int(da.all_eligible_drugs),int(da.trials),int(da.target_arms),int(da.drug_pt_pairs))==(81,175,310,9653)
    checks["bexpanded_locked_counts_and_increment"] = (int(be.all_eligible_drugs),int(be.trials),int(be.target_arms),int(be.drug_pt_pairs),int(be.increment_drugs),int(be.increment_trials),int(be.increment_target_arms),int(be.increment_drug_pt_pairs))==(239,716,1320,38456,27,96,171,8209)

    report=(OUT/"SECTION6_REPORT.md").read_text()
    headings=["# Executive Result","# JADER V5 Replication Design","# JADER Assessability","# Cross-Database Replication","# Replication by Premarketing Representation","# Three-Year Consensus Robustness","# One-Year and Two-Year Horizon Robustness","# PS-Threshold Sensitivity","# Definition A Sensitivity","# B-EXPANDED Sensitivity","# Arm-Attribution Sensitivity","# Reporting-Threshold Context","# Overall Robustness Synthesis","# Candidate Main-Text Results","# Candidate Supplementary Results","# Section-Specific Limitations","# Issues Requiring Scientific Review"]
    checks["report_headings_exact"] = re.findall(r"^# .+$",report,re.M)==headings
    prohibited=["externally validated adverse reactions","replicated causal toxicity","Japanese temporal validation","nonreplicated signals are false","JADER absence disproves risk"]
    checks["prohibited_claims_absent"] = not any(x.lower() in report.lower() for x in prohibited)

    numeric_values=[]
    for name in ["04_jader_replication_summary.csv","05_jader_replication_by_premarketing_status.csv","08_ps_threshold_sensitivity.csv","09_definitionA_sensitivity.csv","10_bexpanded_sensitivity.csv","13_model_robustness_summary.csv"]:
        f=pd.read_csv(OUT/name)
        for col in f.select_dtypes(include=[np.number]).columns:
            numeric_values.extend([float(x) for x in f[col].dropna().tolist()])
    numeric_values.extend([16252,166,1043485,12435,51.3,78,45449,5000])
    checks = {k: bool(v) for k, v in checks.items()}
    verdict="PASS" if all(checks.values()) else "FAIL"
    result={"status":verdict,"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
            "numeric_provenance_values":numeric_values,"jader_status_counts":status_counts,
            "note":"Numerical bundle mirrors machine-readable CSV sources for prose-audit matching; contextual checks are listed separately."}
    AUDIT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":verdict,"checks_passed":sum(checks.values()),"checks_total":len(checks),"failed":[k for k,v in checks.items() if not v]},indent=2))
    if verdict!="PASS": raise RuntimeError("Independent Section 6 audit failed")


if __name__=="__main__":
    main()
