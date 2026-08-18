#!/usr/bin/env python3
"""Summarize FAERS assessability, blind spots, and temporal split options."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/project")
OUT = PROJECT / "preflight_v2"
PROC = PROJECT / "data/processed/preflight_v2"
LABELS = PROC / "faers_fda_anchored_labels_1_2_3y.parquet"
ASSESS = PROC / "faers_drug_assessability_3y.csv"
CASE_DRUG = PROC / "faers_latest_case_drug_ps_fda.parquet"
CASE_PT = PROC / "faers_pt_repair/faers_latest_case_pt_meddra28.parquet"
LATEST = PROC / "faers_pt_repair/faers_latest_cases.parquet"
REGISTRY = OUT / "bstrict_candidate_registry.parquet"
MASTER = OUT / "drug_identity_master.csv"
PT_CANON = OUT / "faers_pt_repair/faers_meddra28_canonical.csv"
ALL_SIGNALS = PROC / "faers_all_exposed_pair_signals_3y.parquet"
METRICS = PROC / "assessability_coverage_split_metrics.json"


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def md_table(rows, fields):
    head = "| " + " | ".join(label for _, label in fields) + " |"
    sep = "|" + "|".join("---" for _ in fields) + "|"
    body = []
    for r in rows:
        vals = []
        for k, _ in fields:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:,.3f}"
            elif isinstance(v, int):
                v = f"{v:,}"
            vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep, *body])


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def concentration(values, k):
    total = sum(values)
    return pct(sum(sorted(values, reverse=True)[:k]), total)


def main() -> None:
    con = duckdb.connect()
    con.execute(
        "SET threads=8; SET memory_limit='20GB'; SET temp_directory='/private/tmp/pvml_duckdb'; "
        "SET preserve_insertion_order=false; SET enable_progress_bar=false"
    )
    con.execute(f"CREATE TABLE labels3 AS SELECT * FROM read_parquet('{LABELS}') WHERE horizon_years=3")
    con.execute(f"CREATE TABLE assess AS SELECT * FROM read_csv_auto('{ASSESS}',header=true)")

    assess_rows = []
    total_pos0 = {}
    for criterion, col in (("R","criterion_r"),("IC","criterion_ic"),("CONSENSUS","consensus")):
        total_pos0[criterion] = con.execute(f"SELECT sum({col}::INT) FROM labels3").fetchone()[0]
    for threshold in (0, 50, 100, 200, 500):
        for criterion, col in (("R","criterion_r"),("IC","criterion_ic"),("CONSENSUS","consensus")):
            drug = con.execute(
                f"""
                SELECT l.canonical_active_moiety, count(*) pairs, sum(l.{col}::INT) positives
                FROM labels3 l JOIN assess a USING(canonical_active_moiety)
                WHERE a.ps_reports_3y>={threshold} GROUP BY 1
                """
            ).fetchall()
            positives = [int(x[2]) for x in drug]
            n_pairs = sum(int(x[1]) for x in drug)
            n_pos = sum(positives)
            assess_rows.append({
                "ps_minimum": threshold, "signal_definition": criterion,
                "active_moieties": len(drug), "candidate_drug_pt_pairs": n_pairs,
                "positive_pairs": n_pos, "positive_prevalence_pct": pct(n_pos,n_pairs),
                "drugs_with_zero_positives": sum(x==0 for x in positives),
                "median_positive_pairs_per_drug": statistics.median(positives) if positives else 0,
                "top1_positive_concentration_pct": concentration(positives,1),
                "top3_positive_concentration_pct": concentration(positives,3),
                "top5_positive_concentration_pct": concentration(positives,5),
                "top10_positive_concentration_pct": concentration(positives,10),
                "all_positive_pairs_retained_pct": pct(n_pos,total_pos0[criterion]),
            })
    write_csv(OUT / "04_outcome_assessability.csv", assess_rows)
    rrows = [r for r in assess_rows if r["signal_definition"] == "R"]
    crows = [r for r in assess_rows if r["signal_definition"] == "CONSENSUS"]
    (OUT / "04_outcome_assessability.md").write_text(
        "# 04 — Three-year FAERS outcome assessability\n\n"
        "Counts use B-STRICT candidate drug–PT pairs and unique latest FAERS cases. "
        "Thresholds are descriptive; none is selected using model performance.\n\n"
        "## Criterion R\n\n" + md_table(rrows, [
            ("ps_minimum","Minimum PS"),("active_moieties","Drugs"),("candidate_drug_pt_pairs","Pairs"),
            ("positive_pairs","Positive"),("positive_prevalence_pct","Prevalence %"),
            ("drugs_with_zero_positives","Zero-positive drugs"),("median_positive_pairs_per_drug","Median positive/drug"),
            ("top3_positive_concentration_pct","Top-3 %"),("all_positive_pairs_retained_pct","All positives retained %")]) +
        "\n\n## Consensus\n\n" + md_table(crows, [
            ("ps_minimum","Minimum PS"),("active_moieties","Drugs"),("candidate_drug_pt_pairs","Pairs"),
            ("positive_pairs","Positive"),("positive_prevalence_pct","Prevalence %"),
            ("drugs_with_zero_positives","Zero-positive drugs"),("median_positive_pairs_per_drug","Median positive/drug"),
            ("top3_positive_concentration_pct","Top-3 %"),("all_positive_pairs_retained_pct","All positives retained %")]) +
        "\n\nThe full CSV also reports Criterion IC and top-1/5/10 concentration.\n",
        encoding="utf-8")

    # Build a separate all-exposed-pair signal universe. This is essential for
    # blind-spot analysis: restricting to the B-STRICT candidate registry would
    # mechanically classify every signal as premarketing observed.
    con.execute(
        f"""
        CREATE TABLE windows AS
        SELECT DISTINCT l.canonical_active_moiety, l.approval_date,
               (l.approval_date + INTERVAL 3 YEAR)::DATE window_end,
               l.approval_year, l.nda_bla, l.orphan_designation, l.accelerated_approval,
               a.ps_reports_3y target_ps_cases,
               max(l.total_cases) total_cases
        FROM labels3 l JOIN assess a USING(canonical_active_moiety)
        GROUP BY ALL
        """
    )
    con.execute(
        f"""
        CREATE TABLE all_a AS
        SELECT w.canonical_active_moiety, e.canonical_pt_code,
               count(DISTINCT d.caseid) a
        FROM windows w JOIN read_parquet('{CASE_DRUG}') d
          ON d.canonical_active_moiety=w.canonical_active_moiety
         AND d.report_date BETWEEN w.approval_date AND w.window_end
        JOIN read_parquet('{CASE_PT}') e ON e.caseid=d.caseid AND e.primaryid=d.primaryid
        GROUP BY 1,2 HAVING count(DISTINCT d.caseid)>=3
        """
    )
    con.execute(
        f"""
        CREATE TABLE pt_daily AS
        SELECT try_strptime(cast(fda_dt AS VARCHAR), '%Y%m%d')::DATE report_date,
               canonical_pt_code, count(*) n
        FROM read_parquet('{CASE_PT}') GROUP BY 1,2
        """
    )
    con.execute(
        """
        CREATE TABLE all_m AS
        SELECT a.canonical_active_moiety,a.canonical_pt_code,sum(d.n) total_pt_cases
        FROM all_a a JOIN windows w USING(canonical_active_moiety)
        JOIN pt_daily d ON d.canonical_pt_code=a.canonical_pt_code
                       AND d.report_date BETWEEN w.approval_date AND w.window_end
        GROUP BY 1,2
        """
    )
    con.execute(
        f"""
        COPY (
          WITH pt AS (
            SELECT cast(canonical_pt_code AS BIGINT) canonical_pt_code,
                   any_value(canonical_pt_name) canonical_pt_name,
                   any_value(canonical_soc_name) canonical_soc_name
            FROM read_csv('{PT_CANON}',header=true,all_varchar=true)
            WHERE mapping_status='MAPPED' GROUP BY 1
          ), pre AS (
            SELECT DISTINCT canonical_active_moiety,canonical_pt_code
            FROM read_parquet('{REGISTRY}')
          ), cells AS (
            SELECT w.*,a.canonical_pt_code,a.a,m.total_pt_cases,
                   w.target_ps_cases-a.a b, m.total_pt_cases-a.a c,
                   w.total_cases-w.target_ps_cases-m.total_pt_cases+a.a d
            FROM all_a a JOIN windows w USING(canonical_active_moiety)
            JOIN all_m m USING(canonical_active_moiety,canonical_pt_code)
          ), stat AS (
            SELECT *,
              CASE WHEN a>0 AND b>0 AND c>0 AND d>0 THEN (a::DOUBLE*d)/(b::DOUBLE*c) END ror,
              CASE WHEN a>0 AND b>0 AND c>0 AND d>0
                THEN exp(ln((a::DOUBLE*d)/(b::DOUBLE*c))-1.96*sqrt(1.0/a+1.0/b+1.0/c+1.0/d)) END ror_lcl95,
              log2((a+0.5)/((target_ps_cases::DOUBLE*total_pt_cases/NULLIF(total_cases,0))+0.5))
                -3.3*pow(a+0.5,-0.5)-2.0*pow(a+0.5,-1.5) ic025
            FROM cells
          )
          SELECT s.*,pt.canonical_pt_name,pt.canonical_soc_name,
                 (ror_lcl95>1) criterion_r, (ic025>0) criterion_ic,
                 (ror_lcl95>1 AND ic025>0) consensus,
                 CASE WHEN pre.canonical_active_moiety IS NOT NULL
                      THEN 'PREMARKETING_OBSERVED' ELSE 'POSTMARKETING_ONLY' END coverage_class
          FROM stat s LEFT JOIN pt USING(canonical_pt_code)
          LEFT JOIN pre USING(canonical_active_moiety,canonical_pt_code)
        ) TO '{ALL_SIGNALS}' (FORMAT PARQUET,COMPRESSION ZSTD)
        """
    )
    all_qc = con.execute(f"SELECT count(*) filter(where a<0 or b<0 or c<0 or d<0) FROM read_parquet('{ALL_SIGNALS}')").fetchone()[0]
    if all_qc:
        raise RuntimeError(f"Negative cells in all-pair registry: {all_qc}")

    coverage_rows = []
    for threshold in (0,50,100,200,500):
        for criterion,col in (("R","criterion_r"),("IC","criterion_ic"),("CONSENSUS","consensus")):
            rows = con.execute(
                f"""
                SELECT coverage_class,count(*) n
                FROM read_parquet('{ALL_SIGNALS}')
                WHERE target_ps_cases>={threshold} AND {col} GROUP BY 1
                """
            ).fetchall()
            d = dict(rows); total=sum(d.values())
            for klass in ("PREMARKETING_OBSERVED","POSTMARKETING_ONLY"):
                coverage_rows.append({"ps_minimum":threshold,"signal_definition":criterion,
                                      "coverage_class":klass,"positive_pairs":d.get(klass,0),
                                      "percent_of_positive_pairs":pct(d.get(klass,0),total)})
    # Required stratified distributions, emitted in the same long-form CSV.
    for dim,col in (("SOC","canonical_soc_name"),("NDA_BLA","nda_bla"),("ORPHAN","orphan_designation"),
                    ("ACCELERATED","accelerated_approval"),("APPROVAL_YEAR","approval_year")):
        rows = con.execute(
            f"""SELECT coverage_class,coalesce(cast({col} AS VARCHAR),'MISSING') stratum_level,count(*) n
                FROM read_parquet('{ALL_SIGNALS}') WHERE criterion_r GROUP BY 1,2""").fetchall()
        for klass,level,n in rows:
            coverage_rows.append({"ps_minimum":0,"signal_definition":"R_STRATIFIED_"+dim,
                                  "coverage_class":klass+":"+level,"positive_pairs":n,
                                  "percent_of_positive_pairs":""})
    write_csv(OUT / "05_signal_coverage_analysis.csv", coverage_rows)
    cov_r = [r for r in coverage_rows if r["ps_minimum"]==0 and r["signal_definition"] in ("R","IC","CONSENSUS")]
    (OUT / "05_signal_coverage_analysis.md").write_text(
        "# 05 — Premarketing safety coverage and blind spots\n\n"
        "The signal universe includes every target-PS drug–PT pair with `a≥3` in the exact three-year FDA window; "
        "it is not restricted to PTs seen in B-STRICT trials. `POSTMARKETING_ONLY` means absent from the qualifying "
        "AACT safety profile, not a novel ADR.\n\n" + md_table(cov_r,[
            ("signal_definition","Definition"),("coverage_class","Coverage"),("positive_pairs","Pairs"),
            ("percent_of_positive_pairs","Percent")]) +
        "\n\nSOC and regulatory-feature distributions are provided in the long-form CSV. A broad therapeutic-area taxonomy "
        "was not imposed because the FDA indication text has no frozen reproducible grouping dictionary.\n",
        encoding="utf-8")

    split_rows=[]
    for threshold in (0,50,100,200):
        for cut in (2017,2018,2019,2020):
            for partition,cond in (("DEVELOPMENT",f"l.approval_year<={cut}"),("TEMPORAL_HOLDOUT",f"l.approval_year>{cut}")):
                vals=con.execute(
                    f"""
                    WITH d AS (
                      SELECT l.canonical_active_moiety,count(*) pairs,
                             sum(l.criterion_r::INT) r_pos,sum(l.consensus::INT) con_pos
                      FROM labels3 l JOIN assess a USING(canonical_active_moiety)
                      WHERE a.ps_reports_3y>={threshold} AND {cond} GROUP BY 1
                    )
                    SELECT count(*) drugs,coalesce(sum(pairs),0) pairs,coalesce(sum(r_pos),0) r_pos,
                           coalesce(sum(con_pos),0) con_pos,sum((r_pos=0)::INT) zero_r,
                           median(r_pos) median_r
                    FROM d
                    """
                ).fetchone()
                pos_by_drug=[x[0] for x in con.execute(
                    f"""SELECT sum(l.criterion_r::INT) FROM labels3 l JOIN assess a USING(canonical_active_moiety)
                         WHERE a.ps_reports_3y>={threshold} AND {cond} GROUP BY l.canonical_active_moiety""").fetchall()]
                split_rows.append({"ps_minimum":threshold,"cut_year":cut,"partition":partition,
                    "active_moieties":vals[0],"drug_pt_pairs":vals[1],"criterion_r_positive_pairs":vals[2],
                    "criterion_r_prevalence_pct":pct(vals[2],vals[1]),"consensus_positive_pairs":vals[3],
                    "consensus_prevalence_pct":pct(vals[3],vals[1]),"drugs_with_zero_r_positives":vals[4] or 0,
                    "median_r_positives_per_drug":vals[5] or 0,"top3_r_positive_concentration_pct":concentration(pos_by_drug,3)})
    write_csv(OUT / "06_temporal_split_options.csv",split_rows)

    metrics={"assessability":assess_rows,"coverage":coverage_rows,"temporal_splits":split_rows,
             "all_exposed_pair_rows_a_ge_3":con.execute(f"select count(*) from read_parquet('{ALL_SIGNALS}')").fetchone()[0]}
    METRICS.write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"assessability_R":rrows,"coverage_primary":cov_r,
                      "split_threshold100":[r for r in split_rows if r['ps_minimum']==100],
                      "all_exposed_pair_rows":metrics['all_exposed_pair_rows_a_ge_3']},indent=2))


if __name__ == "__main__":
    main()
