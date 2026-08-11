#!/usr/bin/env python3
"""Cumulative JADER V5 replication of repaired three-year FAERS signals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb


PROJECT = Path("/path/to/PDS")
OUT = PROJECT / "preflight_v2"
PROC = PROJECT / "data/processed/preflight_v2"
JADER = Path("/path/to/Database/Jader")
JDRUG = JADER / "jader_v5_drug.parquet"
JREAC = JADER / "jader_v5_reac.parquet"
JDEMO = JADER / "jader_v5_demo.parquet"
JMAP = PROC / "jader_v5_name_to_fda_identity.csv"
MASTER = OUT / "drug_identity_master.csv"
BREG = OUT / "bstrict_candidate_registry.parquet"
FAERS = PROC / "faers_all_exposed_pair_signals_3y.parquet"
REGISTRY = PROC / "jader_v5_faers_positive_pair_replication.parquet"
COUNTS = OUT / "11_jader_v5_external_replication_counts.csv"
REPORT = OUT / "11_jader_v5_external_replication_audit.md"
METRICS = PROC / "jader_v5_replication_metrics.json"


def pct(n,d): return 100.0*n/d if d else 0.0


def write_csv(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def md_table(rows, fields):
    h="| "+" | ".join(x[1] for x in fields)+" |"; s="|"+"|".join("---" for _ in fields)+"|"
    body=[]
    for r in rows:
        vals=[]
        for k,_ in fields:
            v=r.get(k,"")
            if isinstance(v,float): v=f"{v:,.3f}"
            elif isinstance(v,int): v=f"{v:,}"
            vals.append(str(v))
        body.append("| "+" | ".join(vals)+" |")
    return "\n".join([h,s,*body])


def main() -> None:
    con=duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='20GB'; SET preserve_insertion_order=false; SET enable_progress_bar=false")
    # Match any of the normalized JADER generic/INN/molecule/English/brand
    # fields, but accept a drug row only if all matching fields resolve to one
    # FDA moiety. The source mapping itself contains no AACT synonym bridge.
    con.execute(f"""
      CREATE TABLE jdrug_ps AS
      WITH aliases AS (
        SELECT ID,DRUG_SEQ,DRUGNAME_CLEANED source_name FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,DRUG_INN_EN FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,DRUG_MOLECULE FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,DRUGNAME_EN FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
        UNION ALL SELECT ID,DRUG_SEQ,BRANDNAME_KEY FROM read_parquet('{JDRUG}') WHERE ROLE_STD='PS'
      ), hits AS (
        SELECT a.ID,a.DRUG_SEQ,m.canonical_active_moiety
        FROM aliases a JOIN read_csv('{JMAP}',header=true,all_varchar=true) m
          ON a.source_name=m.jader_source_name
      ), one AS (
        SELECT ID,DRUG_SEQ,any_value(canonical_active_moiety) canonical_active_moiety
        FROM hits GROUP BY 1,2 HAVING count(DISTINCT canonical_active_moiety)=1
      )
      SELECT DISTINCT ID,canonical_active_moiety FROM one
    """)
    con.execute(f"""
      CREATE TABLE jpt AS
      SELECT ID,try_cast(PT_CODE AS BIGINT) canonical_pt_code
      FROM read_parquet('{JREAC}') WHERE try_cast(PT_CODE AS BIGINT) IS NOT NULL GROUP BY 1,2
    """)
    con.execute("CREATE TABLE drug_n AS SELECT canonical_active_moiety,count(DISTINCT ID) n1 FROM jdrug_ps GROUP BY 1")
    con.execute("CREATE TABLE pt_n AS SELECT canonical_pt_code,count(DISTINCT ID) m FROM jpt GROUP BY 1")
    con.execute(f"CREATE TABLE pairs AS SELECT * FROM read_parquet('{FAERS}') WHERE criterion_r")
    con.execute("""
      CREATE TABLE overlap AS
      SELECT p.canonical_active_moiety,p.canonical_pt_code,count(DISTINCT d.ID) a
      FROM pairs p JOIN jdrug_ps d USING(canonical_active_moiety)
      JOIN jpt e ON e.ID=d.ID AND e.canonical_pt_code=p.canonical_pt_code
      GROUP BY 1,2
    """)
    n=con.execute(f"SELECT count(*) FROM read_parquet('{JDEMO}')").fetchone()[0]
    con.execute(f"""
      COPY (
        WITH cells AS (
          SELECT p.*,coalesce(d.n1,0)::BIGINT jader_ps_cases,coalesce(t.m,0)::BIGINT jader_pt_cases,
                 coalesce(o.a,0)::BIGINT jader_a,
                 (coalesce(d.n1,0)-coalesce(o.a,0))::BIGINT jader_b,
                 (coalesce(t.m,0)-coalesce(o.a,0))::BIGINT jader_c,
                 ({n}-coalesce(d.n1,0)-coalesce(t.m,0)+coalesce(o.a,0))::BIGINT jader_d
          FROM pairs p LEFT JOIN drug_n d USING(canonical_active_moiety)
          LEFT JOIN pt_n t USING(canonical_pt_code)
          LEFT JOIN overlap o USING(canonical_active_moiety,canonical_pt_code)
        ), stat AS (
          SELECT *,
            CASE WHEN jader_a>0 AND jader_b>0 AND jader_c>0 AND jader_d>0
              THEN (jader_a::DOUBLE*jader_d)/(jader_b::DOUBLE*jader_c) END jader_ror,
            CASE WHEN jader_a>0 AND jader_b>0 AND jader_c>0 AND jader_d>0
              THEN exp(ln((jader_a::DOUBLE*jader_d)/(jader_b::DOUBLE*jader_c))
                       -1.96*sqrt(1.0/jader_a+1.0/jader_b+1.0/jader_c+1.0/jader_d)) END jader_ror_lcl95,
            log2((jader_a+0.5)/((jader_ps_cases::DOUBLE*jader_pt_cases/{n})+0.5))
              -3.3*pow(jader_a+0.5,-0.5)-2.0*pow(jader_a+0.5,-1.5) jader_ic025
          FROM cells
        )
        SELECT *,
          (jader_ps_cases>0 AND jader_pt_cases>0) pair_represented,
          (jader_ps_cases>=50 AND jader_pt_cases>0) jader_assessable,
          (jader_ror>1) directional_ror_gt1,
          (jader_ps_cases>=50 AND jader_pt_cases>0 AND jader_a>=3 AND jader_ror_lcl95>1) jader_criterion_r,
          (jader_ps_cases>=50 AND jader_pt_cases>0 AND jader_a>=3 AND jader_ror_lcl95>1 AND jader_ic025>0) jader_consensus,
          CASE WHEN jader_ps_cases<50 OR jader_pt_cases=0 THEN 'NOT_ASSESSABLE'
               WHEN jader_a>=3 AND jader_ror_lcl95>1 THEN 'REPLICATED'
               ELSE 'NOT_REPLICATED' END replication_class
        FROM stat
      ) TO '{REGISTRY}' (FORMAT PARQUET,COMPRESSION ZSTD)
    """)
    qc=con.execute(f"""SELECT count(*) filter(where jader_a<0 or jader_b<0 or jader_c<0 or jader_d<0),
             count(*) filter(where jader_a+jader_b+jader_c+jader_d<>{n}) FROM read_parquet('{REGISTRY}')""").fetchone()
    if qc != (0,0): raise RuntimeError(f"JADER 2x2 QC failure: {qc}")

    cohort_drugs=con.execute(f"""SELECT count(*) FROM read_csv('{MASTER}',header=true,all_varchar=true)
                                  WHERE cast(approval_year AS INTEGER) between 2012 and 2022 and exclusion_flag='False'""").fetchone()[0]
    bstrict_drugs=con.execute(f"select count(distinct canonical_active_moiety) from read_parquet('{BREG}')").fetchone()[0]
    mapped_cohort=con.execute(f"""SELECT count(DISTINCT m.canonical_active_moiety)
      FROM read_csv('{MASTER}',header=true,all_varchar=true) m JOIN drug_n d USING(canonical_active_moiety)
      WHERE cast(m.approval_year AS INTEGER) between 2012 and 2022 and m.exclusion_flag='False'""").fetchone()[0]
    tier_rows=[]
    for t in (1,50,100,200,500):
        x=con.execute(f"""SELECT count(*) FROM (SELECT DISTINCT p.canonical_active_moiety
                     FROM read_parquet('{BREG}') p JOIN drug_n d USING(canonical_active_moiety) WHERE d.n1>={t})""").fetchone()[0]
        tier_rows.append({"section":"BSTRICT_DRUG_PS_TIER","metric":f"JADER_PS_GE_{t}","n":x,"denominator":bstrict_drugs,"percent":pct(x,bstrict_drugs)})
    pair=con.execute(f"""SELECT count(*) total,sum(pair_represented::INT) represented,sum(jader_assessable::INT) assessable,
              sum(directional_ror_gt1::INT) filter(where jader_assessable) directional,
              sum(jader_criterion_r::INT) filter(where jader_assessable) replicated_r,
              sum(jader_consensus::INT) filter(where jader_assessable) replicated_consensus,
              sum((replication_class='NOT_REPLICATED')::INT) not_replicated,
              sum((replication_class='NOT_ASSESSABLE')::INT) not_assessable
              FROM read_parquet('{REGISTRY}')""").fetchone()
    counts=[
      {"section":"FDA_COHORT","metric":"active_moieties_2012_2022","n":cohort_drugs,"denominator":cohort_drugs,"percent":100.0},
      {"section":"FDA_COHORT","metric":"mapped_to_jader_any_ps","n":mapped_cohort,"denominator":cohort_drugs,"percent":pct(mapped_cohort,cohort_drugs)},
      *tier_rows,
      {"section":"FAERS_R_POSITIVE_PAIRS","metric":"total","n":pair[0],"denominator":pair[0],"percent":100.0},
      {"section":"FAERS_R_POSITIVE_PAIRS","metric":"represented_drug_and_pt","n":pair[1],"denominator":pair[0],"percent":pct(pair[1],pair[0])},
      {"section":"FAERS_R_POSITIVE_PAIRS","metric":"assessable_jader_ps_ge50","n":pair[2],"denominator":pair[0],"percent":pct(pair[2],pair[0])},
      {"section":"ASSESSABLE_PAIRS","metric":"directional_ror_gt1","n":pair[3],"denominator":pair[2],"percent":pct(pair[3],pair[2])},
      {"section":"ASSESSABLE_PAIRS","metric":"a_ge3_ror_lcl_gt1","n":pair[4],"denominator":pair[2],"percent":pct(pair[4],pair[2])},
      {"section":"ASSESSABLE_PAIRS","metric":"a_ge3_ror_lcl_gt1_ic025_gt0","n":pair[5],"denominator":pair[2],"percent":pct(pair[5],pair[2])},
      {"section":"CLASSIFICATION","metric":"NOT_REPLICATED","n":pair[6],"denominator":pair[0],"percent":pct(pair[6],pair[0])},
      {"section":"CLASSIFICATION","metric":"NOT_ASSESSABLE","n":pair[7],"denominator":pair[0],"percent":pct(pair[7],pair[0])},
    ]
    write_csv(COUNTS,counts)
    REPORT.write_text(
      "# 11 — JADER V5 cumulative external replication\n\n"
      "**Role:** cumulative cross-database replication only. No US-approval-anchored JADER window, TTO, death enrichment, or Japanese temporal validation was used.\n\n"
      "The prespecified provisional assessability rule is JADER target-drug PS ≥50 with the PT represented in JADER. "
      "The provisional replication rule is `a≥3` and ROR lower 95% CI >1; IC025>0 is reported as a stricter consensus.\n\n"+
      md_table(counts,[("section","Section"),("metric","Metric"),("n","N"),("denominator","Denominator"),("percent","Percent")])+
      "\n\nEvery four-cell table uses distinct JADER `ID` values over the same cumulative V5 database. Cases carrying the target as PS remain exclusively on the exposed side. "
      "Identity uses the frozen FDA master and direct FDA active/brand or unique salt/suffix mappings; the rejected AACT cross-database alias bridge is not used.\n",
      encoding="utf-8")
    metrics={"jader_total_cases":n,"fda_cohort_drugs":cohort_drugs,"fda_cohort_drugs_mapped_any_ps":mapped_cohort,
             "bstrict_drugs":bstrict_drugs,"pair_counts":dict(zip(["total","represented","assessable","directional","replicated_r","replicated_consensus","not_replicated","not_assessable"],pair)),
             "qc":{"negative_cells":qc[0],"invalid_totals":qc[1]},"assessability_ps_minimum":50,
             "replication_rule":"a>=3 and ROR lower 95% CI >1"}
    METRICS.write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__": main()
