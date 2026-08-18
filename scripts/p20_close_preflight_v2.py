#!/usr/bin/env python3
"""Assemble the Command 02C feature lock, reconciliation, and final decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT=Path("/path/to/project")
OUT=PROJECT/"preflight_v2"
PROC=PROJECT/"data/processed/preflight_v2"


def loadj(path): return json.loads(path.read_text(encoding="utf-8"))
def loadc(path):
    with path.open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
def pct(n,d): return round(100*n/d,2) if d else 0.0
def n(x): return f"{int(float(x)):,}"


def main():
    fda=loadj(PROC/"fda_regulatory_cohort_metrics.json")
    ident=loadj(PROC/"drug_identity_metrics.json")
    bst=loadj(PROC/"bstrict_metrics.json")
    labels=loadj(PROC/"faers_label_metrics.json")
    asc=loadc(OUT/"04_outcome_assessability.csv")
    cov=loadc(OUT/"05_signal_coverage_analysis.csv")
    splits=loadc(OUT/"06_temporal_split_options.csv")
    jad=loadj(PROC/"jader_v5_replication_metrics.json")
    term=loadj(PROC/"faers_pt_repair/terminology_repair_metrics.json")
    casept=loadj(PROC/"faers_pt_repair/faers_case_pt_rebuild_metrics.json")

    feature_text="""# 07 — Feature feasibility and incremental-value lock

No model was fitted. All features must be computed from information whose qualifying trial completion precedes FDA first approval. FAERS/JADER counts, labels, approval-window outcomes, and future holdout outcomes are forbidden as predictors.

## Feature Set 0 — Baseline / Context

| Level | Locked variables | Construction note |
|---|---|---|
| Drug | approval year; NDA/BLA; orphan; accelerated approval; breakthrough; fast track; priority review; route; dosage form | Frozen FDA master only |
| PT | canonical MedDRA 28.0 PT code/name and SOC | Identity/context, not spontaneous-report frequency |
| Drug | number of qualifying B-STRICT trials and uniquely attributable target arms; total nonduplicated target-arm exposure; phase, randomized and masked fractions | Exposure is deduplicated once per `nct_id × result/design group`, never summed over PT rows |
| Pair | drug and PT identity keys only | No pair-specific AE occurrence or count in Set 0 |

## Feature Set 1 — Pair-specific premarketing safety

Feature Set 0 plus:

- number of B-STRICT trials reporting the canonical PT and fraction of the drug's qualifying trials;
- nonduplicated treatment-arm subjects at risk for the PT;
- trial/arm-specific affected proportion, pooled descriptive proportion, median and maximum proportion;
- between-trial variability of arm-level proportions, with the exact scale/estimator frozen before modelling;
- ClinicalTrials.gov serious-event presence and fraction of PT-reporting trials with `event_type='serious'`;
- minimum/maximum reported frequency threshold and threshold-zero fraction;
- phase distribution and randomized/masked fractions among PT-reporting trials.

`serious` is the ClinicalTrials.gov results-table category and must never be described as CTCAE grade ≥3. Missing trial-design fields receive explicit missing indicators/categories; no outcome-informed imputation or feature selection is allowed.

## Incremental-value estimand

The future analysis must compare Feature Set 0 versus Feature Set 1 under identical approval-year splits, drug-level grouping, preprocessing and metrics. The locked question is whether pair-specific premarketing trial safety information adds predictive value for three-year FAERS disproportionality signals beyond baseline regulatory, drug, event and trial context.
"""
    (OUT/"07_feature_feasibility.md").write_text(feature_text,encoding="utf-8")

    legacy_codes=13887
    numeric=f"""# 12 — Numeric terminology reconciliation

The word *term*, *PT*, and *pair* are not interchangeable. The authoritative reconciliation is:

| Object | Grain/meaning | Count |
|---|---|---:|
| Legacy FAERS universe | rows grouped on the defective stored `pt_u` object | 20,385 |
| Legacy distinct stored PT codes | distinct `pt_code` values in that legacy universe; not authoritative | {legacy_codes:,} |
| Repaired source terms | distinct original FAERS `pt` spellings | 39,640 |
| Repaired mapped source terms | original source spellings with a unique MedDRA 28.0 resolution | 39,602 |
| Repaired canonical FAERS PTs | distinct canonical MedDRA 28.0 PT codes represented by mapped source terms | 22,704 |
| Latest-case canonical PTs | canonical PT codes actually represented after latest-case selection | 22,612 |

Thus 38 source terms remain unresolved/ambiguous, and 92 canonical PTs from the repaired source-term universe are absent after latest-case selection. The 20,385 legacy rows are not a MedDRA PT count and cannot be compared as if they were a one-to-one PT universe. The legacy object's independent `any_value()` selections misaligned term, name and code; 6,646 of 20,347 mappable legacy terms carried the wrong stored code.

Downstream counts use explicit labels: **source reaction terms**, **mapped source terms**, **distinct canonical PTs**, or **canonical drug–PT pairs**. No legacy FAERS PT code was used in B-STRICT linkage, FAERS labels, coverage, or JADER replication.
"""
    (OUT/"12_numeric_terminology_reconciliation.md").write_text(numeric,encoding="utf-8")

    map_rows=[]
    for win,v in ident["windows"].items():
        total=v["active_moieties"]
        map_rows.append(f"| {win} | {total:,} | {v['mapped_aact']:,} ({pct(v['mapped_aact'],total):.2f}%) | {v['mapped_faers']:,} ({pct(v['mapped_faers'],total):.2f}%) | {v['mapped_jader']:,} ({pct(v['mapped_jader'],total):.2f}%) | {v['mapped_all_three']:,} | {v['unresolved_all']:,} |")
    arows={int(r['ps_minimum']):r for r in asc if r['signal_definition']=='R'}
    icrows={int(r['ps_minimum']):r for r in asc if r['signal_definition']=='IC'}
    conrows={int(r['ps_minimum']):r for r in asc if r['signal_definition']=='CONSENSUS'}
    assess_lines=[]
    for t in (0,50,100,200,500):
        r=arows[t]; i=icrows[t]; c=conrows[t]
        assess_lines.append(f"| {t} | {n(r['active_moieties'])} | {n(r['candidate_drug_pt_pairs'])} | {n(r['positive_pairs'])} | {n(i['positive_pairs'])} | {n(c['positive_pairs'])} | {float(r['positive_prevalence_pct']):.2f}% | {float(r['top3_positive_concentration_pct']):.2f}% |")
    split100=[r for r in splits if r['ps_minimum']=='100']
    split_lines=[f"| {r['cut_year']} | {r['partition']} | {n(r['active_moieties'])} | {n(r['drug_pt_pairs'])} | {n(r['criterion_r_positive_pairs'])} | {n(r['consensus_positive_pairs'])} | {float(r['top3_r_positive_concentration_pct']):.2f}% |" for r in split100]
    temporal={x['definition']:x for x in bst['temporal_definitions'] if x['window']=='2012-2022'}
    cov0={(r['signal_definition'],r['coverage_class']):r for r in cov if r['ps_minimum']=='0' and r['signal_definition'] in ('R','IC','CONSENSUS')}
    h3=[x for x in labels['horizons'] if x['horizon_years']==3][0]

    report=f"""# PREFLIGHT V2 REPORT — FDA-ANCHORED CLOSEOUT

**Overall decision: CONDITIONAL**  
**Primary design:** B-STRICT, FDA approvals 2012–2022  
**Proposed prespecified restriction for scientific review:** ≥100 three-year FAERS PS cases; development approvals 2012–2018 and temporal holdout 2019–2022  
**Machine learning:** not performed

## Decision

The study is viable: B-STRICT provides 212 unique active moieties, 620 trials, 1,149 target treatment arms and 30,247 canonical drug–PT pairs. At the proposed PS≥100 restriction, 166 drugs and 26,151 pairs remain, retaining 97.51% of Criterion-R positives. A 2019–2022 temporal holdout then contains 59 drugs, 9,681 pairs, 1,110 Criterion-R positives and 943 Consensus positives.

The decision is **CONDITIONAL**, rather than PASS, because exact-title arm attribution is materially selective (maximum absolute SMD 0.349) and because the outcome-assessability threshold and split remain for the scientific architect to freeze. B-EXPANDED cannot replace B-STRICT. JADER limitations do not invalidate the primary US study.

## 1. Official FDA backbone and regulatory cohort

The immutable FDA CDER 1985–2025 XLSX and data dictionary are archived under `data_external/fda/` with source URL, retrieval date, byte size and SHA256 in `SOURCE_MANIFEST.json`. The FDA file contains 1,387 approval records and 1,382 exact active-ingredient/moiety groups. Five groups have multiple records, 69 multi-active products are flagged and excluded from the primary cohort, 32 administrative transition/new-license groups are retained for audit, and 237 salt/ester/prodrug flags receive identity review.

| Approval window | FDA source identities | Primary single-active moieties | Combination exclusions | Complete exact 3y follow-up |
|---|---:|---:|---:|---:|
| 2012–2022 | 466 | 432 | 34 | 432 |
| 2013–2022 | 427 | 395 | 32 | 395 |
| 2014–2022 | 400 | 371 | 29 | 371 |
| 2015–2022 | 359 | 334 | 25 | 334 |

All retained approvals through 2022 satisfy exact approval date + 3 calendar years ≤ 2025-12-31.

## 2. Frozen drug identity

`drug_identity_master.csv` has 1,382 rows and SHA256 `{ident['master_sha256']}`. FAERS and JADER accept only direct FDA active/brand matches or unique deterministic salt/suffix normalization. An initially tested AACT cross-database synonym bridge was rejected after it produced a false methylprednisolone→tacrolimus link; all affected mappings were rebuilt before labels. Outcome fields, PTs, signal status and fuzzy matching were never used.

| Window | FDA moieties | AACT mapped | FAERS mapped | JADER mapped | All three | Unresolved all |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(map_rows)}

All 360 final-window review flags have dispositions and audit notes; 3,110 AACT, 242 FAERS and 9 JADER ambiguous one-to-many source names are explicitly excluded in `drug_identity_ambiguous_source_names.csv`. The single fully unresolved identity is not silently imputed.

## 3. B-STRICT and temporal definitions

Primary B-STRICT requires actual study completion on/before FDA first approval, excludes Phase 4, invalid AE denominators, noncanonical terms, ambiguous arm attribution, non-target/combination arms and placebo/control arms.

| Definition (2012–2022) | Drugs | Trials | Arms | Canonical drug–PT pairs |
|---|---:|---:|---:|---:|
| A: results posted by approval (sensitivity) | {temporal['A_RESULTS_POSTED']['active_moieties']:,} | {temporal['A_RESULTS_POSTED']['trials']:,} | {temporal['A_RESULTS_POSTED']['arms']:,} | {temporal['A_RESULTS_POSTED']['drug_pt_pairs']:,} |
| B-STRICT actual completion (primary) | {temporal['B_STRICT_ACTUAL_COMPLETION']['active_moieties']:,} | {temporal['B_STRICT_ACTUAL_COMPLETION']['trials']:,} | {temporal['B_STRICT_ACTUAL_COMPLETION']['arms']:,} | {temporal['B_STRICT_ACTUAL_COMPLETION']['drug_pt_pairs']:,} |
| B-EXPANDED primary completion (sensitivity) | {temporal['B_EXPANDED_PRIMARY_COMPLETION']['active_moieties']:,} | {temporal['B_EXPANDED_PRIMARY_COMPLETION']['trials']:,} | {temporal['B_EXPANDED_PRIMARY_COMPLETION']['arms']:,} | {temporal['B_EXPANDED_PRIMARY_COMPLETION']['drug_pt_pairs']:,} |

B-EXPANDED adds 27 drugs, 96 trials, 171 arms and 8,209 pairs over B-STRICT; it remains sensitivity only. The full cascade and all candidate windows are in `02_temporal_definition_reaudit.md` and `bstrict_candidate_registry_summary.csv`.

## 4. Arm-attribution selection

Among 45,449 result-bearing interventional drug/biological trials, 28,245 have at least one uniquely title-mapped Reported Event group and 17,204 have none. Maximum |SMD| is 0.349: mapped trials are less often industry-sponsored (0.540 vs 0.688), less often crossover (0.032 vs 0.113), more often parallel (0.648 vs 0.495), and have fewer AE rows on average (175 vs 348). This is meaningful selection; positional arm recovery remains prohibited. The estimand must be stated as safety information available from uniquely attributable target monotherapy arms, not all preapproval trials.

## 5. FDA-anchored FAERS labels

For each drug the inclusive interval is FDA first approval date through its exact 1/2/3-year calendar anniversary; all earlier reports are excluded. Exposure is the frozen active moiety reported as PS in the selected latest case. For each candidate pair and drug-specific window: `a=PS target+PT`, `b=PS target without PT`, `c=non-target-PS background+PT`, `d=non-target-PS background without PT`. Each case appears on exactly one exposure side. Counts use distinct `caseid`; drug×PT source rows are never cases.

ROR-LCL uses the log-Wald 95% CI without outcome-driven continuity correction. IC025 uses `IC - 3.3(a+0.5)^-0.5 - 2(a+0.5)^-1.5`, with `IC=log2((a+0.5)/(E+0.5))`. Cell QC found zero negative cells and zero marginal/total inconsistencies.

At three years the 30,247 B-STRICT pairs contain {h3['criterion_r_positive']:,} Criterion-R positives ({pct(h3['criterion_r_positive'],h3['pairs']):.2f}%), {h3['criterion_ic_positive']:,} Criterion-IC positives ({pct(h3['criterion_ic_positive'],h3['pairs']):.2f}%), and {h3['consensus_positive']:,} Consensus positives. R and IC disagree for {h3['r_ic_disagreement']:,} pairs.

## 6. Outcome assessability

| Minimum 3y PS | Drugs | Candidate pairs | R positive | IC positive | Consensus | R prevalence | R top-3 concentration |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(assess_lines)}

PS≥100 is proposed because it retains 166 drugs, four zero-R-positive drugs and 97.51% of all B-STRICT Criterion-R positives while avoiding the sparsest 46 drugs. This is a scientific assessability choice, not a model-performance choice, and requires architect approval.

## 7. Premarketing coverage and blind spots

Coverage uses a separate all-exposed-pair registry so it is not mechanically restricted to PTs already seen in AACT. Among {n(int(cov0[('R','PREMARKETING_OBSERVED')]['positive_pairs'])+int(cov0[('R','POSTMARKETING_ONLY')]['positive_pairs']))} three-year Criterion-R signals, {cov0[('R','PREMARKETING_OBSERVED')]['positive_pairs']} ({float(cov0[('R','PREMARKETING_OBSERVED')]['percent_of_positive_pairs']):.2f}%) are `PREMARKETING_OBSERVED` and {cov0[('R','POSTMARKETING_ONLY')]['positive_pairs']} ({float(cov0[('R','POSTMARKETING_ONLY')]['percent_of_positive_pairs']):.2f}%) are `POSTMARKETING_ONLY`. The latter are reporting signals absent from qualifying B-STRICT safety profiles, not “novel ADRs.” SOC and regulatory-stratum distributions are in `05_signal_coverage_analysis.csv`.

## 8. Temporal split options at proposed PS≥100

| Cut year | Partition | Drugs | Pairs | R positive | Consensus | R top-3 concentration |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(split_lines)}

The proposed cut is 2018: development 2012–2018 (107 drugs) and holdout 2019–2022 (59 drugs). It balances unique-drug counts and preserves a genuinely later cohort; it was not chosen using model performance.

## 9. JADER V5 cumulative replication

JADER contains 1,043,485 distinct cases. Of the 432 FDA-window drugs, 199 have a verified JADER PS mapping; among 212 B-STRICT drugs, 86 have ≥50 JADER PS cases. Of 16,541 three-year FAERS Criterion-R signals across the all-exposed-pair universe, 11,201 have both drug and PT represented and 9,753 meet the provisional JADER PS≥50 assessability rule. Among assessable pairs, 3,113 have ROR>1, 1,531 meet `a≥3 and ROR-LCL>1`, and 1,372 also have IC025>0. JADER is cumulative cross-database replication only.

## 10. Feature and incremental-value lock

Feature Set 0 contains FDA regulatory/drug context, canonical PT/SOC identity and drug-level B-STRICT trial context/exposure. Feature Set 1 adds only pair-specific premarketing trial evidence: PT-reporting trial counts/fractions, nonduplicated arm exposure, arm-level AE proportions and summaries, serious-event presence/fraction, between-trial variability, threshold characteristics, phase and randomized/masked fractions. Identical drug-grouped temporal splits must compare Set 0 with Set 1. No FAERS/JADER outcome-derived predictor is allowed.

## 11. Numeric terminology reconciliation

The legacy object has 20,385 rows but only 13,887 distinct stored PT codes; neither is authoritative. The repaired layer begins with 39,640 original reaction terms, maps 39,602 of them, resolves to 22,704 canonical PTs, and represents 22,612 canonical PTs in the latest-case layer. See `12_numeric_terminology_reconciliation.md`.

## 12. Residual methodological risks

1. Exact-title arm attribution is materially selective (max |SMD| 0.349), limiting transportability to all preapproval trials.
2. FAERS and JADER are spontaneous-report systems: disproportionality is neither incidence nor causality and remains sensitive to reporting, notoriety, indication and channeling bias.
3. The proposed FAERS PS≥100 restriction excludes 46 B-STRICT drugs; the scientific architect must freeze the assessability rule before modelling.
4. JADER drug coverage is limited (86/212 B-STRICT drugs at PS≥50), and cross-country product use/reporting differences constrain replication interpretation.
5. ClinicalTrials.gov AE reporting thresholds, incomplete historical posting and monotherapy/unique-arm restrictions can omit genuine premarketing evidence; `POSTMARKETING_ONLY` therefore means unrepresented in this data design, not biologically new.

## 13. Final disposition and next lock

**CONDITIONAL:** authorize modelling only after the scientific architect explicitly freezes or revises (i) PS-volume threshold, (ii) approval-year split, and (iii) the uniquely attributable-arm estimand/selection limitation. Recommended locks are PS≥100 and a 2018 cut. B-STRICT remains primary, Criterion R remains the primary signal rule, Consensus remains sensitivity, and all validation must be grouped by active moiety. No model was trained in this command.

## Reproducibility

Authoritative scripts are `scripts/p14_build_fda_regulatory_cohort.py` through `scripts/p20_close_preflight_v2.py`. Machine-readable FDA, identity, B-STRICT, FAERS label, coverage/split and JADER metrics are under `data/processed/preflight_v2/`. Raw FDA, AACT, FAERS, JADER and MedDRA assets were read only.
"""
    (OUT/"PREFLIGHT_V2_REPORT.md").write_text(report,encoding="utf-8")

    decision={
      "overall_status":"CONDITIONAL",
      "analysis_stage":"PREFLIGHT_V2_CLOSED_AWAITING_SCIENTIFIC_LOCK",
      "primary_design":"B_STRICT_ACTUAL_COMPLETION",
      "primary_approval_window":"2012-2022",
      "proposed_faers_ps_minimum":100,
      "proposed_temporal_split":{"development":"2012-2018","holdout":"2019-2022","cut_year":2018},
      "primary_signal_definition":"Criterion R: a>=3 and ROR lower 95% CI >1",
      "sensitivity_signal_definition":"Criterion IC and R+IC Consensus",
      "identity_master_sha256":ident['master_sha256'],
      "bstrict":{"active_moieties":212,"trials":620,"arms":1149,"drug_pt_pairs":30247},
      "faers_3y":{"criterion_r_positive_pairs":3255,"criterion_ic_positive_pairs":2834,"consensus_positive_pairs":2834},
      "proposed_analysis_population":{"active_moieties":166,"drug_pt_pairs":26151,"criterion_r_positive_pairs":3174,"consensus_positive_pairs":2768},
      "arm_mapping":{"mapped_trials":28245,"unmapped_trials":17204,"max_abs_smd":0.3493978707539853,"selection_behavior":"MATERIAL_REQUIRES_EXPLICIT_ESTIMAND_LIMITATION"},
      "jader":{"role":"cumulative cross-database replication","assessable_faers_r_pairs":9753,"replicated_r_pairs":1531,"replicated_consensus_pairs":1372},
      "conditions_before_modelling":["scientific architect freezes FAERS PS threshold","scientific architect freezes temporal cut","manuscript estimand explicitly restricted to uniquely attributable target monotherapy arms"],
      "machine_learning_trained":False,"tto_performed":False,"death_enrichment_performed":False,
      "raw_sources_modified":False
    }
    (OUT/"PREFLIGHT_V2_DECISION.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(decision,indent=2))


if __name__=="__main__": main()
