#!/usr/bin/env python3
"""Finalize Command 02B audit artefacts without fabricating blocked results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT = Path("/path/to/PDS")
OUT = PROJECT / "preflight_v2/faers_pt_repair"
PROC = PROJECT / "data/processed/preflight_v2/faers_pt_repair"


def load(name: str) -> dict:
    return json.loads((PROC / name).read_text(encoding="utf-8"))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    term = load("terminology_repair_metrics.json")["decision"]
    case = load("faers_case_pt_rebuild_metrics.json")
    aact = load("aact_meddra28_rebuild_metrics.json")

    blockers = [
        "validated FDA first-approval dates and frozen FDA active-moiety cohort are absent",
        "drug_identity_master.csv is absent",
        "the B-STRICT registry cannot be formed without those two inputs",
        "the original Command 02 temporal split and regulatory subgroup locks are absent",
    ]
    blocker_text = "; ".join(blockers)

    (OUT / "03_faers_temporal_labels_rebuilt.md").write_text(
        f"""# 03 — FAERS temporal labels rebuild

**Status: BLOCKED; no labels calculated.**

The repaired terminology layer passed, and the latest-case object now contains {case['latest_cases']:,} cases and {case['case_pt_rows']:,} unique case–PT rows. The remaining temporal calculation is not identifiable from local inputs because the validated FDA first-approval date and frozen FDA drug identity are absent.

Consequently, 1-, 2-, and 3-year Criterion R, Criterion IC, and consensus labels are **NOT COMPUTABLE**. The first FAERS report date, AACT completion/posting date, or calendar-year approximation was not substituted. The earlier technical-anchor pilot and all its prevalence values are superseded and are not repaired results.

Blockers: {blocker_text}.
""",
        encoding="utf-8",
    )

    assess_rows = [
        {
            "min_3y_ps_cases": threshold,
            "drugs": "",
            "drug_pt_pairs": "",
            "criterion_r_positives": "",
            "consensus_positives": "",
            "criterion_r_prevalence_pct": "",
            "consensus_prevalence_pct": "",
            "median_positives_per_drug": "",
            "drugs_zero_positive_pairs": "",
            "top1_positive_concentration_pct": "",
            "top3_positive_concentration_pct": "",
            "top5_positive_concentration_pct": "",
            "top10_positive_concentration_pct": "",
            "status": "BLOCKED",
            "blocker": "repaired 3-year FDA-anchored labels unavailable",
        }
        for threshold in (0, 50, 100, 200, 500)
    ]
    write_csv(
        OUT / "04_outcome_assessability_rebuilt.csv",
        list(assess_rows[0]),
        assess_rows,
    )

    (OUT / "05_signal_coverage_rebuilt.md").write_text(
        """# 05 — Safety-coverage rebuild

**Status: BLOCKED; no coverage percentages calculated.**

Classification into `PREMARKETING_OBSERVED` and `POSTMARKETING_ONLY` requires both a repaired, frozen three-year FAERS-positive pair registry and a qualifying B-STRICT AACT pair registry on the same drug identity. Neither registry can be constructed without validated FDA approval dates and the frozen cross-database drug master. The approval-independent AACT ceiling is not substituted for B-STRICT.

No pair is labelled a novel ADR. SOC and regulatory subgroup distributions remain not computable.
""",
        encoding="utf-8",
    )

    split_fields = [
        "split_definition", "development_approval_years", "holdout_approval_years",
        "set", "unique_drugs", "drug_pt_pairs", "positive_labels", "prevalence_pct",
        "drugs_zero_positives", "top1_concentration_pct", "top3_concentration_pct",
        "top5_concentration_pct", "top10_concentration_pct", "status", "blocker",
    ]
    write_csv(
        OUT / "06_temporal_split_counts_rebuilt.csv",
        split_fields,
        [{
            "split_definition": "NOT SPECIFIED IN AVAILABLE PROJECT FILES",
            "development_approval_years": "",
            "holdout_approval_years": "",
            "set": "NOT COMPUTABLE",
            "unique_drugs": "", "drug_pt_pairs": "", "positive_labels": "",
            "prevalence_pct": "", "drugs_zero_positives": "",
            "top1_concentration_pct": "", "top3_concentration_pct": "",
            "top5_concentration_pct": "", "top10_concentration_pct": "",
            "status": "BLOCKED",
            "blocker": "approval-year split lock, FDA approval years, and repaired labels absent",
        }],
    )

    (OUT / "07_jader_v5_replication_rebuilt.md").write_text(
        f"""# 07 — JADER V5 replication feasibility after FAERS PT repair

**Global terminology assessability: PASS.**  
**FDA-cohort repaired-pair replication: BLOCKED; no replication rates calculated.**

The frozen JADER V5 structural results remain unchanged. Repaired FAERS contains {term['distinct_repaired_faers_pt_codes']:,} current PTs; JADER contains {term['jader_v5_pt_codes']:,}; {term['shared_faers_jader_pt_codes']:,} are shared ({term['jader_to_faers_coverage_pct']:.4f}% of JADER PTs). This establishes terminology capacity only.

`REPLICATED`, `NOT_REPLICATED`, and `NOT_ASSESSABLE` can be assigned only after the FDA-cohort repaired FAERS-positive pairs and frozen drug identities exist. Therefore FDA-cohort drugs in JADER, ≥50/100/200/500 PS tiers, represented positive pairs, directional replication, ROR-LCL replication, and ROR-LCL+IC025 consensus replication are **NOT COMPUTABLE**. No Japanese time window, death enrichment, or TTO analysis was performed.
""",
        encoding="utf-8",
    )

    old_pt = 13_887
    new_pt = term["distinct_repaired_faers_pt_codes"]
    old_shared = 11_313
    new_shared = term["shared_faers_jader_pt_codes"]
    (OUT / "08_old_vs_repaired_impact.md").write_text(
        f"""# 08 — Old versus repaired impact

**Impact status: scientifically material at terminology level; endpoint impact not yet estimable.**

| Quantity | OLD | REPAIRED | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| Distinct stored/repaired FAERS PT codes | {old_pt:,} | {new_pt:,} | {new_pt-old_pt:+,} | {100*(new_pt-old_pt)/old_pt:+.4f}% |
| Global shared FAERS↔JADER PT codes | {old_shared:,} | {new_shared:,} | {new_shared-old_shared:+,} | {100*(new_shared-old_shared)/old_shared:+.4f}% |
| JADER→FAERS global terminology coverage | 97.9396% | {term['jader_to_faers_coverage_pct']:.4f}% | {term['jader_to_faers_coverage_pct']-97.9396:+.4f} pp | NA |
| B-STRICT drug–PT pairs | NOT VALID | NOT COMPUTABLE | NA | NA |
| 1-year positive labels | superseded pilot | NOT COMPUTABLE | NA | NA |
| 2-year positive labels | superseded pilot | NOT COMPUTABLE | NA | NA |
| 3-year positive labels/prevalence | superseded pilot | NOT COMPUTABLE | NA | NA |
| Signal coverage | not valid | NOT COMPUTABLE | NA | NA |
| FDA-cohort JADER-overlap pairs | provisional/not frozen | NOT COMPUTABLE | NA | NA |
| JADER-replicated pairs | provisional/not frozen | NOT COMPUTABLE | NA | NA |

The repaired audit found {term['legacy_mappable_terms_wrong_code']:,} of {term['legacy_terms_mappable']:,} conservatively mappable legacy terms ({term['legacy_wrong_code_pct']:.4f}%) with a wrong stored code. In the rebuilt latest-case corpus, a same-text comparison to the legacy term records flags 11,133,019 `caseid × source-term` occurrences across 7,749,698 cases as potentially affected; this is an impact bound, **not** a drug–PT label-change count, because the legacy aggregate was created at an invalid expanded grain.

Pairs whose label changed and pairs whose canonical identity changed remain not computable until B-STRICT drug identity and FDA-anchored labels are rebuilt. Reporting the old approval-independent AACT table or technical-anchor pilot as OLD B-STRICT would create a false comparison.
""",
        encoding="utf-8",
    )

    (OUT / "DEPRECATED_ARTEFACTS.md").write_text(
        """# Deprecated artefacts after FAERS PT repair

No file was deleted. The following artefacts are retained for provenance but must not be used as authoritative results.

| Artefact | Deprecation scope | Replacement/status |
|---|---|---|
| `data/processed/faers_pt_universe.parquet` | Entire PT identity object (`pt_u`, name, stored code) | `faers_meddra28_canonical.csv`; latest case–PT Parquet |
| `data/processed/aact_candidate_pairs.parquet` | PT codes and all downstream pair counts | Approval-independent repaired ceiling exists; B-STRICT still blocked |
| `data/processed/pt_match_flags.parquet` | FAERS/JADER-augmented target mapping | `aact_meddra28_term_mapping.csv` |
| `data/processed/faers_label_pilot_pairs.csv.gz` | All 1/2/3-year labels and prevalences | No valid replacement until FDA dates exist |
| `preflight/05_pt_mapping_audit.md` and `05_pt_*.csv` | Old crosswalk percentages as authoritative Command 02B results | AACT rebuild audit |
| `preflight/06_candidate_cohort_counts.csv` | Cohort counts that rely on non-frozen/proxy temporal logic | Blocked pending FDA cohort |
| `preflight/07_candidate_pair_counts.csv`, `07_cascade_ae_rows.csv`, `07_pair_contribution_stats.csv` | Old drug–PT universe | B-STRICT rebuild blocked |
| `preflight/08_faers_label_pilot.md`, `08_faers_label_pilot_summary.csv`, `08_faers_positives_by_drug.csv` | Technical-anchor labels and prevalence | Explicitly superseded; no scientific replacement yet |
| `preflight/09_jader_validation_audit.md` | Old cohort/pair replication feasibility counts | Repaired-pair replication blocked |
| `preflight_v2/09_jader_v5_pt_mapping_audit.md` | FAERS overlap subsection and 20,329/6,148 preliminary screen only | JADER-internal PT QC remains valid; FAERS subsection replaced by repaired audit |
| Prior `11,313 / 11,551` FAERS↔JADER overlap in V2 reports | Global PT overlap only | Replaced by repaired `02_cross_database_pt_overlap_rebuilt.md` |
| Any old signal-coverage, temporal-split positive, or JADER-replicated-pair result copied outside this repository | All values depending on legacy FAERS PT codes | Must be regenerated after blockers resolve |

The JADER V5 normalized tables and their structural integrity findings are **not deprecated**. MedDRA 28.0 source dictionaries are **not deprecated**.
""",
        encoding="utf-8",
    )

    pass_criteria = {
        "1_all_final_codes_current": "PASS",
        "2_name_code_consistency_100pct": "PASS",
        "3_hierarchy_consistency_100pct": "PASS",
        "4_mapping_provenance_complete": "PASS",
        "5_unresolved_explicit": "PASS",
        "6_old_code_not_used_for_mapping": "PASS",
        "7_bstrict_pair_universe_rebuilt": "BLOCKED",
        "8_faers_labels_rebuilt": "BLOCKED",
        "9_cross_database_overlap_recomputed": "PARTIAL_GLOBAL_PASS_COHORT_BLOCKED",
        "10_jader_replication_recomputed": "BLOCKED",
    }
    decision = {
        "command": "COMMAND_02B_FAERS_MEDDRA_TERMINOLOGY_REPAIR_AND_DOWNSTREAM_REBUILD",
        "status": "CONDITIONAL",
        "scientific_hold": True,
        "terminology_layer": "PASS",
        "latest_case_event_layer": "PASS",
        "aact_canonical_pt_layer": "PASS",
        "b_strict": "BLOCKED",
        "faers_temporal_labels": "BLOCKED",
        "signal_coverage": "BLOCKED",
        "temporal_split": "BLOCKED",
        "jader_pair_replication": "BLOCKED",
        "pass_criteria": pass_criteria,
        "terminology_metrics": term,
        "case_event_metrics": case,
        "aact_metrics": aact,
        "blockers": blockers,
        "performed": {
            "machine_learning": False,
            "time_to_onset": False,
            "jader_death_enrichment": False,
            "proxy_approval_anchor_substitution": False,
        },
        "repair_report": str(OUT),
    }
    (OUT / "FAERS_PT_REPAIR_DECISION.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": decision["status"],
        "pass_criteria": pass_criteria,
        "required_outputs": len(list(OUT.iterdir())),
    }, indent=2))


if __name__ == "__main__":
    main()
