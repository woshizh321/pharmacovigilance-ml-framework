# Reproducibility Guide

## Scope

This release documents the original workflow but does not distribute the data or frozen result artifacts needed to rerun it. Reproduction requires lawful independent access to source databases, compatible snapshots/schemas, and licensed terminology. Do not substitute convenient proxy data or legacy PT-code assets.

## Environment

- Python: 3.14.4 for the frozen modeling environment.
- Principal package versions: `requirements.txt`.
- The modeling versions recorded by the frozen specification are NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, XGBoost 3.4.0, and joblib 1.5.3.
- Auxiliary I/O/plotting dependencies were captured during release preparation and are not presented as a separate scientific environment lock.

## Paths

The release scripts use sanitized placeholders such as `/path/to/PDS` and `/path/to/Database`. `config/paths.example.toml` is a documentation inventory; scripts do not automatically read it. Configure path constants only in an isolated reproduction copy.

Do not point a reproduction script at the source project or any raw-data directory with write permissions. Use separate, private output directories.

## Workflow stages

### A. Terminology and source preflight

1. `p08_jader_v5_repreflight.py`
2. `p10_faers_meddra28_repair.py`
3. `p11_rebuild_faers_case_pt.py`
4. `p12_rebuild_aact_meddra28_ceiling.py`
5. `p13_finalize_faers_pt_repair_hold.py`

These scripts repair/verify canonical terminology. Earlier preliminary scripts and JADER-v4 comparisons are deliberately omitted.

### B. FDA-anchored cohort and outcomes

1. `p14_build_fda_regulatory_cohort.py`
2. `p15_build_drug_identity_master.py`
3. `p16_build_bstrict_and_arm_audit.py`
4. `p17_build_fda_anchored_faers_labels.py`
5. `p18_assessability_coverage_splits.py`
6. `p19_jader_v5_replication_rebuild.py`
7. `p20_close_preflight_v2.py`

Identity matching must remain deterministic and outcome-independent. Ambiguous names are excluded rather than forced.

### C. Closed scientific sections

1. `s01_section1_analysis.py`
2. `s02_section2_coverage.py`
3. `s02_command07_targeted_amendment.py`
4. `s03a_feature_matrix_and_protocol.py`
5. `s03b_nested_training_and_freeze.py`
6. `s03b_finalize_from_saved_outputs.py` when finalizing already saved outputs only
7. `s03c_preholdout_lock.py`
8. `s04a_holdout_feature_scoring.py`
9. `s04b_holdout_outcome_evaluation.py`
10. `s04c_finalize_section4_qc.py`
11. `s04d_audit_section4_outputs.py`
12. `s05_cross_model_interpretation.py`
13. `s05_audit_interpretation_outputs.py`
14. `s06_cross_database_robustness.py`
15. `s06_audit_robustness_outputs.py`

The original execution used explicit scientific authorizations and lock transitions. Do not treat this order as authorization to rerun analyses.

### D. Publication rendering

1. `s17_publication_assets.py`
2. `s17_visualization_qc.py`
3. `s19_table_qc.py`
4. `s20_visual_lock.py`

Generated figures and tables are not included in this release.

## Path-sanitization provenance

The public-release copies were made from the frozen source scripts. Only local absolute path prefixes were mechanically replaced with neutral placeholders. `docs/CODE_PROVENANCE.md` records source and release hashes. No claim is made that the sanitized copies reproduce the original hashes.

## Validation before use

At minimum:

```bash
python tools/check_release.py
python -m compileall -q scripts tools
```

Scientific reproduction additionally requires source-schema checks, cohort/denominator reconciliation, terminology QC, protected holdout chronology, and comparison with an authorized result registry. Those registries are not public in this package.

## What must not be substituted

- Do not use the legacy FAERS PT-code object.
- Do not use JADER v4 indication fields or a JADER master cross-product as cases.
- Do not encode unreported AACT adverse events as biological zero.
- Do not use positional arm recovery in the primary B-STRICT analysis.
- Do not random-split drug–PT pairs across development and validation.
- Do not use holdout outcomes to change features, models, tuning, calibration, or interpretation rules.
