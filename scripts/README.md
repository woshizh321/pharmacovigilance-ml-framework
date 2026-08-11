# Script Map

The scripts are selected from the repaired/final workflow. Preliminary scripts `p01`–`p07`, JADER-v4 delta code, manuscript drafting/QC scripts, and all data/result artifacts are intentionally excluded.

## Terminology/source repair

- `p08_jader_v5_repreflight.py`: structural JADER v5 audit.
- `p10_faers_meddra28_repair.py`: canonical MedDRA 28.0 mapping utilities.
- `p11_rebuild_faers_case_pt.py`: latest-case FAERS case–PT rebuild.
- `p12_rebuild_aact_meddra28_ceiling.py`: AACT terminology rebuild.
- `p13_finalize_faers_pt_repair_hold.py`: terminology repair decision/QC closure.

## Regulatory cohort and preflight

- `p14_build_fda_regulatory_cohort.py`: FDA first-approval cohort.
- `p15_build_drug_identity_master.py`: deterministic cross-source drug identity.
- `p16_build_bstrict_and_arm_audit.py`: B-STRICT and exact-title arm audit.
- `p17_build_fda_anchored_faers_labels.py`: FDA-anchored FAERS endpoints.
- `p18_assessability_coverage_splits.py`: assessability, coverage, and temporal splits.
- `p19_jader_v5_replication_rebuild.py`: JADER v5 cumulative replication.
- `p20_close_preflight_v2.py`: preflight closeout checks/report.

## Scientific Sections 1–6

- `s01_section1_analysis.py`: cohort/characteristics lock.
- `s02_section2_coverage.py` and `s02_command07_targeted_amendment.py`: coverage and corrected decomposition.
- `s03a_feature_matrix_and_protocol.py`: feature registry/leakage audit.
- `s03b_nested_training_and_freeze.py`: grouped nested development modeling.
- `s03b_finalize_from_saved_outputs.py`: saved-output finalization helper.
- `s03c_preholdout_lock.py`: final model/preprocessing lock.
- `s04a_holdout_feature_scoring.py`: pre-outcome holdout scoring.
- `s04b_holdout_outcome_evaluation.py`: one-time outcome join/evaluation.
- `s04c_finalize_section4_qc.py` and `s04d_audit_section4_outputs.py`: temporal result QC/audit.
- `s05_cross_model_interpretation.py` and `s05_audit_interpretation_outputs.py`: OOF/out-of-drug interpretation and audit.
- `s06_cross_database_robustness.py` and `s06_audit_robustness_outputs.py`: JADER/robustness and audit.

## Publication rendering

- `s17_publication_assets.py` and `s17_visualization_qc.py`: figures and visual provenance QC.
- `s19_table_qc.py`: publication-table QC.
- `s20_visual_lock.py`: final visual/source lock.

See `docs/REPRODUCIBILITY.md` before attempting execution. Sanitized paths must be configured in an isolated private working copy.
