# GitHub Release QC

**Status:** PASS  
**Prepared from:** private frozen source project
**Release root:** curated standalone repository

## Scope

This QC covers the curated code-and-design release only. It does not validate or reproduce scientific results because data, models, predictions, result registries, figures, tables, and manuscripts are intentionally absent.

## Included content

- 31 selected repaired/final Python workflow scripts.
- Study-design, reproducibility, claim-boundary, and disclosure-boundary documentation.
- Exact package version inventory for the frozen modeling environment plus release-preparation dependencies.
- Source-to-release code checksum provenance.
- Fail-closed disclosure checker and GitHub Actions boundary workflow.

## Exclusion checks

- PASS — no data/raw/processed/derived directory.
- PASS — no CSV, TSV, Parquet, Excel, XPT, R-data, database, archive, or compressed data file.
- PASS — no trained model, serialized preprocessing object, prediction, SHAP matrix, bootstrap replicate, or calibration source data.
- PASS — no figure, publication table, manuscript, supplement, or submission metadata.
- PASS — no MedDRA dictionary, terminology crosswalk, drug-identity crosswalk, or source snapshot.
- PASS — no `.env`, PKCS#11 configuration, local machine-resource file, certificate, or private key.
- PASS — no file exceeds 5 MiB; package size is approximately 1 MiB before Git metadata.

## Content checks

- PASS — no OpenAI-style key, GitHub token, AWS key, private-key block, or quoted secret assignment detected.
- PASS — no local macOS user-home path remains in released scientific scripts or documentation.
- PASS — all Python files parse/compile when bytecode cache is directed to a writable temporary directory.
- PASS — `python tools/check_release.py` passes across all release text files.
- PASS — code provenance records the frozen source and sanitized release SHA-256 for all 31 scripts.
- PASS — original frozen project scripts were not edited; only release copies were path-sanitized.

## Scientific boundary checks

- PASS — README/design documentation preserves the reporting-signal/noncausal claim boundary.
- PASS — JADER is described as cumulative replication only.
- PASS — legacy FAERS PT-code and JADER v4 warnings are present.
- PASS — AACT nonreporting, reporting-threshold, selective attribution, and no-positional-recovery warnings are present.
- PASS — no exact result registry or patient/case-level data is disclosed.

## Repository-policy checks

- PASS — `.gitignore` excludes scientific data, models, predictions, generated outputs, manuscripts, local configuration, and common secrets.
- PASS — CI runs the release-boundary checker and Python parse check only; it does not run scientific analyses.
- PASS — `LICENSE` and `CITATION.cff` are intentionally absent pending PI/author decisions.
- PASS — the GitHub repository remains private; no public visibility, license, or reuse rights are implied.

## Disposition

The private GitHub repository is `woshizh321/pharmacovigilance-ml-framework`; the generic-name commit is synchronized to `main` and the matching feature branch. No public visibility change, license, or redistribution permission is implied by this synchronization.
