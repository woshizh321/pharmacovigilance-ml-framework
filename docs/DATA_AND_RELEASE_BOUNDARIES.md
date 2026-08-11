# Data and Release Boundaries

## Release principle

This is a code-and-design repository. It intentionally contains no scientific data or generated scientific outputs. The default is exclusion unless an artifact has been explicitly reviewed as non-data, nonrestricted, and necessary for understanding the code.

## Included

- selected final/repaired workflow scripts;
- study-design and claim-boundary documentation;
- environment/version documentation;
- path placeholders and execution-order guidance;
- code provenance and disclosure-boundary QC.

## Excluded

- raw FDA, AACT, FAERS, JADER, and MedDRA files;
- every `data/`, `data_external/`, raw, processed, and derived directory;
- row-level and aggregate analysis datasets;
- drug-identity mappings, terminology crosswalks, case/event registries, and cohort tables;
- CSV, Parquet, Excel, XPT, R data, database, archive, and compressed data files;
- fitted models, preprocessing objects, PT mappers, coefficients, trees, and serialized objects;
- native or out-of-fold predictions;
- SHAP/contribution matrices, feature-rank outputs, bootstrap replicates, and calibration-bin data;
- analysis reports containing locked numerical result registries;
- figures, tables, captions, manuscripts, supplements, and submission metadata;
- MedDRA source dictionaries or mapped terminology assets;
- local paths, usernames, machine-resource files, credentials, tokens, certificates, `.env` files, and PKCS#11 configuration.

## Source access

FDA regulatory data, ClinicalTrials.gov/AACT, FAERS, and PMDA JADER source datasets are publicly accessible from their custodians. Public accessibility does not imply that this repository may redistribute local snapshots, harmonized derivatives, or cross-database mappings.

MedDRA files and terminology-derived redistribution must comply with the applicable MedDRA license. This release contains no MedDRA dictionary or crosswalk.

## Identity and privacy boundary

The scientific unit is a drug–Preferred Term pair rather than a person, but spontaneous-report and registry files may still carry fields that should not be republished without review. No source row or patient/case-level record is included. Drug identity crosswalks and ambiguous-name adjudications are also withheld pending PI/data-steward and licensing decisions.

## Result boundary

The release does not include numerical source tables, result registries, figures, or manuscript text. Code may contain method constants, prespecified thresholds, and internal assertion counts needed to document the original workflow; these are not a substitute for scientific source data and must not be treated as a distributed dataset.

## Publication visibility

Until the PI decides repository visibility, code license, citation metadata, and redistribution terms, push this repository to a private GitHub remote. Making a repository public is a separate external-state decision and should occur only after a final release-boundary review.

## Automated protection

Run `python tools/check_release.py` before every commit and in CI. The checker rejects forbidden file types/directories, local absolute paths, oversized files, and common secret formats. This reduces but does not replace human review.
