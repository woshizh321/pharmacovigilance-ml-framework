# Pharmacovigilance ML Framework

Premarketing safety profiles and postmarketing reporting-signal prediction.

This repository contains the methods code and study-design documentation for a database-linked pharmacovigilance prediction study. It is prepared as a **code-only release**: no row-level or aggregate analytic datasets, trained models, predictions, SHAP values, figures, publication tables, manuscripts, licensed terminology files, identity crosswalks, or local system configuration are included.

## Study question

Among drug–Preferred Term pairs represented in qualifying premarketing clinical-trial safety profiles, does pair-specific premarketing safety information add predictive information for three-year FAERS disproportionality status beyond regulatory, event-identity, and general trial-programme context, and does that increment transport to later-approved drugs?

The study concerns reporting-signal prediction. It does not estimate adverse-event incidence, individual-patient risk, causal toxicity, or demonstrated clinical/regulatory utility.

## Design at a glance

- Regulatory backbone: FDA CDER novel single-active-moiety approvals, 2012–2022.
- Premarketing evidence: ClinicalTrials.gov/AACT snapshot dated 2026-05-01.
- Primary evidence definition: B-STRICT, requiring actual final study completion on or before first FDA approval and uniquely attributable target-drug monotherapy arms.
- Analytical unit: canonical active moiety × MedDRA 28.0 Preferred Term.
- Primary outcome: three-year FAERS Criterion R (`a≥3` and reporting-odds-ratio lower 95% confidence limit >1).
- Development: drugs approved during 2012–2018.
- Temporal evaluation: drugs approved during 2019–2022.
- Model families: penalized logistic regression and gradient-boosted trees.
- Primary comparison: pair-specific feature set versus context-only feature set within each model family.
- JADER: cumulative cross-database replication only, not Japanese temporal validation.

See [Study design](docs/STUDY_DESIGN.md) and [claim boundaries](docs/CLAIM_BOUNDARIES.md).

## Release boundary

The repository intentionally excludes all scientific data and generated scientific artifacts. Public source databases must be obtained independently from their custodians, and MedDRA terminology access/redistribution remains subject to its license. Local harmonized derivatives and crosswalks are not redistributed here.

The current package should be treated as suitable for a **private GitHub repository pending PI decisions on code license, citation metadata, public visibility, and redistribution terms**. Absence of a `LICENSE` file does not grant reuse rights.

See [Data and release boundaries](docs/DATA_AND_RELEASE_BOUNDARIES.md).

## Repository contents

- `scripts/`: selected repaired/final workflow scripts; preliminary and superseded scripts are omitted.
- `docs/STUDY_DESIGN.md`: stable scientific design and analytical boundary.
- `docs/REPRODUCIBILITY.md`: environment, path configuration, and execution order.
- `docs/DATA_AND_RELEASE_BOUNDARIES.md`: what is intentionally excluded and why.
- `docs/CODE_PROVENANCE.md`: source-to-release script checksums and the only applied sanitization.
- `tools/check_release.py`: local/CI disclosure-boundary audit.
- `.github/workflows/boundary-check.yml`: prevents accidental addition of data, binary artifacts, local paths, or common secrets.

## Environment

The frozen modeling environment used Python 3.14.4 with the principal versions recorded in `requirements.txt`. Auxiliary I/O and plotting dependencies reflect the environment used when preparing this code release; the exact scientific results remain governed by the original frozen project artifacts, which are not distributed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_release.py
```

## Path configuration

Absolute paths from the local analysis environment were replaced in the release copies with `/path/to/project`, `/path/to/Database`, or `/path/to/user`. This was a disclosure-only transformation; algorithmic logic was not intentionally changed.

The scripts are archival workflow code rather than a turnkey package. Before execution, configure an isolated working copy using `config/paths.example.toml` and update the placeholder path constants. Do not edit the frozen source project to configure a public reproduction.

## Reproducibility limits

Running the code requires independently obtained source databases, compatible source schemas/snapshots, and licensed terminology where applicable. The release contains no source data, processed data, fitted model, prediction, or expected-result registry. Therefore, GitHub CI verifies only repository boundaries and code parseability; it does not reproduce or assert scientific results.

See [Reproducibility](docs/REPRODUCIBILITY.md) for the staged workflow and authoritative limitations.

## Data-quality safeguards reflected in the code

- The legacy FAERS PT-code aggregate is defective and must not be reused; repaired MedDRA 28.0 canonical event identity is required.
- Normalized cumulative JADER tables are used; earlier internal indication fields are deprecated.
- Unreported ClinicalTrials.gov adverse events are not biological zeros.
- Exact-title arm attribution is selective; positional arm recovery is not part of the primary analysis.
- Drug-grouped development and a later-drug temporal holdout protect against pair-level leakage.

## Citation and license

The study citation and software license require PI/author approval and are intentionally not guessed. A `CITATION.cff` and `LICENSE` should be added only after those decisions are documented.

## Contact

Corresponding-author details are intentionally omitted pending the final submission metadata lock.
