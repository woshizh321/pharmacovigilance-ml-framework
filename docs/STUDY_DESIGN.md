# Study Design

## Research question

Among adverse-event pairs represented in qualifying premarketing clinical-trial safety profiles, does pair-specific premarketing safety information improve prediction of three-year postmarketing FAERS disproportionality status beyond regulatory, event-identity, and general trial-programme context, and does that incremental predictive value transport to later-approved drugs?

## Data sources

- FDA CDER novel molecular entity and new therapeutic biologic approval compilation: first original single-active-moiety approvals during 2012–2022.
- ClinicalTrials.gov/AACT snapshot dated 2026-05-01: trial, arm, adverse-event, denominator, and programme characteristics.
- FAERS through 2025 Q4: FDA-anchored postapproval reporting outcomes.
- MedDRA 28.0: canonical Preferred Term and System Organ Class identity.
- Normalized cumulative JADER snapshot dated 2026-07: cumulative cross-database replication.

No source data or licensed terminology is included in this repository.

## Regulatory and premarketing evidence cohort

The regulatory backbone excludes multi-active combination products and uses the first original FDA approval date as the chronological anchor.

B-STRICT is the primary premarketing evidence definition. It requires:

- actual final study completion on or before FDA first approval;
- non-Phase-4 status;
- adverse-event results with valid denominators;
- canonical MedDRA event identity;
- unique normalized-title attribution from result group to design arm;
- target-drug monotherapy;
- exclusion of placebo, control, comparator-only, combination, and non-target arms.

Positional arm recovery is not used. B-STRICT reflects completed qualifying evidence; it does not assert that results were publicly posted by approval.

## Analytical unit and universes

The primary unit is canonical active moiety × MedDRA 28.0 Preferred Term.

The prediction universe contains pairs observed in at least one qualifying B-STRICT target-monotherapy arm among drugs with reliable FAERS identity and the frozen outcome-assessability rule.

The coverage universe contains all qualifying three-year FAERS Criterion-R signals among assessable drugs, including PTs outside the trial-observed prediction universe. `POSTMARKETING_ONLY` means not represented in B-STRICT; it is not a novelty, incidence, causal, or truth-status label.

## FAERS endpoint

FAERS cases use the selected latest case version. Exposure requires the target active moiety as primary suspect. For each drug–PT pair and exact FDA-anchored three-year window, mutually exclusive case-level cells define target-exposed/event-present (`a`), target-exposed/event-absent (`b`), background/event-present (`c`), and background/event-absent (`d`).

Criterion R requires `a≥3` and a reporting-odds-ratio lower 95% confidence limit greater than 1. Consensus additionally requires `IC025>0` and is a robustness endpoint. Disproportionality is not incidence, patient-level risk, or causality.

## Feature architecture

Feature Set 0 contains regulatory attributes, canonical PT/SOC identity, and general premarketing trial-programme context. Feature Set 1 nests Set 0 and adds pair-specific premarketing characteristics covering evidence recurrence/volume, row- and arm-level adverse-event magnitude, serious-event context, cross-trial variation, registry reporting thresholds, and contributing-trial design.

Exact active-moiety identity is a linkage/grouping key, not a predictor. FAERS/JADER outcome, report-volume, ROR, IC, calibration, and replication variables are excluded from predictors.

## Modeling and validation

Development is restricted to drugs approved during 2012–2018. Five outer active-moiety-grouped folds and four grouped inner folds are used. Imputation, transformation, standardization, categorical vocabularies, rare-PT mapping, tuning, and fitting occur within the relevant training partition.

The two model families are elastic-net penalized logistic regression and gradient-boosted trees. The scientific contrast is Feature Set 1 versus Feature Set 0 within each family; no family winner is selected.

Temporal evaluation uses drugs approved during 2019–2022. Full-development pipelines and native predictions are frozen before outcome joining. No refitting, retuning, threshold optimization, or recalibration is performed. Uncertainty is clustered by active moiety.

## Interpretation

Primary model interpretation is out-of-fold/out-of-drug. Encoded contributions are aggregated to conceptual predictors by signed within-row summation. The cross-model-supported set is defined by the intersection of the top pair-specific predictors in both model families. Feature/domain attribution describes fitted predictive behavior under correlated predictors and is noncausal.

## JADER role

The normalized cumulative JADER data use normalized tables, distinct case ID, target-drug primary-suspect exposure, and canonical PT identity. They do not require FAERS-style case-version deduplication. JADER provides cumulative cross-database reporting consistency only because a validated Japanese approval/market-entry anchor was not incorporated.

## Scientific boundary

Approved interpretations concern premarketing representation, reporting signals, incremental predictive information, later-drug temporal validation, and cumulative cross-database replication. The study does not establish adverse-reaction incidence, causal toxicity, patient-level risk, clinical utility, regulatory utility, Japanese temporal validation, or model-family superiority.
