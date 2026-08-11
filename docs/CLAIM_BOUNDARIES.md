# Claim Boundaries

## Approved language

- FAERS disproportionality signal or status;
- reporting signal;
- premarketing representation;
- incremental predictive information/value;
- temporal validation in later-approved drugs;
- cumulative cross-database replication;
- cautious pharmacovigilance prioritization.

## Prohibited language or implication

- predicts ADR risk or predicts adverse reactions;
- adverse-event incidence or patient-level probability;
- causal toxicity or confirmed adverse reactions;
- demonstrated clinical utility or regulatory utility;
- JADER external validation or Japanese temporal validation;
- trials missed 80% of ADRs or 80% novel toxicities;
- zero-shot adverse-event prediction;
- superiority of gradient-boosted trees over penalized logistic regression.

## Representation label

`POSTMARKETING_ONLY` means only that the signal was not represented in the qualifying B-STRICT profile. It does not mean biologically novel, absent from all preapproval evidence, causal, clinically unexpected, or false/true based on cross-database replication.

## Interpretation boundary

Coefficients, TreeSHAP values, ranks, dependence summaries, and domain attribution describe fitted predictive behavior under correlated inputs. They do not identify mechanisms, causal drivers, independently modifiable effects, or a proportion of incremental average precision explained.
