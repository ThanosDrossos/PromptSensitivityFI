---
name: improve-iclr-experimental-setup
description: Use when drafting, revising, designing, or auditing experimental setup or evaluation protocol for an ICLR 2027 paper; not for requests primarily about reporting already-computed findings.
---

# Improve ICLR Experimental Setup

Experimental setup turns the paper’s contribution into credible evidence: it specifies how each claim is tested, what the resulting evidence can establish, its scope, and its significance. Preserve supported scientific meaning, notation, citations, and local voice; use `write-iclr-2027-paper` for whole-paper consistency or current venue requirements.

## Input check

Identify diagnosis, revision, or drafting. Obtain research questions and claims, planned comparisons, available data and artifacts, and author-verified protocol details. Name missing context instead of treating “standard practice” as a specification.

## Acceptance-impact audit

1. Map every central claim or research question to the comparison, ablation, control, or analysis able to test it. Prioritize confounds that could invalidate the central claim over missing minutiae.
2. Define task and estimand or hypothesis; data provenance, licensing, splits, inclusion/exclusion, preprocessing, and concrete leakage controls.
3. Record exact model, checkpoint, and tool versions; relevant prompts and decoding; hyperparameters, seeds, training and inference budgets, and compute.
4. Justify baselines and enforce fair data, tuning, and compute budgets. Treat unmatched compute or test-set prompt tuning as acceptance-critical threats, requiring matched budgets and a validation-only selection protocol.
5. Specify metrics, unit of analysis, dependence, uncertainty, multiple comparisons, missingness, robustness, and decision rules where applicable.
6. For human or automated judging, specify annotator or judge validation, reliability, blinding, and grading procedures where applicable.

## Revise

Organize the protocol by claim and its threat controls, then make comparison and analysis decisions reproducible. Preserve supplied choices; flag absent versions, budgets, selection rules, leakage defenses, or statistical details for author action. Do not invent datasets, baselines, seeds, controls, or protocol outcomes.

## Output

Return `Revised section` only when requested, followed by `Material changes`, `Evidence gaps`, and `Reviewer risks`.
