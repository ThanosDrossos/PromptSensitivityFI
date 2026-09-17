---
name: improve-iclr-results
description: Use when drafting, revising, restructuring, or auditing the results or analysis section and its figures or tables for an ICLR 2027 paper; not for experimental-design validity before results exist.
---

# Improve ICLR Results

Results provide the evidence link in the paper’s problem–gap–contribution–evidence–scope–significance chain: they state what was observed and what, at most, those observations support. Preserve supplied values, scientific meaning, notation, citations, and local voice; use `write-iclr-2027-paper` for whole-paper consistency or current venue requirements.

## Input check

Identify diagnosis, revision, or drafting. Obtain research questions, result data or author-verified summaries, analysis status, tables and figures, and intended interpretations. Name absent magnitudes, uncertainty, units, or scope rather than writing around the gap.

## Acceptance-impact audit

1. Order evidence by research question or contribution: prediction or question, observation, uncertainty, then calibrated interpretation.
2. Prefer effect sizes and uncertainty over isolated significance labels. A “significant improvement” without exact magnitude, uncertainty or interval, sample or seed count, metric definition, and comparator cannot support a strong claim; retain it only as provisional.
3. Separate confirmatory from exploratory analysis, observations from explanations, and associations from causal claims.
4. Report supplied null or negative findings, relevant robustness and sensitivity checks, and each result’s scope.
5. Check every supplied number against prose, tables, figures, and captions; never recompute or alter a value without underlying data.
6. For each float, check axes, units, aggregation, uncertainty definition, sample size and unit, abbreviations, and text references.

## Revise

Reorganize around question-to-evidence links and narrow conclusions to the measurements provided. Preserve all supplied numbers and label exploratory results. Request missing magnitude, uncertainty, comparison, and scope needed for a claim; do not invent values, robustness, or explanations.

## Output

Return `Revised section` only when requested, followed by `Material changes`, `Evidence gaps`, and `Reviewer risks`.
