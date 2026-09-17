---
name: improve-iclr-discussion-limitations
description: Use when drafting, revising, restructuring, or auditing discussion and limitations content for an ICLR 2027 paper.
---

# Improve ICLR Discussion and Limitations

Discussion and limitations close the problem–gap–contribution–evidence–scope–significance chain: they explain what findings establish, how they change the literature, and the boundaries that qualify significance. A standalone section is conditional, but scope limits must appear somewhere appropriate. Preserve supplied scientific meaning, notation, citations, and local voice; use `write-iclr-2027-paper` for whole-paper consistency or current venue requirements.

## Input check

Identify diagnosis, revision, or drafting. Obtain the research question, supported findings and uncertainty, contribution and literature context, known limitations, and intended implication. If these are absent, name the missing context; do not turn a limitation into reassuring prose.

## Acceptance-impact audit

1. For each finding, state the demonstrated result, its literature-facing implication, and what it does not establish; distinguish demonstrated implications, plausible interpretation, speculation, and future work.
2. Check deployment, causal, universality, robustness, and generalization claims against the evaluated setting and evidence.
3. For every material limitation, name the affected claim, bias direction or boundary, likely severity, and only author-supplied mitigation or resolving evidence.
4. Test plausible alternative explanations and the strongest reviewer rejection argument; state which supported claims survive each concern.
5. Ensure future work is specific to an unresolved evidentiary question, not generic language that neutralizes a limitation.

## Revise

Organize the discussion from finding to interpretation to qualified significance, then make limitations concrete. Retain demonstrated conclusions while narrowing unsupported extrapolation. Flag prose-incurable gaps—such as absent diverse evaluation, controls, or deployment evidence—for author evidence or judgment; never invent mitigation, severity, or reassurance.

## Output

Return `Revised section` only when requested, followed by `Material changes`, `Evidence gaps`, and `Reviewer risks`.
