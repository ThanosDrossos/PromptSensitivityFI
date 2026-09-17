---
name: improve-iclr-abstract
description: Use when drafting, revising, or auditing an abstract for a machine-learning paper targeting ICLR 2027.
---

# Improve ICLR Abstract

An abstract is the compact acceptance case: it connects the precise problem and gap to the contribution, decision-relevant evidence, its scope, and the resulting significance. Keep the author’s supported scientific meaning, notation, citations, and voice; use `write-iclr-2027-paper` for whole-paper consistency or current venue requirements.

## Input check

Identify whether the request is diagnosis, revision, or drafting. Obtain the paper’s problem, gap, approach or study design, result evidence, and intended implication; name missing author-supplied context rather than filling it in.

## Acceptance-impact audit

1. Check that one self-contained paragraph states the precise problem and gap, the approach or study design, the most decision-relevant supported finding, and its implication. Treat this as a default, not an ICLR structure requirement.
2. Prefer a concrete supplied finding over “extensive experiments”; retain only numbers, datasets, comparators, uncertainty, and scope the authors provide.
3. Remove undefined acronyms, promises, unsupported quantities, and superiority claims without a stated scope. Preserve author citations by default. If citations undermine a self-contained abstract or may conflict with current venue requirements, flag them for author decision and route the venue check to `write-iclr-2027-paper`; remove or relocate citations only when the author requests it or a verified current requirement establishes it.
4. Reconcile the abstract with the introduction, results, and conclusion; flag conflicts rather than silently changing the science.

## Revise

Make the smallest revision that makes the problem-to-evidence chain assessable. Distinguish observed outcomes from interpretation, and calibrate the implication to the evidence. Do not invent a result magnitude, benchmark coverage, comparator, or generalization boundary; report such prose-incurable evidence gaps for author action separately from substantive revisions.

## Output

Return `Revised section` only when requested, followed by `Material changes`, `Evidence gaps`, and `Reviewer risks`.
