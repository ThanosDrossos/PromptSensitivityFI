---
name: improve-iclr-background
description: Use when drafting, revising, compressing, or auditing background or preliminaries for an ICLR 2027 paper.
---

# Improve ICLR Background

Background supplies only the established concepts needed to understand the paper’s problem, gap, contribution, evidence, scope, and significance; it must not obscure those links with tutorial material or new method claims. Preserve supported scientific meaning, assumptions, definitions, notation, citations, and local voice; use `write-iclr-2027-paper` for whole-paper consistency or current venue requirements.

## Input check

Determine whether the request is diagnosis, revision or compression, or drafting. Obtain the downstream derivations, methods, experiments, established prerequisites, and notation; identify missing author-supplied context before deciding what can be removed.

## Acceptance-impact audit

1. Treat a standalone background section as conditional; retain only concepts, definitions, notation, and established results needed downstream.
2. Order prerequisites before their use, define every symbol once, and keep notation stable.
3. Separate established material from the paper’s new method and results.
4. Move novelty positioning to related work and implementation choices to method or setup.
5. Remove tutorial exposition that no later claim, derivation, method, or experiment uses, while preserving assumptions and definitions whose removal would change scientific meaning.

## Revise

Compress or reorganize around downstream use, preserving cited established material and required assumptions. Relocate rather than delete necessary new-method content, and separate substantive revisions from evidence gaps requiring author action; never infer missing definitions, assumptions, or results.

## Output

Return `Revised section` only when requested, followed by `Material changes`, `Evidence gaps`, and `Reviewer risks`.
