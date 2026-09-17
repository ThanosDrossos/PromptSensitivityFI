---
name: improve-iclr-theory
description: Use when drafting, revising, or auditing definitions, assumptions, propositions, theorems, proof sketches, or theoretical interpretation in an ICLR 2027 paper.
---

# Improve ICLR Theory

Theory supplies a conditional link from the paper’s mechanism and assumptions to a formal guarantee, constraining how its evidence, scope, and significance may be interpreted. Preserve supported scientific meaning, notation, citations, and local voice; use `write-iclr-2027-paper` for whole-paper consistency or current venue requirements.

## Input check

Identify diagnosis, revision, or drafting. Obtain definitions, assumptions, statement dependencies, proof material, and the empirical claims interpreted through the result. Name missing author-supplied context or proof steps; do not turn intuition into a derivation.

Theory content and a standalone theory section are optional: do not introduce or retain one unless the contribution needs formal claims. When only a small definition or argument is necessary, integrate it into the relevant method or background section instead.

## Acceptance-impact audit

1. Treat every theoretical result as conditional. Establish definitions and assumptions before dependent propositions or theorems.
2. Make domains, quantifiers, probability spaces, asymptotic regimes, randomness, and dependencies explicit where applicable; keep notation and terms such as robustness stable.
3. Check that the formal statement matches the proof: distinguish an oracle, optimized quantity, or idealized condition from a learned or deployed system.
4. State precisely what informal prose and empirical interpretation inherit from the guarantee, no more. Separate proved facts, proof sketches, conjectures, intuition, and empirical evidence.
5. Decide whether proof detail required to trust the headline claim belongs in the main paper; identify the appendix dependency plainly.

## Revise

Order definitions, assumptions, statements, and proof support so each claim has a visible basis. Scope prose guarantees to the theorem’s actual regime and label conditional conclusions. Flag proof gaps, missing dependencies, or inconsistent assumptions for author action; never fabricate derivations or silently strengthen assumptions.

## Output

Return `Revised section` only when requested, followed by `Material changes`, `Evidence gaps`, and `Reviewer risks`.
