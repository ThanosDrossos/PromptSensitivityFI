---
name: improve-iclr-method
description: Use when drafting, revising, restructuring, or auditing the method or approach section of an ICLR 2027 paper; not for experimental-protocol requests focused on datasets, baselines, metrics, statistics, or evaluation design.
---

# Improve ICLR Method

The method makes the contribution assessable: it connects the problem and gap to a specified mechanism whose evidence, scope, and significance can be judged. Preserve supported scientific meaning, notation, citations, and local voice; use `write-iclr-2027-paper` for whole-paper consistency or current venue requirements.

## Input check

Identify diagnosis, revision, or drafting. Obtain the problem setting, contribution, existing notation, implementation facts, and intended claims. State missing author-supplied context rather than inferring it. Keep experimental configuration in the experimental setup unless it is necessary to define the conceptual method.

## Acceptance-impact audit

1. Define the setting, inputs, outputs, learned and fixed components, objective, procedure, and assumptions. Flag an undefined objective, interface, or assumption as an evidence gap.
2. Stabilize every symbol across prose, equations, algorithms, figures, and appendix; make the routing or decision rule and each loss term assessable.
3. Separate training from inference: specify optimization and data flow during training, then inference behavior, decisions, and failure conditions.
4. Explain material component interactions and why choices support the central claim; distinguish the conceptual method from experimental configuration.
5. State complexity, compute, memory, latency, or deployment requirements when they affect feasibility or comparisons. Mark missing implementation facts, hyperparameters, or compute choices for author confirmation rather than guessing.

## Revise

Reorder from setting and interfaces through objective and procedure to assumptions and operational consequences. Make the smallest substantive revision that preserves supplied notation and claims. Do not invent loss terms, hyperparameters, algorithms, training behavior, or inference semantics; flag those prose-incurable gaps for author action.

## Output

Return `Revised section` only when requested, followed by `Material changes`, `Evidence gaps`, and `Reviewer risks`.
