# Writing and Revision Guide

Use this reference when drafting or revising an ICLR paper. ICLR does not require a single section recipe; choose a structure that makes the claim-evidence chain easy to verify.

## Global argument

Maintain one consistent hierarchy:

- **Problem:** the specific scientific or practical question.
- **Gap:** what the closest literature cannot yet establish or do.
- **Contribution:** the new knowledge, method, theory, dataset, or capability supplied by this work.
- **Evidence:** the analysis that supports each contribution.
- **Scope:** the populations, models, datasets, assumptions, and conditions under which the claim holds.
- **Significance:** what changes if the claim is true.

The title, abstract, introduction, results, discussion, and conclusion should express the same hierarchy at different resolutions. If two sections imply different contributions, fix the argument before line editing.

## Section functions

### Abstract

Write one self-contained paragraph after the core analyses stabilize. Include the problem and gap, the approach or study design, the most decision-relevant result with calibrated specificity, and the implication. Avoid citations, undefined acronyms, promises, and generic claims such as "extensive experiments" without the finding they establish.

### Introduction

Move quickly from context to the precise problem and gap. State contributions as verifiable outcomes, not activities. Pair each contribution with the evidence later used to support it, and explain why it matters to ICLR readers. Do not spend the page budget on a long tutorial that belongs in background.

### Related work

Organize by the distinctions needed to establish novelty. Compare the paper to the closest alternatives on question, assumptions, method, evidence, and scope. Cite primary sources and verify every bibliographic claim. Describe limitations of prior work precisely and respectfully.

### Methods and experimental design

Report enough for validity assessment and reproduction. As applicable, specify:

- task definition, estimand, hypotheses, and decision rules;
- data provenance, licensing, inclusion/exclusion, splits, leakage controls, preprocessing, and selection effects;
- models, exact versions/checkpoints, prompts, decoding, seeds, hyperparameters, training or inference budget, and compute;
- baselines and why they are the right comparators;
- evaluation and grading procedures, annotator or judge validation, reliability, and blinding;
- statistical model, unit of analysis, dependence structure, uncertainty, multiple comparisons, missingness, and robustness checks;
- ablations or interventions that isolate the proposed mechanism;
- human-subject, privacy, bias, safety, environmental, and release considerations.

Justify choices that materially affect the claims. Put derivations, exhaustive parameters, secondary robustness results, and lengthy examples in the appendix when the main text contains enough to judge the headline result.

### Results

Order results by research question or contribution. Start each unit with the question or prediction, report the relevant evidence, and then interpret it. Prefer effect sizes and uncertainty to isolated significance labels. Distinguish confirmatory analyses from exploratory ones, report null or negative findings plainly, and avoid causal language unless the design identifies a causal effect.

Figures and tables should be readable without reconstructing the paper: label axes and units, define abbreviations, state aggregation and uncertainty, identify sample size or unit where useful, and make captions carry the central observation. Refer to every float from the text. Use accessible colors and inspect the compiled PDF at final size.

### Discussion and conclusion

Explain what the findings establish, how they change the literature, and what they do not establish. Give limitations a direction: state which inference may be biased or narrowed and why. Separate evidence-backed implications from hypotheses for future work. The conclusion should synthesize the answer to the research question without adding results.

## Scientific prose

- Prefer direct sentences, stable terminology, and explicit logical relationships.
- Use authorial "we" for actions when compatible with the paper's local voice; use present tense for claims and figures, past tense for completed procedures.
- Define symbols and acronyms at first use. Keep notation identical across prose, equations, figures, and appendix.
- Place citations at the claims they support. Never cite a paper based only on a search snippet or secondary summary.
- Calibrate verbs: "shows" for directly demonstrated results, "supports" for convergent evidence, "suggests" for limited evidence, and "we hypothesize" for explanations not tested.
- Remove hype, vague novelty, rhetorical filler, and anthropomorphic descriptions that obscure the mechanism.

## Nine-page compression

Compression must preserve the acceptance case. Work in this order:

1. Remove repetition and background not needed to understand novelty.
2. Merge overlapping claims and present one authoritative result per point.
3. Replace prose enumerations with compact tables or diagrams only when they improve comprehension.
4. Move secondary analyses, full derivations, extra examples, implementation minutiae, and additional robustness checks to a clearly referenced appendix.
5. Shorten captions and sentences without deleting assumptions, units, uncertainty, limitations, or experimental details needed to assess central claims.

Never reduce margins, font size, line spacing, or official style settings. Recompile after structural edits because float movement can change the page count.

## Revision output

When reporting work to the authors, separate:

- scientific or argumentative changes;
- evidence gaps requiring author action or new analysis;
- wording and layout changes;
- compliance risks;
- checks actually run on the source and compiled artifact.
