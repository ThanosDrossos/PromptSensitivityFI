# Reviewer Lens for ICLR 2027

Use this reference to diagnose a draft, prioritize revisions, or simulate an acceptance-oriented review. The official reviewer guide frames the decision around whether the submission contributes new knowledge and sufficient value to the community. State-of-the-art performance is not required.

## The four decision questions

Every submission should let a careful reviewer answer:

1. **What specific question or problem does the paper tackle?** The paper must distinguish the scientific problem from the proposed technique, metric, benchmark, or artifact.
2. **Is the approach well motivated and situated in the literature?** The gap must be supported by fair comparisons to the closest work, not by a straw-man field summary.
3. **Do the results support the claims?** Technical arguments, experiments, baselines, controls, evaluation, statistics, assumptions, and scope must line up with the language of each claim.
4. **What is the significance?** Explain what new knowledge or capability changes for researchers or practitioners, why the result is non-obvious, and where it applies.

Reviewers also look for clarity, technical correctness, experimental rigor, reproducibility, novelty, and relevance. Treat these as coupled: a novel claim that cannot be verified, or a rigorous result whose significance is unclear, remains weak.

## Acceptance-case audit

Create a compact table or internal map with one row per headline contribution:

| Contribution | Novel relative to | Evidence | Scope and assumptions | Likely reviewer objection |
|---|---|---|---|---|

For each row, test:

- Is the contribution a result, method, theory, dataset, benchmark, measurement framework, empirical finding, or synthesis? Do not mix categories in one vague sentence.
- Does the cited closest work actually lack the claimed contribution?
- Is there a result in the paper that isolates the contribution rather than only demonstrating the full system?
- Are the comparison conditions fair and the estimand explicit?
- Are effect sizes, uncertainty, failure cases, and sensitivity analyses sufficient for the claim?
- Could the same observation be explained by leakage, selection, grader behavior, preprocessing, compute, prompt choice, random seeds, model scale, or another confound?
- Does the paper say where the result should and should not generalize?

## Prioritize revisions

Rank issues by their likely influence on the decision:

- **Acceptance-critical:** incorrect or unsupported central claim, unclear novelty, invalid experiment, missing decisive control, anonymity or policy violation, or no clear ICLR-relevant significance.
- **Confidence-limiting:** incomplete robustness, unclear implementation, weak uncertainty reporting, missing scope conditions, or a central presentation obstacle.
- **Polish:** local prose, layout, minor citation, or secondary analysis improvements.

Do not bury the authors under many low-value comments. Identify the smallest set of changes that would most increase confidence in correctness, novelty, rigor, reproducibility, and significance.

## Red-team pass

Before finalizing a major revision, write the strongest plausible reject argument in two or three sentences. Then check whether the main paper contains direct evidence that answers it. If answering it would require new data, experiments, theory, or author judgment, report that requirement rather than masking it with prose.
