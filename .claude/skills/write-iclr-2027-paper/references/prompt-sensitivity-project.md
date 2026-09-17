# Prompt-Sensitivity Manuscript Context

Read this reference only when working on the manuscript in
`../DSI-Seminar-Prompt-Sensitivity-FI---Paper` (GitHub
`ThanosDrossos/DSI-Seminar-Prompt-Sensitivity-FI---Paper`) or the analysis repo
`PromptSensitivityFI` that produces its numbers. These are project observations,
not venue-wide rules. Reinspect the current files because the draft changes
between sessions; the state below was checked on 2026-09-17.

## Manuscript identity and argument

The ICLR version is titled **"Rephrase and Repeat: Separating Accuracy,
Formulation Dependence, and Output Dispersion in LLM Prompts"** and lives on
`main` of the paper repo. Its argument separates **three factors** — mean task
success, formulation dependence, and output dispersion — and manipulates **two
variables**, question specificity (content) and paraphrase-universe width
(form). Formulation dependence is measured by ρ_F, a one-way ICC with decoding
noise subtracted; output dispersion by H_sem; mean task success by graded
accuracy and the FI_in curve. FI_spec quantifies the setting of the specificity
variable and is not a fourth measurement. Hidden-state probe results
(underspecification, dispersion, fragility, correctness) are a secondary
contribution.

The vocabulary is fixed in `CLAUDE.md` of the analysis repo (section "The
frame — use this vocabulary") and takes precedence over any older wording in
drafts, reviews, or LaTeX comments. Do not reintroduce the retired terms:
*competence*, *formulation sensitivity*, *axis/axes*, *indices*, *candidate
metrics*, *dial*, *stress test*, *reliability probe*, *vagueness*. The fourteen
quantities are *metrics*; the trio are *factors*; a member of a paraphrase
universe is a *formulation*; a *cell* is one question at one specificity level
for one model.

When revising this paper, keep these distinctions explicit:

- factors (measured outcomes) versus manipulated variables;
- target-gold versus union-gold scoring;
- formulation-induced variance versus decoding noise;
- algebraic identity, empirical association, and construct validity;
- descriptive evidence, intervention evidence, and causal language.

## Where the numbers come from

Every quantitative claim traces to a committed artifact in the analysis repo:
`data/stats_hygiene.md` (the declared 12-test primary family),
`data/metric_selection.md` (the two-stage metric analysis that selects the
three representatives), and the other `data/*.md` artifacts listed in
`CLAUDE.md`. Those files are script-generated; fix the generating script, not
the markdown. Figures come from
`prompt_sensitivity.scripts.make_paper_figures` and carry no hardcoded
numbers, so never hand-patch a figure. `CLAUDE.md` also lists the corrections
from the 2026-08-27 numbers audit; older drafts and review documents still
carry the superseded versions.

## File and LaTeX conventions

- `main.tex`: preamble, title/author state, abstract, includes, statements,
  bibliography, appendix.
- `main_body.tex`: main sections. Thanos's German `%` comments are his open
  work list; read them as author intent.
- `appendix.tex`: derivations, estimator conventions, metric definitions, and
  additional results.
- `math_commands.tex`, `references.bib`: notation and bibliography.
  `references.bib` keeps per-entry verification comments; preserve them.
- `1_Figures/`: figure assets and the TikZ visual abstract.
- `iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty`,
  `fancyhdr.sty`: official venue files; do not edit.
- `template.md` and `writing_style.md`: local project guidance. Treat them as
  author preferences and technical notes, not as factual or venue authority.

Use the manuscript's existing notation macros (`\FIin`, `\FIout`, `\FIspec`,
`\Hsem`, and `\AUFI`) rather than spelling their definitions repeatedly.
Preserve its `natbib` conventions (`\citep` and `\citet`) and `\tabnote`
table-note macro. Protect math in headings with `\texorpdfstring` where needed.

The paper repo syncs with Overleaf on `main`. Always `git pull` before editing
the tex; Thanos edits in Overleaf between sessions. Grep for `<<<<<<<` before
committing any merge. Build with `tectonic main.tex`. `\iclrfinalcopy` is
commented out for double-blind review; keep it that way. The main text must
fit **9 pages**: compile and measure the actual extent before planning cuts
rather than trusting any page count written in a note.

## Acceptance-critical checks for this paper

Prioritize these questions during review or revision:

1. Are novelty claims about published prompt-sensitivity metrics supported by a
   current, comprehensive, and accurately characterized literature comparison?
2. Does the evidence distinguish arithmetic equivalence among metrics from
   empirical construct validity and from the claimed three-factor structure?
   The component analysis that justifies a representative must not contain
   that representative: Stage 1 decomposes the ten published metrics only,
   Stage 2 adds ρ_F (see `data/metric_selection.md`).
3. Are the hierarchical intraclass-correlation estimand, priors, shrinkage,
   boundary behavior, coverage, recovery simulation, and cross-model
   comparisons reported well enough to audit?
4. Does union-gold scoring fully address the grading lottery, and are residual
   rank or evidence-coverage biases quantified without overstating the
   specificity effect?
5. Do the specificity and width interventions isolate content and form, or
   could paraphrase generation, filtering, judge behavior, sample count, or
   generator replacement explain the dissociation?
6. Are the selection of 150 from 609 eligible questions, model family/scale
   coverage, paraphrase cap, ten-sample decoding design, NLI clustering, and
   judge validity reflected in the scope of all claims?
7. Are multiple analyses, model-by-level strata, uncertainty, robustness
   choices, and exploratory probe claims handled without selective reporting?
8. Is the hidden-state probe essential to the central contribution and
   supported strongly enough for the 9-page main paper, or should it be
   shortened or moved while preserving the acceptance case?
9. Do all quantitative statements in the abstract, introduction, captions,
   results, and conclusion trace to the same current analysis artifacts and
   denominators?
10. Does the AI use statement accurately disclose the real workflow, including
    any assistance used in methods, code, interpretation, literature work,
    writing, or figures?

Do not resolve these questions through stronger prose alone. Ask for analysis
artifacts or author judgment when the source does not contain enough evidence.
