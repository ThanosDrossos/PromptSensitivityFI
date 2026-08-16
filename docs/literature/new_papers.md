# New literature scan (Group F) — 2025/2026 work adjacent to FI-for-prompts

Search date: 2026-08-02. Method: WebSearch + WebFetch over arXiv abstract pages, PNAS/Nature/PubMed,
Semantic Scholar. Three axes were probed separately:

- **(a)** Szostak/Hazen *functional information* applied to LLMs, prompts, or AI systems
- **(b)** ICC-style / variance-decomposition measures of prompt sensitivity
- **(c)** "prompt specificity" / "question ambiguity" effects on LLM accuracy with an
  information-theoretic dial

**Bottom line up front: the novelty claim survives, but it must be narrowed.** No work was found that
applies Szostak–Hazen functional information (I = −log₂ F(E_x)) to prompts or LLMs — axis (a) is clean.
Axes (b) and (c) are *not* clean: several 2025–2026 papers decompose LLM evaluation variance into
item vs. prompt components, and at least one measures prompt specificity against accuracy. Claims of
the form "we are the first to quantify prompt sensitivity via variance decomposition" or "the first
to sweep a specificity dial" would now be false. See "Novelty threat assessment" at the bottom for
exact wording recommendations.

**Sharpest single finding: Cox et al., "Mapping from Meaning," is not a 2025 preprint but a published
AAAI-25 paper, and it defines a "Prompt Sensitivity Ratio" ρ_u that is structurally the same object as
ρ_F** — a normalized share of total uncertainty attributed to across-paraphrase variance. It is the
nearest prior art in the entire reference list and needs explicit differentiation, not just a
citation. Details in axis (b), item 0.

---

## Axis (a): functional information applied to LLMs / prompts / AI — NOTHING FOUND

Multiple query formulations were tried (functional information + prompts; "law of increasing
functional information" + AI/LLM; Szostak/Hazen citation + prompt engineering; −log₂ F +
language models; Semantic Scholar recency sort). **No paper applies the Szostak–Hazen construct to
prompts, LLMs, or AI evaluation.** The only 2025–2026 uses of the term outside biology are in
physics/materials, and neither touches AI:

### 1. Functional Information in Quantum Darwinism: An Operational Measure of Classical Objectivity
- https://arxiv.org/abs/2509.17775
- Ports functional-information-style reasoning into quantum decoherence / objectivity. Domain is
  quantum foundations; no AI, no prompts.
- **Threat: none.** Useful only as evidence that the FI construct is being exported out of biology
  into other fields — i.e., a mild *supporting* citation for "porting FI to a new domain is a
  recognized move."

### 2. Function, Complexity and Thermodynamics in Adaptive and Intelligent Soft Matter Systems: An Information-Theoretical Framework
- https://arxiv.org/abs/2605.19795 — George S. Attard, May 2026
- Defines its own three-metric information framework (I₁, I₂, I₃) for classifying responsive vs.
  adaptive vs. "intelligent" materials, benchmarking 16 systems against the Landauer–Bennett limit.
  Despite the word "intelligent," the scope is soft matter, memristors, shape-memory alloys — not AI.
  On inspection it does **not** adopt the Szostak–Hazen −log₂ F(E*) formalism.
- **Threat: none.** Note the terminology collision only: "I₂" here means something unrelated.

> Caveat on this axis: search-engine coverage of very recent arXiv preprints is imperfect, and a
> negative result from web search is weaker evidence than a systematic citation-graph query. If the
> novelty claim is load-bearing for the grade, it is worth one manual pass over the "cited by" lists
> of Hazen et al. 2007 and Wong et al. 2023 on Google Scholar / Semantic Scholar, filtered to cs.*.

---

## Axis (b): ICC-style / variance-decomposition measures of prompt sensitivity — REAL OVERLAP

This is the axis that has moved fastest since late 2025 and it is where the paper is most exposed.

### 0. Cox et al., Mapping from Meaning (AAAI-25)  ← NEAREST PRIOR ART TO ρ_F. READ THIS ONE FIRST.
- https://ojs.aaai.org/index.php/AAAI/article/view/34540 — Cox, Xu, Han, Xu, Li, Hsu, Chen, Gerych,
  Ding. **AAAI-25, Vol. 39 No. 22, pp. 23696–23703**, DOI 10.1609/aaai.v39i22.34540.
- **Venue correction:** this is a *published AAAI-25 paper*, not a 2025 preprint — the arXiv posting
  (2510.17028) actually postdates the AAAI publication.
- What it defines (from the abstract/body): a **decomposable embedding-variance uncertainty metric**
  U_t = tr(Σ_t), computed in sentence-embedding space rather than over discretized answer classes,
  which splits additively into an intra-sample (aleatoric) part U_a = tr(Σ_a) and an
  inter-sample/across-paraphrase (epistemic) part U_e = tr(Σ_e), so U_t = U_a + U_e. On top of that
  decomposition they define the **Prompt Sensitivity Ratio ρ_u = (U_e + ε)/(U_t + 2ε)**, described in
  the paper as quantifying the proportion of total uncertainty attributable to epistemic uncertainty
  "and thus how prompt-sensitive a model is at a given semantic concept."
- **So: yes — it is explicitly an ICC-like variance-ratio decomposition whose numerator is the
  across-paraphrase (prompt/semantic) variance component.** Structurally this is the same object as
  ρ_F: a normalized share of total variability attributed to prompt phrasing.
- **Threat: HIGHEST of anything found — but to the ρ_F *headline*, not to the FI framing.** The
  differences that remain genuinely yours: their variance lives in **embedding space** over
  paraphrases of a single question and is a **continuous trace-of-covariance**, whereas ρ_F is built
  on **functional information in bits** (fraction of the prompt space clearing a functional
  threshold), which gives interpretable units and an ability curve rather than a unit-free ratio.
  **Action required: this must be cited and explicitly differentiated in the related-work section.**
  Presenting ρ_F as the first prompt-sensitivity variance share would not survive review. Also note
  the notational collision — their ratio is ρ_u, yours is ρ_F; keep the subscripts distinct and
  define both if you discuss them side by side.

### 3. Brittlebench: Quantifying LLM robustness via prompt sensitivity  ← MOST IMPORTANT NEW PREPRINT
- https://arxiv.org/abs/2603.13285 — Romanou, Ibrahim, Ross, Shaib, Oktar, Bell, Ovalle, Dodge,
  Bosselut, Sinha, Williams (Feb 2026, v2 Apr 2026). Meta AI / EPFL / AI2 author list.
- Introduces a theoretical framework for "brittleness" that **explicitly disentangles data-induced
  difficulty from prompt-related variability** — a random-effects ANOVA with semantics-preserving
  perturbations nested within data items. Applies it across frontier open-weight and commercial
  models; finds perturbations account for **up to half of a model's performance variance**, and that
  a single perturbation flips model rankings in 63% of cases.
- **Why it matters: this is the closest prior art to a variance-decomposition sensitivity headline.**
  It is the same statistical skeleton as an ICC (between-item vs. within-item/perturbation variance),
  just not labelled ICC. It is also a strong, well-resourced, highly citable paper — a reviewer in
  this area will know it.
- **Threat: HIGH for any "first to decompose prompt vs. item variance" claim. Zero threat to the FI
  framing.** Brittlebench has no functional-information construct, no specificity dial, and no
  ability/capacity reading — it stops at "how much variance is prompt-attributable." Cite it as
  related work and position the FI contribution as the *interpretable-units* upgrade (bits of
  functional information, ability curve) over a bare variance ratio.

### 4. Stochasticity in Agentic Evaluations: Quantifying Inconsistency with Intraclass Correlation
- https://arxiv.org/abs/2512.06710 — Mustahsan, Lim, Anand, Jain, McCann (Dec 2025)
- Uses **ICC explicitly**: "ICC decomposes observed variance into between-query variance (task
  difficulty) and within-query variance (agent inconsistency)." Reports ICC 0.304–0.774 on GAIA and
  FRAMES depending on task type and model.
- Important distinction in your favour: the within-query term here is **sampling/agent stochasticity
  at fixed input**, not variation across prompt phrasings. It is an ICC for *run-to-run* noise, not
  for *prompt* noise.
- **Threat: MODERATE.** It establishes ICC-for-LLM-evaluation as prior art, so "we introduce ICC to
  LLM evaluation" is unavailable. "We apply an ICC-style decomposition across *prompt phrasings*
  rather than repeated samples" is still defensible — but make the distinction explicitly in text,
  because a reader who knows this paper will otherwise assume you are re-deriving it.

### 5. The Unsampled Truth: Psychometrics in SLMs Measure Prompt Artifacts, Not Psychological Constructs
- https://arxiv.org/abs/2606.03357 — Schwager, Hau, Münker, Rettinger (June 2026)
- Distance-based variance decomposition (PERMANOVA) separating **semantic signal from prompt
  artifacts** across 13 open-weight models (0.6B–14B), varying personas, instructions, items and
  option symbols. Headline: artifactual variance frequently *overpowers* the semantic signal.
- **Threat: LOW-MODERATE.** Different domain (psychometric self-report, not QA accuracy) and a
  different statistic (PERMANOVA on distances, not an ICC on scores), but it is the same conceptual
  move: partition output variance into "what we meant to measure" vs. "prompt form." Worth one
  sentence in related work; it also supplies a nice supporting quote that prompt-form variance can
  dominate.

### 6. Understanding the Prompt Sensitivity
- https://arxiv.org/abs/2604.18389 — Yang Liu, Chenhui Chu (Apr 2026)
- Theoretical: first-order Taylor expansion treating the LLM as a multivariate function, deriving a
  Cauchy–Schwarz upper bound on log-probability differences between semantically equivalent prompts.
  Shows LLMs *disperse* rather than cluster similar inputs. Notably finds that **prompt templates
  exert greater influence on logits than the questions themselves**, and that the bound correlates
  with the existing PromptSensiScore metric.
- **Threat: LOW.** White-box/logit-level and analytic; complementary rather than competing. But the
  template-dominates-question finding is a *confound warning* worth engaging with: if template
  variance swamps question variance, a sensitivity metric computed over paraphrases needs to state
  clearly which of the two it is varying.

### 7. Format Sensitivity Index (FSI) / Parseability Sensitivity Index (PSI)
- https://arxiv.org/abs/2607.09665 — Deep Pankajbhai Mehta (2026; listed date and arXiv id are
  mutually inconsistent on the abstract page — treat the id as authoritative and re-check before
  citing)
- Token-controlled protocol over 140k generations, 7 QA tasks, 5 wrapper families, 4 instruct models
  (7B–72B). FSI = accuracy range induced by wrapper choice; PSI = corresponding range in answer
  parseability. FSI varies 30-fold across models, mostly driven by *compliance failures*.
- **Threat: LOW to the metric, but HIGH as a methodological objection.** The finding that most
  measured "sensitivity" is actually answer-parsing failure is a confound any prompt-sensitivity
  paper must rule out. Recommend a sentence stating how parse failures are handled in the FI
  pipeline (counted as wrong? excluded?), because a reviewer can use this paper to argue a
  sensitivity result is really a formatting-compliance result.
- Naming collision to be aware of: "FSI" vs. "FI" are one letter apart in the same subfield.

### 8. Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking
- https://arxiv.org/abs/2604.11581 (Apr 2026) — not fetched in depth
- Surfaced repeatedly under variance-decomposition queries; appears to quantify how much variance in
  prompt-level metrics is attributable to model identity vs. prompt identity via a two-way
  fixed-effects decomposition.
- **Threat: to be assessed.** Flagged for follow-up; if it really does a two-way model × prompt
  decomposition it belongs alongside Brittlebench in related work.

---

## Axis (c): prompt specificity / ambiguity as a dial — PARTIAL OVERLAP

### 9. DETAIL Matters: Measuring the Impact of Prompt Specificity on Reasoning in LLMs
- https://arxiv.org/abs/2512.02246 — Olivia Kim (Dec 2025). **This reference is real** (it was on the
  to-verify list with a question mark).
- DETAIL = "Degree of Explicitness in Textual Assembly for Instruction-based LLM Reasoning."
  Generates multi-level prompts with GPT-4, **quantifies specificity via perplexity**, scores
  correctness by GPT-based semantic equivalence, over 30 novel reasoning tasks with GPT-4 and
  o3-mini. Finds specificity improves accuracy, most strongly for smaller models and procedural tasks.
- **Threat: MODERATE — this is the direct competitor on the specificity axis.** It already does
  "sweep a specificity dial, measure accuracy, quantify the dial information-theoretically."
  The differentiator to lean on: DETAIL's dial is **perplexity of the prompt** (a property of the
  text under a model), whereas an FI dial is **−log₂ of the fraction of the prompt space clearing a
  functional threshold** (a property of the task/answer-set geometry). Those are genuinely different
  quantities and the paper should say so in one explicit sentence.

### 10. More Than a Score: Probing the Impact of Prompt Specificity on LLM Code Generation
- https://arxiv.org/abs/2508.03678 (Aug 2025) — specificity dial in the code-generation domain.
- **Threat: LOW** (different domain, no information-theoretic dial), but it plus DETAIL establishes
  "prompt specificity affects accuracy" as a known result, not a finding. Frame specificity results
  as *calibration of the dial*, not as a discovery.

### 11. Testing LLMs on Code Generation with Varying Levels of Prompt Specificity
- https://arxiv.org/abs/2311.07599 (2023) — earlier prior art on the same idea, again code-gen.

### 12. Localizing Prompt Ambiguity in LLMs with Probe-Targeted Attribution
- https://arxiv.org/html/2606.05486 (June 2026) — attributes downstream errors to ambiguity *in the
  input* rather than to missing model competence, localizing which prompt spans carry the ambiguity.
- **Threat: LOW.** Complementary — it localizes ambiguity, it does not measure the accuracy response
  to a specificity dial. Useful as support for the premise that ambiguity, not capability, drives a
  meaningful share of errors (a claim the FI framing depends on).

### 13. Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations
- https://arxiv.org/abs/2510.12699 (Oct 2025) — calibrates how open-ended a prompt is, i.e. how large
  the admissible output set is.
- **Threat: LOW-MODERATE, but conceptually the nearest neighbour to FI on the output side.**
  "Size of the set of acceptable outputs" is the reciprocal intuition to "fraction of configurations
  achieving function." If any single paper is going to be raised as "isn't this the same idea?",
  this is a likely candidate. Worth reading properly before submission.

### 14. Prompt-Dependent Ranking of LLMs with Uncertainty Quantification
- https://arxiv.org/html/2603.03336 (Mar 2026) — rankings conditional on prompt, with a *Specificity*
  category shown to alter both predicted ranks and their uncertainty.
- **Threat: LOW.** Supports the motivation (specificity shifts conclusions about models) without
  competing on the metric.

---

## Also surfaced, worth knowing

- **Promptception: How Sensitive Are Large Multimodal Models to Prompts?** —
  https://arxiv.org/abs/2509.03986 (Sep 2025). Multimodal extension of the sensitivity literature.
- **Structured Prompts Improve Evaluation of Language Models** —
  https://arxiv.org/abs/2511.20836 (Nov 2025).
- **Measuring what Matters: Construct Validity in Large Language Model Benchmarks** —
  https://arxiv.org/abs/2511.04703 (Nov 2025). Construct-validity framing; a good citation for
  "single-prompt accuracy is not a valid measurement of the construct."
- **What Single-Prompt Accuracy Misses: A Multi-Variant Reliability Audit of Language Models** —
  https://arxiv.org/abs/2605.02038, Karmakar & Chatterjee (May 2026). 15 open-weight models,
  5 benchmarks × 5 prompt variants; reports accuracy, token-probability calibration,
  verbal-confidence calibration, parse rate, and prompt-perturbation spread. Checked directly:
  **does not** define a fraction-of-variants-above-threshold metric and **does not** cite functional
  information. Safe, and a clean citation for "single-prompt accuracy is insufficient."

---

## Novelty threat assessment — recommended claim wording

| Claim | Status |
|---|---|
| "No prior work applies Szostak–Hazen functional information to prompts / LLMs." | **HOLDS.** Nothing found on any query formulation. Recommend keeping the claim but hedging to "to the best of our knowledge" and, ideally, backing it with one manual citation-graph check of Hazen 2007 / Wong 2023. |
| "We are the first to decompose LLM evaluation variance into item vs. prompt components." | **FALSE — do not claim.** Brittlebench (2603.13285) does exactly this and is prominent. |
| "We are the first to use an ICC-style ratio in LLM evaluation." | **FALSE — do not claim.** Mustahsan et al. (2512.06710) use ICC explicitly, though for run-to-run rather than across-prompt variance. |
| "ρ_F is the first normalized variance share attributing model uncertainty to prompt phrasing." | **FALSE — do not claim.** Cox et al. (AAAI-25) define the Prompt Sensitivity Ratio ρ_u = (U_e+ε)/(U_t+2ε) for exactly this purpose. Cite and differentiate on *units and construction* (bits of functional information vs. trace of embedding covariance). |
| "We are the first to sweep a prompt-specificity dial against accuracy." | **FALSE — do not claim.** DETAIL (2512.02246) and the code-gen specificity papers precede it. |
| "We give the first *information-theoretic, in-bits* account of prompt sensitivity grounded in a fraction-of-configurations-achieving-function measure, yielding an ability curve rather than a bare variance ratio." | **DEFENSIBLE and recommended as the headline.** This is what none of the above do. |

**Net verdict: the novelty claim as literally stated ("no prior work applies Szostak–Hazen functional
information to prompts") is not threatened.** What has changed is the surrounding context: as of
mid-2026 the *problem* (prompt variance is a large, ranking-flipping share of measured performance)
is well-established and no longer needs arguing, and three of the paper's likely secondary claims —
variance decomposition, a normalized prompt-variance *ratio*, and the specificity dial — now have
clear prior art. The strategic move is to concede all three early, cite **Cox et al. (AAAI-25) as the
direct ρ_F antecedent** plus Brittlebench, Mustahsan et al. and DETAIL as the immediate baselines,
and let the FI construct carry the novelty: interpretable bits, a fraction-of-space semantics, and an
ability curve, in place of a unit-free variance ratio.

The single most important action item from this scan: **Cox et al. must move from "a reference in the
list" to "the paper we differentiate against."** It was on the to-verify list as an unattributed 2025
item; it is in fact a published AAAI-25 paper defining a Prompt Sensitivity Ratio with the same
structural role as ρ_F. Everything else on this page is context; that one is a direct neighbour.

**Two additional risks that are methodological rather than novelty-related, both worth pre-empting:**
1. **Parse-failure confound** (FSI/PSI, 2607.09665): most apparent format sensitivity was answer
   non-compliance, not reasoning failure. State how parse failures are counted.
2. **Template-vs-question confound** (2604.18389): prompt *templates* move logits more than the
   *questions* do. Be explicit about which is being varied when computing sensitivity.
