# Research Design v2: Specificity-Centered Prompt Sensitivity with Functional Information

**Author:** Thanos Drossos. **Supervisor:** Moritz Diener. **Date:** 2026-05-04.
**Replaces:** §6 of the original synthesis.
**Companion:** §7 expansion (FI_in / FI_out).

---

## 0. Reading guide

This document is the implementation-ready research design after the supervisor meeting. It commits to specificity change as the primary perturbation axis, keeps the seven-metric stack with explicit two-number reporting after Errica, and centers Functional Information as the novel contribution. It also folds in Kirchhof, Kasneci, Kasneci (ICML 2025) on uncertainty for LLM agents, which gives the right epistemic frame for "what does specificity perturbation actually measure."

The structure follows the Booth, Colomb, Williams research cycle from *The Craft of Research* (4th ed., 2016, ch. 4): practical problem → motivates → research question → defines → research problem → solved by → research answer → contributes back to → practical problem. Each step is pinned with citations.

---

## 1. The research cycle, instantiated

### 1.1 Practical problem

Users and operators of LLM-based systems cannot tell, before they hit a query, whether the model's response is reliable for that query. A reformulation that looks identical to a human can swing accuracy by tens of points (Sclar, Choi, Tsvetkov, Suhr 2023, *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*, arXiv:2310.11324, report up to 76 percentage-point gaps between formatting templates). Worst-case paraphrases of AlpacaEval prompts drop Llama-2-70B-chat from ≈55% to 9.4% accuracy on the same query (Cao, Yu, Wang, Yu 2024, *On the Worst Prompt Performance of Large Language Models*, NeurIPS 2024). Half of Natural Questions are ambiguous in the first place (Min, Michael, Hajishirzi, Zettlemoyer 2020, *AmbigQA: Answering Ambiguous Open-domain Questions*, EMNLP 2020, arXiv:2004.10645).

Operationally, this means deployers cannot answer "if my user under-specifies, will the model still give the right answer, or will I be paged at 3 a.m." That is the practical problem.

### 1.2 Research question (motivated by the practical problem)

How sensitive is a given LLM's output to changes in *prompt specificity*, where specificity is defined as the bit-cost of moving the prompt from a maximally underspecified form to a maximally specified form on the same query, holding answer-defining content constant?

This is a one-axis cut of the broader question "how sensitive are LLMs to semantically equivalent prompt change." Specificity is privileged because (i) it is the axis Diener flagged, (ii) it maps directly onto Kirchhof et al.'s underspecification uncertainty, and (iii) it admits the cleanest functional-information reading.

### 1.3 Research problem (defined by the research question)

Existing prompt-sensitivity metrics quantify variability of output across paraphrases (Sclar 2023; Lu et al. 2023, ACL 2024; Cao, Yu, Wang, Yu 2024; Chatterjee, Renduchintala et al. 2024 POSIX, arXiv:2410.02185; Errica, Niepert, Reisach 2025, *What Did I Do Wrong?*, NAACL 2025, arXiv:2406.12334; Zhuo et al. 2024 ProSA), but none quantify the *bit-rate* at which model performance recovers as you move along a specificity gradient on a single query. POSIX, Errica's S_τ, and Farquhar's semantic entropy (Farquhar, Kossen, Kuhn, Gal 2024, *Detecting hallucinations in large language models using semantic entropy*, *Nature* 630:625–630) all measure the response distribution. None measures the prompt-utility distribution.

Functional Information, defined by Hazen, Griffin, Carothers, Szostak 2007 (*Functional Information and the Emergence of Biocomplexity*, in *In the Light of Evolution Vol. 1*, NAS Press, pp. 28–30; also PNAS 104:8574–8581) and generalized beyond biology by Wong, Cleland, Arend, Bartlett, Cleaves, Demarest, Prabhu, Lunine, Hazen 2023 (*On the roles of function and selection in evolving systems*, PNAS 120(43):e2310223120), gives the language to do this. Wong et al. 2023 explicitly license the formalism for any (configuration space U, scalar function F) pair under selection, removing the "biology only" objection. This has not been applied to prompts: a web search for "functional information Hazen Szostak prompt language model" returns only the original biology references and no applications.

The research problem is therefore: define, estimate, and validate a Functional Information-based metric `FI_in` over specificity-graded prompt families that complements existing sensitivity indices and gives a Kirchhof-style underspecification reading.

### 1.4 Research answer (the deliverable)

A two-component metric stack:

1. **Primary axis (novel):** `FI_in(q, k)` and its summary `AUFI_in(q)`, computed over a specificity-graded paraphrase universe `U_q` constructed from a multi-level specificity ladder (FollowBench-style; Jiang et al. 2024 ACL FollowBench, arXiv:2310.20410; PartialOrderEval-style; Zi, Menon, Guha 2025 NAACL/ACL, *More Than a Score*, arXiv:2508.03678).

2. **Companion two-number axis (Errica-confirmed):** `(S_τ, 1 - TVD)` reported per query and aggregated, plus the five complementary metrics from §3 below.

The metric is validated in three tiers (V1 concurrent, V2 predictive, V3 geometric) on TriviaQA, HotpotQA, AmbigQA, and a FollowBench-derived constraint-graded subset.

### 1.5 How the answer feeds back into the practical problem

A deployer can compute `AUFI_in` per query type and get a calibrated answer to "for this class of query, if the user under-specifies by k bits, what fraction of paraphrases still yield the correct answer." That converts the Sclar/Cao 76-point spread observation from a qualitative warning into a quantitative budget. Combined with the two-number Errica report it tells the deployer whether the model is wrong-but-confident (low `1 - TVD`, high accuracy variance) or right-but-flaky (high `S_τ`, mode answer correct).

The answer also gives the *user-side* recipe: if the model has a steep `FI_in` curve, the user must add specificity; if flat, the user can phrase casually. This is the deliverable the practical problem demands.

---

## 2. Literature gap analysis: what others have done on specificity, and what is missing

I read the four most recent specificity-focused papers in detail. Each has a clear methodological contribution and each has a gap that FI_in fills.

### 2.1 Kim 2025, DETAIL Matters (arXiv:2512.02246)

**What they did.** Three specificity levels (vague, moderate, detailed) generated by GPT-4 paraphrase. Specificity quantified via perplexity (GPT-2-medium, formula `exp(-1/n Σ log P_θ(x_i|x_<i))`). Tested on 30 novel reasoning tasks across math, logic, code, commonsense, decision-making. Two models (GPT-4, O3-mini). Found gain L3 minus L1 ranges from +0.47 (math) to +0.02 (decision-making).

**What is missing.**
- Three discrete levels, not a continuous gradient → no curve.
- Perplexity as specificity proxy depends on the scoring model and on token probability, not on information content of constraints added.
- No information-theoretic interpretation. No framing in bits.
- No connection to prompt sensitivity literature; treats specificity as a prompt-engineering knob, not a robustness probe.
- Only 30 tasks, only 2 models; underpowered.

**What FI_in adds.** A continuous-k FI curve with bootstrap CIs, definition of specificity in bits over a constructed paraphrase universe, and explicit linkage to Hazen-Szostak.

### 2.2 Zi, Menon, Guha 2025, PartialOrderEval (arXiv:2508.03678)

**What they did.** Augment any code generation benchmark (HumanEval, ParEval) with a partial order of prompts from minimal (`p_bot` = function signature only) to maximally detailed (`p_top` = full natural-language spec). Three augmentation strategies: LLM Summarization at word budgets `L ∈ {10, 25, 50, 75, 100, 150, 200}`, Paragraph Sampling at retain ratios `r_p ∈ {0.2, 0.4, 0.6, 0.8}`, Sentence Block Masking at mask ratios `r_s ∈ {0.2, 0.4, 0.6, 0.8}`. 41 distinct prompts per problem. Models: Qwen2.5-Coder 1.5B/3B/7B/14B, Llama-3.x 1B/3B/8B/70B. Metric: pass@1 vs prompt detail. Identified 4 prompt-detail categories qualitatively: Functional Specification, Constraints and Robustness, Solution Structure and Design Guidance, Verification and Integration.

**What is missing.**
- Code generation only.
- Pass@1 is binary; no probability or distributional information used.
- No connection to information theory; the partial order is empirical, not bit-counted.
- "Detail" is conflated with "length" via the word-budget L.
- No model uncertainty signal alongside pass@1.

**What FI_in adds.** A formal bit-count of the specificity ladder, applicability to Q&A and not just code, integration with Errica's two-number sensitivity-consistency report.

### 2.3 Jiang, Xu, Zhao, Wang, Liu, Zhao 2024, FollowBench (ACL 2024, arXiv:2310.20410)

**What they did.** 820 instructions across 50+ NLP tasks, with a level mechanism that adds one constraint per level (1 to 5). Five constraint types: **Content**, **Situation**, **Style**, **Format**, **Example**. Evolution paths used as input to LLM-as-judge (88% agreement with experts; falls 9 points without the evolution context). Measures *capability* (does the model follow the constraint), not robustness.

**What is missing.**
- Capability not robustness; FollowBench tells you whether the model obeys, not whether it gives consistent answers when the constraint set changes.
- Constraint addition is unidirectional; no measurement of "what happens when a real user under-specifies by removing the same constraint."

**What FI_in adds.** Use FollowBench's exact constraint-addition methodology in reverse — constraint *removal* — to construct underspecification ladders, then measure FI_in on the resulting `U_q`.

### 2.4 Zhang et al. 2024 / CFBench (Zhang, Xu et al. 2024, *CFBench: A Comprehensive Constraints-Following Benchmark*, ACL 2025, arXiv:2408.01122)

**What they did.** 1000 samples, 200 real-life scenarios, 50+ NLP tasks. 10 primary constraint categories with 25+ subcategories. Multi-dimensional assessment with requirement prioritization.

**What is missing.** Same as FollowBench: capability-focused. Useful as a constraint taxonomy donor.

**What FI_in adds.** Adopt CFBench's 10-category taxonomy as the perturbation axis catalog for the specificity ladder.

### 2.5 Hua, Tang, Gu, Gu, Wong, Qin 2025, *Flaw or Artifact?* (EMNLP 2025, arXiv:2509.01790)

**Critical methodological warning.** They evaluate 7 LLMs across 6 benchmarks with 12 prompt templates and find that much of the prompt sensitivity reported in the literature stems from log-likelihood scoring or rigid answer matching that fails to recognize semantically correct paraphrased answers. Switching to LLM-as-judge dramatically reduces measured sensitivity.

**Implication for our design.** F(x) in the FI_in definition must be computed semantically, not via exact match. Otherwise FI_in measurements conflate true model sensitivity with evaluation artifact. Concrete consequence: use NLI-with-gold (DeBERTa-v3-large-MNLI) or LLM-as-judge for Q&A correctness, never raw exact match. This is non-negotiable; it is the strongest critique that will hit the paper otherwise.

### 2.6 The specific gap our paper fills

The state of the art on specificity is split:
- Specificity-as-capability work (DETAIL, PartialOrderEval, FollowBench, CFBench) measures *how well* the model handles each level.
- Specificity-as-sensitivity work does not exist as a coherent line. The closest is Cao 2024 RobustAlpacaEval (worst-case paraphrase) and Sclar 2023 (format gap), neither of which factors specificity.

Functional Information closes that gap by giving a single bit-counted scalar (and a curve) that says "given this query and this model, the cost of going from underspecified to specified is X bits, and below threshold k, only Y% of paraphrases work." That is exactly the deployer-relevant question and exactly the missing piece of the literature.

---

## 3. Refined metric stack with two-number Errica reporting

After the supervisor confirmation that the deliverable is a multi-number report (sensitivity and consistency are weakly correlated per Errica 2025), the stack stays at seven, organized into three tiers.

### Tier A: Primary novel contribution

**A1. `FI_in(q, k)` and `AUFI_in(q)`** over a specificity-graded `U_q`.
- Definition: `FI_in(q, k) = -log₂(N_k(q) / |U_q|)` where `N_k(q) = |{x ∈ U_q : F(x) ≥ k}|`.
- Source: Hazen et al. 2007, NBK254300 chapter, pp. 28–30; generalised to non-biology by Wong et al. 2023 PNAS.
- F(x) is computed semantically per Hua et al. 2025 EMNLP warning.
- Reporting: curve over k ∈ [0, 1] for continuous F, point at k=1 for binary F, plus AUFI scalar.

**A2. `FI_out(x)`** as the Output-Raum dual, retained as companion.
- Definition: `FI_out(x) = log₂|𝒜_q| - H_sem(Y|X=x)`.
- Source: Farquhar et al. 2024 Nature 630:625–630 for `H_sem`; reformulation as KL-from-uniform on `𝒜_q`.

### Tier B: The Errica two-number deliverable

**B1. Normalised sensitivity `S_τ(x)`** = per-prompt entropy normalised by `ln C` (Errica et al. 2025 NAACL, arXiv:2406.12334, Eq. 3). For free-form generation use semantic-cluster entropy normalised by `log₂|𝒜_q|`.

**B2. Consistency `C(x, x') = 1 - TVD(p_τ(·|x), p_τ(·|x'))`** averaged over pairs in `U_q` (Errica 2025, Eq. 4).

Report `S_τ` and `C` as a *tuple*, not a sum. Errica showed they are decorrelated; collapsing them throws information away.

### Tier C: Complementary diagnostic metrics

**C1. Performance spread.** `max_x F(x) - min_x F(x)` over `U_q`. Source: Sclar et al. 2023; Cao et al. 2024.

**C2. Variation ratio.** `1 - mode_count / |U_q|`. Source: Lu et al. 2023, ACL 2024.

**C3. POSIX `ψ_{M,X}`.** Cross-assignment log-prob divergence (Chatterjee et al. 2024 POSIX, arXiv:2410.02185, formula on pp. 3–5). Open-weight only because it needs logits.

**C4. ESS_in.** Input-embedding dispersion (see §10b). Open-weight or external encoder.

**C5. ρ_u.** Cox 2025 Output-embedding variance ratio U_e/U_t (see §10b). Sample-based.

**C6. Semantic entropy `H_sem` (Farquhar et al. 2024 *Nature* 630:625–630).** The k-sample-and-NLI-cluster pipeline gives a per-prompt entropy `H_sem(Y|X=x) = -Σ_{c∈A_q} p_c(x) log₂ p_c(x)`. Reported as both `mean_x H_sem` (the dominant baseline for QA hallucination detection) and `Var_x H_sem` (a second-order signal: does within-prompt uncertainty itself depend on phrasing?). Note: `H_sem` is mathematically the input to FI_out via `FI_out = log₂|A_q| - H_sem`, but reporting it separately is required (i) as the field's gold-standard baseline against which any new sensitivity metric is measured and (ii) because `Var_x H_sem` is independent information that no metric in the stack captures.

### Reporting protocol

For each (query, model) pair report a 10-tuple
`(AUFI_in, FI_out_mean, S_τ_mean, C_mean, spread, var_ratio, ψ, ESS_in, ρ_u, H_sem_mean, H_sem_var)`.
For each (task, model) aggregate over queries with mean and 5/95-percentile.
For each model report cross-task 10×10 Spearman correlation matrix between the metrics; a Spearman ρ in [0.4, 0.8] between FI_in and the others is the empirical pin for novelty (claim C1 in §7.9 of the §7 expansion). Pre-registered expectations: highest correlation candidates are ρ_u (Cox 2025) and H_sem_mean (Farquhar 2024); if ρ > 0.9 with either, downgrade novelty claim to "information-theoretic re-anchoring of an existing axis in Hazen-Szostak."

---

## 4. Specificity ladder: the construction of `U_q`

This is the technical heart of the work and the place where the earlier draft was hand-wavy. Here is the operational pipeline.

### 4.1 Two-direction construction

For each query `q` we build a *specificity ladder*: a chain `x_0 ≺ x_1 ≺ … ≺ x_L` where `≺` means "x_i is strictly less specific than x_{i+1}" (Zi et al. 2025 PartialOrderEval definition adapted). Both directions are populated:

- **Specification direction (FollowBench).** Start from the user-realistic minimal prompt `x_0` and add one constraint per level using FollowBench's five-type axis (Content, Situation, Style, Format, Example; Jiang et al. 2024, §3.2). Stop at L = 5 to match FollowBench.
- **Generalization direction (DETAIL).** Start from the maximally detailed prompt `x_L` and remove one constraint per level using Kim 2025 DETAIL's `GeneralizePrompt` procedure (LLM-driven; "Rewrite the following question in a more general and less detailed way (Level l). Keep the core question intact but remove guidance and constraints").

Both directions are needed because user underspecification (real-world) and developer overspecification (lab-clean) are different modes and the model may have different `F(x)` curves on each.

### 4.2 At each ladder level, generate paraphrases

For each ladder level `l ∈ {0, …, L}` generate `N = 30` paraphrases at that specificity level using the Razavi et al. 2025 ECIR pipeline (Razavi, Soltangheis, Arabzadeh, Salamat, Zihayat, Bagheri 2025, *Benchmarking Prompt Sensitivity in Large Language Models*, ECIR 2025 pp. 303–313, arXiv:2502.06065): GPT-4o rewrite at temperature 0.8, then NLI-bidirectional filter using DeBERTa-v3-large-MNLI with both directions ≥ 0.9. Deduplicate by edit distance > 5.

Total `|U_q| = 6 × 30 = 180` per query (5 added levels + 1 baseline; or equivalently 5 removed levels + 1 maximal). Comparable to Errica's Q = 30 per condition × 6 conditions.

### 4.3 Specificity bit-count

The novel element: each ladder level has a bit-count.

Define `b(l)` = number of bits of constraint information added between level l-1 and l, computed as `b(l) = log₂(|U_{q, l-1}| / |U_{q, l}|)` if we treat each constraint as halving the satisfying-prompt space, or empirically as the change in `H_sem` of the marginal output distribution between level l-1 and l (this is the Sorensen et al. 2022 ACL mutual-information dual). Report both.

Specificity ladder index: `S(x) = Σ_{j ≤ l(x)} b(j)`, so each prompt has a bit-rank.

This makes specificity *measurable in bits*, the same units as FI_in. That is the bridge that lets us write FI_in as a function of S, not just of k. The slope `dFI_in / dS` is then "how many bits of model-output uncertainty are bought per bit of constraint added," which is the deployer metric.

### 4.4 Constraint catalog

Use CFBench's 10-category taxonomy as the inventory of constraints to add or remove (Zhang et al. 2024 CFBench): content, numerical, stylistic, format, linguistic, situation, contextual, multilingual, example, mixed. Drop multilingual and example for the seminar pilot (out of scope), keep the other eight. Within each category, FollowBench gives concrete patterns.

---

## 5. Mapping to Kirchhof's epistemic / aleatoric / underspecification frame

Kirchhof, Kasneci, Kasneci 2025 (ICML 2025, *Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents*) argues the classical aleatoric/epistemic dichotomy fails for LLM agents and proposes three new directions: underspecification uncertainty, interactive learning, output uncertainties. Two of the three map onto our work.

### 5.1 Specificity perturbation = controlled injection of underspecification uncertainty

Kirchhof et al. (§3.1) define task-underspecification uncertainty via `P(y|x) = ∫_t P(y|t) P(t|x) dt` and context-underspecification uncertainty as missing input information (their example: "When did the first Harry Potter movie come out?" without country; per Min et al. 2020, 56% of NQ test questions are ambiguous in this way).

Removing a constraint from `x_L` to produce `x_{L-1}` is exactly an injection of context-underspecification uncertainty by a known amount. Adding a constraint reduces it. This means our specificity ladder is operationally a *controlled epistemic-via-underspecification axis*: at the maximally specific end, only model-internal aleatoric noise remains; at the minimally specific end, both aleatoric and underspecification uncertainty contribute.

This gives us a clean reading of the FI curve:
- **FI_in(q, k) at high specificity** = aleatoric component only (model's intrinsic prompt-noise on a fully-specified query).
- **FI_in(q, k) at low specificity minus the high-specificity baseline** = the contribution of underspecification uncertainty per bit of removed constraint.

We can decompose AUFI_in into these two components by subtracting the high-specificity-end FI_in from the integral. That is a novel epistemic-aleatoric decomposition of prompt sensitivity that nobody has done.

### 5.2 Output uncertainties: FI_out as a Kirchhof candidate

Kirchhof §3.3 calls for "uncertainties communicated as more than mere numbers" and explicitly invites "metrics similar to those in conformal prediction that measure whether the output answer reflects a set of possibilities." `FI_out(x) = log₂|𝒜_q| - H_sem(Y|X=x)` is exactly this: it gives bits of restrictiveness on the semantic answer space, not just an entropy scalar. Cite this connection in the discussion.

### 5.3 Honest scoping

Kirchhof critiques the very dichotomy as too brittle. We do not claim our decomposition is metaphysically clean; we claim it is operationally useful. Frame it as "a decomposition that is well-defined under our specificity construction, with the underspecification component reflecting Kirchhof's task-/context-underspecification uncertainty by design." This avoids overreach.

---

## 6. Dataset and task choice

After folder + web review, four datasets cover the design.

### 6.1 TriviaQA (Joshi, Choi, Weld, Zettlemoyer 2017, ACL)
- Factoid open-domain. ~95k question-answer pairs.
- F(x) = NLI-with-gold (DeBERTa-v3-large-MNLI) per Hua 2025 warning, NOT exact match.
- Use 100 queries from the unfiltered split. Specificity ladder: add/remove temporal, geographic, and entity-disambiguation constraints.
- Why: clean answer space for `H_sem` computation; Razavi-style rephrasing already validated.

### 6.2 HotpotQA (Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning 2018, EMNLP)
- Multi-hop, 113k pairs.
- Specificity ladder is more interesting here because constraints concern *which document path* to take.
- 100 queries from the distractor setting.

### 6.3 AmbigQA (Min et al. 2020 EMNLP, arXiv:2004.10645)
- Built on NQ-open, 14,042 questions. Crucially: 50%+ of NQ-open questions are intrinsically ambiguous, with disambiguation pairs annotated.
- This dataset is gold for the underspecification side of our experiment: the ambiguous form is the natural `x_{L-1}` and the disambiguated form is `x_L`. We do not need to construct the ladder; AmbigQA gives one ladder rung for free.
- Use 100 ambiguous queries with their disambiguation rewrites as ground-truth specificity edges.

### 6.4 FollowBench (Jiang et al. 2024 ACL)
- 820 multi-level constraint instructions, 50+ NLP tasks, levels 1–5 ≈ 1 added constraint per level.
- Use a 100-instruction Q&A subset (the "open-ended question answering" tasks per their §2.2). The five levels are the specificity ladder out-of-the-box.
- This dataset removes the construction-uncertainty critique in §7.8 P1 of the §7 expansion: the ladder is published.

Total: 400 queries × ~180 paraphrases × 4 models = pilot scope.

### 6.5 Why not MMLU-Pro

MMLU is multiple-choice with C=4, where Errica's S_τ and our FI_out reduce to the same quantity (per §7.4 of the expansion). The novelty margin shrinks. Use it only if a quick reviewer-friendly comparison is needed; otherwise skip.

### 6.6 Why not GSM8K, MATH, GPQA

These are about reasoning capability, not prompt sensitivity. Specificity has been studied there (DETAIL, PartialOrderEval) but the answer space `𝒜_q` is too large and `H_sem` becomes dominated by stylistic noise. Stick with Q&A.

---

## 7. Models

Stay close to the §6.3 of the synthesis but replace the largest open model with a smaller one for compute reality.

- **Llama-3.1-8B-Instruct** (Meta, 2024). Open weights, accessible logits, well-documented tokeniser.
- **Mistral-7B-Instruct-v0.3** (Mistral AI, 2024). Open weights, contrast architecture.
- **Qwen2.5-7B-Instruct** (Alibaba, 2024). Used by PartialOrderEval; cross-paper comparable.
- **GPT-4o** (OpenAI, 2024). Closed; samples only. POSIX and Errica S_τ become Monte-Carlo over k=10 samples per prompt; FI_in works natively because it needs only F(x).

Three open-weight 7B-class models is enough to test cross-architecture generalization without pretending to a 70B run on a seminar budget. If compute budget allows, add Llama-3.1-70B-Instruct as a fourth open model to test scale effects (Zhuo 2024 ProSA: larger models showed lower PSS).

Compute estimate: 400 queries × 180 paraphrases × 11 calls (1 for F + 10 for H_sem) × 4 models = ~3.2M model calls. Polo, Maia, Choshen, Sun, Yurochkin 2024 (NeurIPS, *Efficient Multi-Prompt Evaluation of LLMs*, arXiv:2405.17202) IRT extrapolation should be wired in from day one to compress this to ~1M with statistical control.

---

## 8. Validation tiers

### V1: Concurrent validity (target: ρ between 0.4 and 0.8 with each existing metric)

For each (query, model) compute the seven-tuple. Across 400 queries and 4 models, compute Spearman ρ between AUFI_in and {S_τ, C, POSIX ψ, performance spread, variation ratio} pairwise. Three outcomes:
- All ρ < 0.4 → FI_in is fully orthogonal to existing axes (strongest result, weakest concurrent validity but strongest novelty).
- All ρ in [0.4, 0.8] → FI_in is correlated but distinct (most likely; expected).
- Any ρ > 0.9 → FI_in is a rescaling of that metric, novelty fails.

### V2: Predictive validity (target: AUC > 0.7 over baseline)

Train a logistic regression with features = (AUFI_in, S_τ, C, embedding distance to x*, specificity bit-rank S(x)) on the AmbigQA disambiguation labels (100 ambiguous + 100 disambiguated queries). Compare against a baseline using only variation ratio. Report AUC. Target: at least 0.05 AUC improvement.

### V3: Geometric validity (target: cross-query mean Spearman ρ ≤ -0.4)

For each (query, model), compute `x*(q;M) = argmax_x F(x;M)` and Spearman ρ between F(x) and `||e_M(x) - e_M(x*)||` where `e_M` is the last-layer hidden state of the open-weight model at the final prompt token. Average ρ across queries. Target ρ ≤ -0.4. This is the test of Diener's "embedding distance to most-specific input" conjecture; if it holds, FI_in has a geometric reading as embedding-ball volume around x*.

### V4 (stretch): Underspecification decomposition

Compute the AUFI_in decomposition from §5.1 (high-specificity baseline = aleatoric component; integral above the baseline = underspecification component). Test whether the underspecification component is larger on AmbigQA than on FollowBench-derived sets, where the underspecification is more controlled. If yes, the decomposition reflects something real.

---

## 9. Implementation plan, ordered

This is the ordered work-list once the design is signed off.

1. **Pin the paraphrase generator and NLI filter** (week 1). GPT-4o rewrite + DeBERTa-v3-large-MNLI bidirectional ≥ 0.9. Manual inspection on 10 queries: does every post-filter paraphrase preserve meaning? Tighten thresholds if drift.

2. **Implement the specificity ladder constructor** (week 1–2). Two routines: `add_constraint(prompt, type, level)` and `generalize_prompt(prompt, level)`. Add_constraint draws from the CFBench 10-category catalog filtered to 8. Generalize_prompt uses Kim 2025 DETAIL's GPT-4o instruction. Validate by hand on 20 queries that the resulting ladder is monotone in human judgment of specificity.

3. **Implement the seven-metric module** (week 2). One function per metric, inputs `(list of paraphrases, F-scorer, H_sem-scorer, optional logit-scorer)`, output the scalar plus diagnostics. Logit-requiring metrics behind `if has_logits`. Unit tests with hand-worked toy cases (uniform `F` = FI_in 0, single-magic-phrasing = FI_in `log₂|U_q|`).

4. **Pilot run** (week 3). 50 TriviaQA + 50 AmbigQA queries × ladder of 6 levels × 30 paraphrases × Llama-3.1-8B + GPT-4o = ~36k calls. Compute the seven-tuple per query. Check inter-metric correlations.

5. **Bit-count validation** (week 3). Compute `b(l)` empirically as `log₂(|U_{q, l-1}|/|U_{q, l}|)` and as the entropy-drop in the marginal output distribution. Are they consistent within ±20%? If yes, the bit-counted ladder is well-defined.

6. **Wire IRT extrapolation** (week 4). Implement Polo et al. 2024 NeurIPS PromptEval IRT estimator on the partial pilot grid; predict missing cells; report coverage savings.

7. **Full run** (week 5–6). 400 queries × 180 paraphrases × 4 models = ~3M calls, IRT-compressed to ~1M.

8. **Validation V1, V2, V3, V4** (week 6–7) in order.

9. **Write-up** (week 7–8). Structure: (i) Hazen-Szostak FI review and Wong 2023 generalization, (ii) `FI_in` definition, (iii) specificity ladder construction, (iv) Kirchhof underspecification mapping, (v) seven-tuple results across 4 models × 4 datasets, (vi) V1/V2/V3/V4, (vii) limitations and Hua 2025 caveat addressed, (viii) practical-deployment recipe.

---

## 10. Risks and how the design defends against them

**R1. Hua et al. 2025 EMNLP "artifact" critique.** Mitigation: F(x) computed by NLI-with-gold or LLM-as-judge per §3 throughout. Report exact-match scores in an appendix as a sanity check, never as the headline.

**R2. Generator-dependence of `U_q`.** Mitigation per §7.8 P1: two generators (GPT-4o LLM rewrite + rule-based constraint catalog), report Kendall τ between generator-induced rankings of F(x). If τ stable, generator dependence is a uniform shift.

**R3. Compute budget overrun.** Mitigation: IRT extrapolation from day one; pilot with 50 queries before scaling; halt the full run if pilot inter-metric correlations look noisy.

**R4. FI_in correlates too strongly with an existing metric (V1 fails).** Then claim weakens to "FI_in is a re-interpretation in bits of an existing axis with a Hazen-grounded reading." The Wong 2023 generalization argument keeps the contribution interesting at minimum.

**R5. Specificity ladder is not monotone in human judgment.** Mitigation: hand-check 20 ladders before scaling. If non-monotone, ladder construction needs human-in-the-loop validation, which costs time but does not break the design.

**R6. AmbigQA disambiguation labels do not map cleanly onto our ladder.** Mitigation: use AmbigQA only for V2 validation (predictive task), not as training data for the metric. The metric is defined on FollowBench / DETAIL ladders where construction is controlled.

---

## 10b. Embedding-space sensitivity: a third axis added after the v2 review

### 10b.1 Why this matters

The synthesis flagged it as Gap 1: very few papers tie input-space variation to embedding-space variation rigorously. The strongest direct evidence in the literature is Sclar, Choi, Tsvetkov, Suhr 2023 (arXiv:2310.11324, FormatSpread): "the separability of format embeddings is highly correlated with observed performance spread" (their §6). The strongest indirect evidence is Cox, Xu, Han, Xu, Li, Hsu, Chen, Gerych, Ding 2025 *Mapping from Meaning*, AAAI-25 pp. 23696–23703, code github.com/xocelyk/paraphrase-uncertainty: they show that *output*-embedding variance, decomposed via the law of total covariance, calibrates LLM uncertainty better than entropy, semantic entropy, and affinity-graph eigenvalues on TriviaQA and NQ.

The two findings are complementary. Sclar uses *input* embeddings of the prompt format. Cox uses *output* embeddings of the response. We need both, and the right place to add them is as a third axis to the stack, not as a replacement.

### 10b.2 Definitions

**A3. Embedding-Space Sensitivity (ESS).** Two operationalisations, both reported.

- **A3a. Input-embedding dispersion.** For each query q and model M, compute `e_M(x)` for every x ∈ U_q as the last-layer hidden state at the final prompt token (open-weight) or as the OpenAI text-embedding-3-large vector (closed). Define
  
  `ESS_in(q; M) = (1/|U_q|) · Σ_x ‖e_M(x) - ē‖₂` with `ē = mean_x e_M(x)`.

  Equivalently the trace of the input-embedding covariance, `tr(Cov_x[e_M(x)])`.

- **A3b. Output-embedding variance (Cox 2025).** Sample n_s responses per paraphrase; embed each response with the same encoder; apply the law of total covariance:
  
  `Cov(Y|q) = E_x[Cov(Y|q, x)] + Cov_x[E[Y|q, x]]`
  
  giving `tr(Σ_t) = tr(Σ_a) + tr(Σ_e)`, so `U_t = U_a + U_e`. Headline scalar is the **prompt sensitivity ratio** `ρ_u = U_e / U_t`. Cox 2025 §4. ε-stabilised denominator per their Eq. 11. Cox also derives a perfect-generalization baseline `ρ_u → 1/n_s` against which deviation measures overfitting.

**Headline interpretation.** ρ_u = 0 means all output variance is intra-paraphrase (model gives different answers within one paraphrase, but paraphrases agree). ρ_u = 1 means all variance is inter-paraphrase (model gives different answers because the paraphrasing changed). The latter is exactly what the practical problem in §1 asks about.

### 10b.3 Why this is not the same as POSIX, FI_in, or Errica

| Metric | Domain of variation measured | Requires |
|---|---|---|
| Performance spread | Scalar accuracy across paraphrases | F(x) only |
| Variation ratio | Modal answer agreement | Discrete answers |
| Errica S_τ | Within-paraphrase token entropy | Logits or k samples |
| Errica 1−TVD | Across-paraphrase output distribution distance | Logits or k samples |
| POSIX ψ | Cross-assignment log-prob divergence | Logits |
| Semantic entropy | Within-paraphrase semantic-cluster entropy | k samples + NLI |
| FI_in | Bit-cost of finding a high-F prompt in U_q | F(x) only |
| **ESS_in** | **Geometric spread of prompts in model's hidden space** | **Hidden state OR sentence embedding** |
| **ρ_u (ESS_out)** | **Fraction of output variance attributable to prompt change** | **k samples + embedding model** |

ESS_in is the only one of the eight that asks "does the model represent these paraphrases as the same thing?" — the literal Mapping-from-Meaning question. ρ_u is the only one whose decomposition is directly into Kirchhof's epistemic vs aleatoric components on the output side.

### 10b.4 Five caveats that decide whether ESS_in is meaningful

Embedding distance is not a free lunch. Five known failure modes from the literature, each with a mitigation:

**Caveat 1: Anisotropy.** Ethayarajh 2019 (*How Contextual are Contextualized Word Representations?*, EMNLP 2019) showed that contextual embeddings occupy a narrow cone in space, so cosine similarities between random sentence pairs are systematically inflated. Li, Zhou, He, Wang, Yang, Li 2020 (*On the Sentence Embeddings from Pre-trained Language Models*, EMNLP 2020, arXiv:2011.05864, BERT-flow) and Su, Cao, Liu, Ou 2021 (BERT-whitening) propose post-processing fixes.
- **Mitigation:** report all distances as Mahalanobis or whitened, not raw Euclidean. Whiten with the per-task covariance estimated from a held-out paraphrase set.

**Caveat 2: Layer choice.** Skean et al. 2025 *Layer by Layer: Uncovering Hidden Representations in Language Models*, arXiv:2502.02013, show that last-layer hidden states are optimised for next-token prediction, not for sentence-level semantics; intermediate layers often score higher on probing tasks.
- **Mitigation:** report ESS at the final layer, the middle layer (depth/2), and a learned aggregate (mean over layers); use whichever has highest concurrent validity with F(x) on the pilot. Pre-register the choice before the full run.

**Caveat 3: Pooling.** Final-token, mean-token, attention-weighted, and CLS-equivalent pooling give different distances. Recent work (Liu, Yi, Liu et al. 2025, attention-value embeddings) argues attention-values capture sentence semantics better than hidden states.
- **Mitigation:** report mean-pooling as default and final-token pooling as a sensitivity analysis; if discrepancy > 20% in V1 correlations, flag and discuss.

**Caveat 4: Encoder coupling.** Using `e_M` from the same model M whose F(x) we evaluate confounds metric and measurement. Using an external encoder (text-embedding-3-large, all-mpnet-base-v2) decouples but loses the model's own notion of similarity.
- **Mitigation:** report both. ESS_in^own with M's own hidden states for "does this model encode constraint information?" and ESS_in^ext with sentence-transformer for "is the prompt actually different in any reasonable embedding space?". Predicting F(x) requires the *own* encoder; cross-model comparison requires the *external* encoder.

**Caveat 5: Distance ≠ semantic divergence.** Two NLI-equivalent paraphrases can have large embedding distance (model fails to recognise equivalence) or small distance (model encodes paraphrase invariance). Both are signal. But large embedding distance plus high F(x) consistency is the *desired* state ("model recognises different surface, same answer"), while small distance plus low F(x) consistency is *catastrophic* (model collapsed surface but failed to derive the right answer).
- **Mitigation:** report the 2x2 confusion matrix of (high/low ESS_in) x (high/low F variance) per query.

### 10b.5 Connection to specificity ladder, the bit-counted axis

Here is the test that ties ESS_in to FI_in directly. Per §4.3 of this design, each ladder level l has a known bit-cost `b(l)`. If the model's hidden-state encoder respects constraint information, then

`E_{x ∈ level l, x' ∈ level l-1} ‖e_M(x) - e_M(x')‖ ≈ α · b(l)`

for some monotone function α. We can fit α empirically and report R². If the fit is good (R² > 0.5), the embedding space is *bit-coherent*: it encodes constraint information in proportion to its information content. If not (R² < 0.2), the model is anisotropic with respect to the constraint axis and ESS_in is misleading on this dataset.

This is itself a novel diagnostic. It tests whether the model's representation space is consistent with our information-theoretic ladder.

### 10b.6 Connection to Diener's "embedding distance to most-specific input" conjecture

§7.7 of Section_7.md formalised this as Hypothesis H1: F(x; M) decreases in `‖e_M(x) - e_M(x*)‖`. ESS_in supplies the half of the test that does not depend on x*. Specifically:

- H1 (Diener): Spearman ρ(F(x), ‖e_M(x) - e_M(x*)‖) ≪ 0 cross-query.
- H1' (this v2 addition): Spearman ρ(ESS_in(q;M), AUFI_in(q;M)) > 0 cross-query.

H1' says "queries where prompts are more spread in embedding space are also queries where it is harder to find a good prompt." Equivalent restatement of H1, and easier to compute because no x* is needed. Validates with one regression instead of one per query.

### 10b.7 Updated metric stack and decision

The stack moves from 7 to 9 metrics, organised in the same three tiers, with the additions tagged.

**Tier A (novel primary):** A1 FI_in, A2 FI_out.
**Tier B (Errica two-number, deliverable):** B1 S_τ, B2 1-TVD.
**Tier C (diagnostics):** C1 spread, C2 variation ratio, C3 POSIX ψ, **C4 ESS_in (new)**, **C5 ρ_u (new, Cox 2025)**.

Headline reporting per (query, model) becomes a 9-tuple. The Errica tuple stays as the deliverable per supervisor confirmation. ρ_u becomes a *secondary* headline because it is the cleanest single scalar for "how much output variance is caused by prompt change," which is the deployer-facing question.

### 10b.8 Compute impact

ESS_in adds one forward pass per prompt in the open-weight models (last-layer hidden state extraction), free under the existing pilot run. For the closed model (GPT-4o), use OpenAI text-embedding-3-large at $0.13/1M tokens, ~$15 added cost on the full run.

ρ_u uses the same n_s = 10 samples already collected for FI_out, plus a sentence-transformer embedding pass on each response. Sentence-transformer (all-mpnet-base-v2) is local, ~10ms per response, ~36k responses in the pilot = 6 minutes total. Negligible.

### 10b.9 New validation experiments

Add to §8.

- **V1.5 (concurrent embedding validity).** Spearman ρ between AUFI_in and (ESS_in, ρ_u) per query, across queries. Target: ρ in [0.4, 0.8]. If ρ > 0.9 with ρ_u, FI_in is essentially Cox's epistemic uncertainty under another name; downgrade novelty claim and reframe as "FI_in is an information-theoretic re-derivation of ρ_u."
- **V3.5 (bit-coherence of embeddings).** Linear regression of mean inter-level embedding distance against `b(l)`; report R² across 100 ladder constructions. R² > 0.5 supports the bit-counted ladder; R² < 0.2 is a finding (the model does not encode constraint information geometrically).

### 10b.10 What this changes in the slide deck

Add one slide between current Slide 11 and Slide 12: "ESS and ρ_u: input vs output embedding sensitivity." Two-column slide, left column ESS_in with the formula and the anisotropy caveat, right column ρ_u with Cox's covariance decomposition. One footnote pointing to Sclar 2023 §6 as the empirical motivation and Cox 2025 as the closest existing work.

---

## 11. What changes in the slide deck after this

Slides 12, 13, 14 stay; slides 11 and 16 need edits.

- **Slide 11 (Stack).** Reorder to put `FI_in` first, then Errica `(S_τ, C)` as the explicit two-number axis, then the rest as diagnostics. Reflects supervisor's "two-number deliverable" feedback.
- **Slide 12 (Research Design).** Replace the perturbation taxonomy bullets with the specificity-ladder construction described in §4 above. Add "FollowBench / DETAIL / PartialOrderEval" as the three ancestor methods we adapt.
- **Slide 16 (Open questions).** Strike the Errica decorrelation bullet (resolved); replace with the Kirchhof underspecification decomposition question from §5.1.

A new slide between 12 and 13 ("Specificity ladder and its bit-count") is the cleanest way to make the contribution legible.

---

## 12. Source list (folder + web, deduplicated)

**Folder papers cited above.**
- Sclar, Choi, Tsvetkov, Suhr 2023, *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*, arXiv:2310.11324.
- Cao, Yu, Wang, Yu 2024, *On the Worst Prompt Performance of Large Language Models*, NeurIPS 2024.
- Lu et al. 2023 (variation ratio), ACL 2024.
- Chatterjee, Renduchintala et al. 2024, *POSIX: A Prompt Sensitivity Index For Large Language Models*, arXiv:2410.02185.
- Errica, Niepert, Reisach 2025, *What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering*, NAACL 2025, arXiv:2406.12334; code github.com/nec-research/sensitivity-consistency-LLM.
- Zhuo et al. 2024, *ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs*, Findings of EMNLP 2024.
- Mizrahi et al. 2024, *State of What Art? A Call for Multi-Prompt LLM Evaluation*, TACL.
- Polo, Maia, Choshen, Sun, Yurochkin 2024, *Efficient Multi-Prompt Evaluation of LLMs*, NeurIPS 2024, arXiv:2405.17202.
- Sorensen et al. 2022, *An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels*, ACL 2022.
- Razavi, Soltangheis, Arabzadeh, Salamat, Zihayat, Bagheri 2025, *Benchmarking Prompt Sensitivity in Large Language Models*, ECIR 2025 pp. 303–313, arXiv:2502.06065. (Replaces "Lacharit et al." in the supervisor brief.)
- Kim 2025, *DETAIL Matters: Measuring the Impact of Prompt Specificity on Reasoning in LLMs*, arXiv:2512.02246.
- Zi, Menon, Guha 2025, *More Than a Score: Probing the Impact of Prompt Specificity on LLM Code Generation*, NAACL/ACL 2025, arXiv:2508.03678.
- Kirchhof, Kasneci, Kasneci 2025, *Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents*, ICML 2025 (PMLR 267).
- Farquhar, Kossen, Kuhn, Gal 2024, *Detecting hallucinations in large language models using semantic entropy*, *Nature* 630:625–630.
- Hazen, Griffin, Carothers, Szostak 2007, *Functional Information and the Emergence of Biocomplexity*, NBK254300 chapter 2 pp. 28–30; also PNAS 104:8574–8581.
- Szostak 2003, *Functional information*, *Nature* 423:689.
- Corona et al. 2010, *Functional Information, Biomolecular Messages and Complexity of BioSequences and Structures*, Dagstuhl 10231.
- Wong, Cleland, Arend, Bartlett, Cleaves, Demarest, Prabhu, Lunine, Hazen 2023, *On the roles of function and selection in evolving systems*, PNAS 120(43):e2310223120.
- Hua, Tang, Gu, Gu, Wong, Qin 2025, *Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs*, EMNLP 2025, arXiv:2509.01790.

**Web-sourced papers added in this round.**
- Min, Michael, Hajishirzi, Zettlemoyer 2020, *AmbigQA: Answering Ambiguous Open-domain Questions*, EMNLP 2020, arXiv:2004.10645. github.com/shmsw25/AmbigQA.
- Jiang, Xu, Zhao, Wang, Liu, Zhao 2024, *FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models*, ACL 2024, arXiv:2310.20410. github.com/YJiangcm/FollowBench.
- Zhang et al. 2024, *CFBench: A Comprehensive Constraints-Following Benchmark for LLMs*, ACL 2025, arXiv:2408.01122.
- Joshi, Choi, Weld, Zettlemoyer 2017, *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*, ACL 2017.
- Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning 2018, *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018.
- Cox, Xu, Han, Xu, Li, Hsu, Chen, Gerych, Ding 2025, *Mapping from Meaning: Addressing the Miscalibration of Prompt-Sensitive Language Models*, AAAI-25 pp. 23696–23703. Code github.com/xocelyk/paraphrase-uncertainty.
- Ethayarajh 2019, *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings*, EMNLP 2019.
- Li, Zhou, He, Wang, Yang, Li 2020, *On the Sentence Embeddings from Pre-trained Language Models* (BERT-flow), EMNLP 2020, arXiv:2011.05864.
- Skean et al. 2025, *Layer by Layer: Uncovering Hidden Representations in Language Models*, arXiv:2502.02013.
- Lin, Trivedi, Sun 2023, *Generating with Confidence: Uncertainty Quantification for Black-box LLMs*, arXiv:2305.19187 (affinity-graph U_EigV used as Cox baseline).
- Kuhn, Gal, Farquhar 2022, *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*, ICLR 2023.

**Methodological reference.**
- Booth, Colomb, Williams, Bizup, FitzGerald 2016, *The Craft of Research*, 4th ed., University of Chicago Press, ch. 4: Practical-conceptual problem cycle.

---

## 13. The single sentence that ties this together

"This seminar adapts Hazen-Szostak Functional Information from biopolymer space to LLM prompt space and uses it to measure how robustly a model's task performance survives the controlled removal of constraints, which Kirchhof et al. 2025 call underspecification uncertainty, complementing rather than replacing the seven-axis stack of existing prompt-sensitivity metrics and reporting the Errica two-number sensitivity-consistency tuple alongside as the supervisor confirmed."
