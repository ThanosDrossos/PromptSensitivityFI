# Adversarial review — the three-axis Functional Information metric

**Reviewer stance:** ICLR/NeurIPS area-chair level, hostile-by-default, no prior exposure to the project.
**Date:** 2026-08-06. **Reviewed:** `Code/PromptSensitivityFI` @ current `main`, `Prompt_Sensitivity_Full_Results_final.pptx` (2026-08-04) and `Prompt_Sensitivity_Final_Results_2026-08-03.pptx`, `Paper/SensitivityFunctionalInformationPaper/main_body.tex`, `EXPLAINER_Three_Dimensions.md`, `METRIC_PROPOSALS.md`, `data/final_run_results.md`, `Section_7_*.md`.
**Method:** every quantitative claim below was recomputed independently from the parquets with the repo's own venv; every code claim is quoted from source; the two PDFs that carry decisive weight (AmbigQA, Hazen 2007) were read directly. A parallel literature sweep hunted for prior art. Attacks that were tried and **failed** are recorded in §4 so they are not re-litigated.

> **Read §1, then §4 (what survives), then §7 (recommendations).** §2–§3 and §5 are the evidence.

> **Provenance caveat.** §2–§4 are first-hand: I ran every number. §5 (literature) is largely machine-assisted search; items marked ⚠ there are **unverified** and must be checked before citing.

---

## 1. Verdict

**Score: 4/10 — reject as currently framed, with a clear path to accept.**

The engineering is genuinely good: the pipeline is principled, seeded, cached, tested, and resumable; the metric code is mostly correct; the written Methods section is careful and self-aware. This is not a sloppy project.

But three of the four advertised contributions do not survive contact with the data:

| # | Claimed contribution (deck slide 31) | Verdict |
|---|---|---|
| 1 | ρ_F is "the first prompt-sensitivity metric that is not accuracy or entropy in disguise" | **Partly survives, badly overstated.** Orthogonality is measured on 45–66 % of cells, selected by the outcome; it is partly algebraic; and ".03 with dispersion" is the minimum over the dispersion family, not a representative value. Prior art (Cox AAAI 2025, BrittleBench 2026) already occupies most of the ground. |
| 2 | "First application of the Szostak/Hazen formalism to prompts" | **Fails.** Hazen et al. (2007) — the project's own foundational citation — contains a section titled *"The Functional Information of Letter Sequences"* that computes FI over natural-language strings scored by whether a receiver responds as intended. |
| 3 | "POSIX, S_τ, TVD, \|A_q\| and variation ratio all load on ONE construct" | **Fails as stated, but is recoverable and can be made *stronger*.** Two of the six "costumes" are exact algebraic relabelings of H_sem and three more are functionals of the same cluster distribution — shared-input agreement, not empirical convergence. The one genuinely independent index (POSIX) does *not* statistically discriminate axis 3 from axis 2. |
| 4 | A usable prompt-checker artefact | **Survives narrowly.** The real advantage is zero-shot label efficiency, not detection quality. |

And the headline empirical result — *"turn the dial 1.4 bits and ability roughly doubles"* — is, on the project's own data, **arithmetically indistinguishable from removing a 1-in-m₀ grading lottery** (§2.1). That is the single finding most likely to sink the paper, and it is also the one with the cheapest fix.

**The most important structural observation:** the LaTeX Methods section is markedly more honest than the slide deck and the talk track. Several of the worst-looking problems below are *deck-only* errors that `main_body.tex` already avoids. Fixing the presentation removes a large fraction of the reviewer's ammunition at zero scientific cost.

---

## 2. Fatal issues

### 2.1 The headline effect is a grading lottery, not a specificity effect

**The claim.** L0 = ambiguous question, L1 = its disambiguated version; gold is pinned to one seeded interpretation at both levels. Result: graded accuracy 0.18→0.42, 0.23→0.48, 0.29→0.51.

**The null nobody ran.** At L0 the model is shown a question that does not determine which of m₀ equally-valid answers is wanted, and is graded against one of them chosen by `sha256(question_id + seed)`. If the model simply answers *some* valid reading with its L1 competence, then `acc_L0 ≈ acc_L1 / m₀`.

| model | observed acc_L0 | lottery-predicted acc_L0 | observed Δ | lottery-predicted Δ |
|---|---|---|---|---|
| llama_3_1_8b | 0.180 | **0.180** | +0.238 | **+0.238** |
| mistral_7b_v03 | 0.229 | 0.199 | +0.247 | +0.277 |
| qwen_2_5_7b | 0.286 | 0.218 | +0.225 | +0.293 |

The observed gain is **at or below** the lottery prediction in all three models. On the 405 non-collision cells the residual (observed − predicted) is **−0.0003, 95 % CI [−0.041, +0.044]** (question-clustered bootstrap). Any system with *zero* sensitivity to phrasing reproduces this curve.

**The smoking gun is in their own data.** The pipeline already emits `target_collision` — the pinned answer is shared with another interpretation, so **the lottery cannot be lost**. Split on it:

| subgroup | cells | questions | mean Δaccuracy | 95 % CI | Wilcoxon |
|---|---|---|---|---|---|
| lottery live (no collision) | 405 | 135 | **+0.268** | [+0.208, +0.328] | p = 5e-34 |
| lottery pre-won (collision) | 45 | 15 | **−0.044** | [−0.135, +0.035] | p = 0.43 |

Between-group difference +0.312, Mann-Whitney **p = 3.8e-07**. Where the data itself removes the lottery, the headline effect vanishes.
*Fair caveat:* only 15 collision questions, and they may be systematically more redundant. Suggestive, not decisive alone — but it converges with the arithmetic above and with §2.2.

**The dataset's own protocol disagrees with the design.** Read directly from `Min et al. - 2020 - AmbigQA.pdf` §3.1–3.2: the task is to "output a set of semantically distinct and **equally plausible** answers", and correctness is a **set F1** over all interpretations. Grading one response against one pinned reading is non-standard and depresses L0 by construction.

**The Methods' defence is logically wrong.** `main_body.tex` §Levels argues that crediting any interpretation "would change the gold set between levels". It would not: scoring **both** levels against the union keeps the gold set identical across levels, which satisfies their own fixed-gold guardrail exactly. The second half of the defence ("answering *some* reading cannot respond to disambiguation") is an empirical assumption, not a fact.

→ **Fix: R1.** This is a re-scoring job, not a re-generation job.

> **⚠ RESOLVED (2026-08-07, R1 union-gold arm run on the cluster — see `RESULTS_R1_R2_R3_2026-08-07.md`).**
> The control settles this section in both directions:
> - **The lottery is real but not the whole story.** Δ_target decomposes into Δ_union (+0.064/+0.125/+0.119,
>   all BH-significant) + Δ_targeting (+0.161/+0.113/+0.128). The lottery accounts for **47–72 %** of the
>   headline; a genuine ability effect of 6–13 points survives. The pure-lottery arithmetic above fit the
>   marginals only because union-competence itself rises with disambiguation — which is exactly why the
>   control was required and the arithmetic alone was insufficient.
> - **The collision "smoking gun" was over-read.** Under union gold the collision subgroup is *still*
>   negative (pooled −0.10 vs +0.13), so it was never a lottery-free control — it is an atypical n=15
>   subgroup. The union arm supersedes that argument; do not use the collision split as evidence.
> - The protocol criticism (AmbigQA scores against all interpretations; the Methods' fixed-gold defence is
>   logically wrong) stands unchanged.

### 2.2 The effect requires the evidence bundle, and the talk track claims the opposite

`EXPLAINER_Three_Dimensions.md:195`: *"Both levels are closed-book, so retrieved context cannot sneak in as a confound."*
Actual v3 data: `context_mode == 'uniform_evidence'` for **900/900 rows**, mean **17.9** snippets per cell (min 2, max 32), zero rows with 0 snippets. The paper's Methods describes the evidence bundle correctly; the talk track does not.

Substantively, with the withdrawn evidence-dial numbers (Δ ≈ +0.02 closed-book → +0.235 full evidence), the finding's true scope is **answer selection from supplied evidence**, not model ability — which is exactly what §2.1 says. That is a legitimate scope statement, but it must be *stated*, not contradicted.

### 2.3 ρ_F's independence is produced by deleting the cells that carry the dependence

ρ_F is NaN whenever a cell has no variance (all-correct or all-wrong). Coverage: **45.3 % / 66.0 % / 57.3 %**. The excluded cells are exactly the accuracy extremes (68–87 % of them are all-wrong, 13–32 % all-right). Every reported orthogonality number is therefore computed on a range-restricted sample, and this is not disclosed.

| model | ρ_F ~ accuracy, complete-case | 95 % CI | ρ_F ~ accuracy, imputing 0 on zero-variance cells |
|---|---|---|---|
| qwen | +0.029 | [−0.140, +0.196] | +0.256 (p = 7e-6) |
| llama | **+0.224** (p = .0015) | **[+0.088, +0.353]** — excludes 0 | +0.477 (p = 2e-18) |
| mistral | +0.001 | [−0.149, +0.150] | +0.408 (p = 2e-13) |

The published ".08" is approximately the *mean* of (.029, .224, .001) — an average that hides a model where orthogonality fails outright.

> **⚠ CORRECTION (2026-08-07, from the R2/R3 implementation).** The imputed-0 column above is **an artifact and must not be used as the counter-estimate.** Degenerate cells are heavily skewed toward all-**wrong** rather than all-right (2.2:1 qwen, 6.8:1 llama, 4.8:1 mistral), and corr(is-degenerate, accuracy) = −0.21 / −0.34 / −0.38; substituting a constant 0 drops a spike of zeros at low accuracy and manufactures the correlation. The principled replacement is the hierarchical posterior (R2), under which every within-model, within-level association is **below \|0.09\|**. The *selection* criticism in this section stands in full — complete-case deletes 34–55 % of cells non-randomly — but the honest conclusion is "independence cannot be *established* at any useful bound (the data support only \|ρ\| < 0.33–0.45)", not "the axes are strongly correlated". See `RESULTS_R1_R2_R3_2026-08-07.md`.

**And the orthogonality is partly definitional.** ρ_F is a variance *share* normalised by a mean-dependent Bernoulli scale; dividing by p̄(1−p̄) is what removes accuracy. Presenting a mean-normalised statistic's decorrelation from the mean as an empirical discovery invites the obvious objection.

**".03 with dispersion" is the minimum over the family, not a representative.** Against the other members of the same claimed-interchangeable axis-3 family:

| | qwen | llama | mistral |
|---|---|---|---|
| ρ_F vs H_sem *(the reported number)* | −0.025 | −0.043 | +0.122 |
| ρ_F vs variation ratio | **+0.411** | **+0.283** | **+0.362** |
| ρ_F vs 1−TVD (`consistency_mean`) | **−0.400** | −0.171 | **−0.356** |
| ρ_F vs Var[FI_out] | +0.138 | +0.263 | **+0.405** |
| **ρ_F vs dispersion factor (PC1, 69–78 % var)** | **+0.172** (p=.046) | +0.085 | **+0.256** (p=.0007) |

**Power.** A 90 % CI supports no equivalence bound tighter than **±0.17 / ±0.33 / ±0.13** (vs accuracy). "Orthogonal" is not established; "not strongly correlated" is.

→ **Fixes: R2, R3.**

### 2.4 "One construct, many costumes" is not evidence of convergence — it is one object read six ways

Verified over all 900 rows. *(An earlier draft of this review overstated the case; the precise breakdown is below.)*

| reported agreement | status |
|---|---|
| **S_τ (Errica) = .94** | **Exact identity.** `s_tau_mean ≡ h_sem_mean / log₂(a_q)`, max abs diff **1.67e-16**. Errica's S_τ *is* normalised semantic entropy — a relabeling, not an independent construct. |
| **FI_out_fixed** | **Exact identity.** `fi_out_fixed = log₂(m₀) − h_sem_mean`, residual **0.0e+00**, and m₀ has exactly one distinct value per question ⇒ every paired L0→L1 claim about FI_out_fixed **is** the H_sem claim. Confirmed by identical Wilcoxon p-values (3.74e-07 vs 3.76e-07 llama; 0.0113 vs 0.0112 qwen; 0.034 vs 0.034 mistral). **Reporting both double-counts.** |
| **Var[FI_out] = .70** | Not an identity *as reported*. `fi_out_var ≡ h_sem_var` exactly (4.44e-16), but the reported number pairs Var[FI_out] against H_sem **mean** (+0.758) — a genuine mean-vs-variance correlation of the same quantity. |
| **\|A_q\| = .91, variation ratio = .75, 1−TVD** | Not identities, but all are functionals of the **same pooled `cluster_assignments` object** as H_sem (`orchestrator.py`). Shared-input agreement, not independent convergence. |
| **POSIX ψ = .60** | The only genuinely separate measurement (its own teacher-forced pass) — and it fails to discriminate, see below. |

**The honest claim** is therefore not "six metrics turn out to measure one thing" (which sounds empirical) but "**five of the six are functionals of a single object — the semantic cluster distribution — and two of them are exact relabelings of H_sem**". That is a stronger statement, and it is provable rather than estimated.

**The axis-3 sign flip is a post-hoc choice that reversed a significant result.** `fi_out_mean` (observed reference) moves the *opposite* way and is **significant doing so**: llama Δ = −0.374 (p = 2.2e-04), qwen Δ = −0.339 (p = 3.1e-04). Replacing the observed \|A_q\| with the constant log₂m₀ — adopted after that was seen (`show_specificity.add_fi_out_fixed`, 2026-07-17) — converts a significant result in the wrong direction into a significant result in the right one, by an affine map that cannot add information. The "moving yardstick" rationale is defensible *as stated*, but the sequence must be disclosed and the `fi_out_mean` result reported in the same table.

Side issue: `s_tau_mean` is exactly 0 in **14.6 %** of cells, 94.7 % of which have `a_q ≤ 1` (degenerate normaliser), inflating the apparent agreement.

**And POSIX does not do what slide 28 says.** The claim is that POSIX loads on dispersion and "the phrasing axis stays empty". Computed on the *same* subset (the reported .63/.43/.61 use n=100 while the ρ_F comparison necessarily uses the ρ_F-covered subset — an apples-to-oranges comparison):

| model | n | POSIX~H_sem [95 % CI] | POSIX~ρ_F [95 % CI] | Williams test of the difference |
|---|---|---|---|---|
| qwen | 42 | +0.409 [+0.119, +0.634] | +0.237 [−0.072, +0.505] | t=0.82, **p=0.42** |
| llama | 62 | +0.380 [+0.143, +0.575] | +0.297 [+0.051, +0.509] | t=0.48, **p=0.63** |
| mistral | 62 | +0.560 [+0.360, +0.710] | +0.318 [+0.074, +0.526] | t=1.97, **p=0.054** |

POSIX~ρ_F **excludes zero** for llama and mistral, and in no model is the dispersion loading significantly larger. Contribution #3's discriminant claim is unsupported at this sample size.

→ **Fix: R4** — and note it makes the contribution *stronger*, not weaker.

### 2.5 The novelty claim contradicts the project's own foundational citation

`EXPLAINER:47`: *"A literature search for FI applied to prompts returns nothing. That gap is the paper."* Deck slide 31: *"First application of the Szostak/Hazen formalism to prompts."*

Read directly from `Hazen et al. - 2007 …pdf`, pp. 8575–8576, section **"The Functional Information of Letter Sequences"**:
- configuration space = sequences of n letters (26ⁿ);
- degree of function E_x = "the probability that a local fire department will understand and respond to the message" — i.e. **the probability that a natural-language string evokes the intended response from a receiver**;
- I(E_x) = −log₂[M(E_x)/26ⁿ]; worked example ≈1000 of 26¹⁰ ten-letter sequences ⇒ **≈36 bits**;
- it even discusses degraded variants ("phonetic misspellings (FYRE or MANE), mistakes in grammar or usage… or typing errors") yielding lower response probability.

That is structurally this paper's construction with the receiver swapped from a fire department to an LLM. A reviewer who opens the cited source finds this in its first worked example.

**Same source imposes a constraint the paper violates.** Hazen: rigorous FI "requires knowledge of two attributes: (i) all possible configurations of the system … and (ii) the degree of function x for every configuration." U_q is **10 samples from Phi-4**, not the configuration space. FI_in is therefore not Hazen's I(E_x); it is a **generator-relative** survival rate. Its bits are not comparable across questions, models or papers without fixing the generator.

→ **Fix: R5** — and this converts into a real contribution, see §7.

---

## 3. Major issues

**3.1 The "ruler" covers two of four components.** Eq. 2 is FI = −log₂(\|{c : F(c) ≥ k}\|/\|C\|). FI_in and FI_spec are literal instances. FI_out = log₂\|A\| − H_sem is not — it is KL-from-uniform. ρ_F is a variance share, and the paper's **own Table 1 lists its unit as "share"**, contradicting "every metric is an instance of Equation 2."
*Constructive:* FI_out **is** on the ruler if the surviving count is written as the perplexity (Hill number of order 1), 2^H_sem: FI_out = −log₂(2^H_sem/\|A_q\|) = −log₂(*effective* surviving fraction). State that identity. For ρ_F, stop claiming it is on the ruler — it is a second-order quantity *about* the F that axis 1 counts.

**3.2 "ρ_F is correctly non-responsive to the dial" is underpowered, not established.** Paired n = 36 / 70 / 58 (cells covered at *both* levels). 95 % CI on Δρ_F = [−0.106, +0.143] / [−0.009, +0.087] / [−0.071, +0.104] against metric means of .40 / .13 / .23 — the CIs admit changes of 25–70 % of the metric's own magnitude. Minimum detectable effect at 80 % power = 0.172 / 0.067 / 0.122. Additionally, coverage differs by level, so the two level means are computed on **different question subsets**.

**3.3 "Two independent measurement channels agree at .67" is false as stated (deck only).** `metrics/rho_u.py:5-8` and `orchestrator.py` ("# ρ_u on the per-paraphrase **response** embeddings") show ρ_u is the same variance-ratio design on the **same N×k generations** that produce ρ_F, differing only in readout (embeddings vs binary correctness) and in having **no** noise correction. The talk track's "embedding geometry of the **prompts**" is factually wrong — that quantity is ESS_in (`ess_in_fn(prompt_embeddings)`). **`main_body.tex` gets this right**, including the correct null expectation (N−1)/(Nk−1). Their agreement is an *upper bound* on true construct agreement because sampling error is shared.

**3.4 "Significant in all three models, p ≤ 5e-9" is false for H_sem.** Recomputed paired Wilcoxon over the headline metric family (27 tests, BH correction):

| claim | qwen | llama | mistral |
|---|---|---|---|
| Δaccuracy | 1.3e-09 ✅ | 1.4e-13 ✅ | 1.7e-11 ✅ |
| ΔAUFI | 9.9e-10 ✅ | 8.4e-14 ✅ | 2.5e-11 ✅ |
| **ΔH_sem** | 0.0113 (q=.019) | 3.7e-07 ✅ | **0.034 (q = 0.051 — fails BH)** |
| Δρ_F | 0.71 | 0.26 | 0.88 |

The blanket "p ≤ 5e-9" applies only to Δaccuracy/ΔAUFI. Mistral's ΔH_sem does not survive multiplicity correction. The paper's stated position ("we report unadjusted p-values and state the number of tests") is not sufficient to support the words *"all 3 models"* on the slide.

Over the full computable family (19 metrics × 3 models = 57 paired tests), **Holm**-adjusted: accuracy 3/3 survive, AUFI 3/3, **H_sem 1/3** (llama 2.3e-05; mistral 0.88; qwen 0.36), FI_out_fixed 1/3 (identical test), ΔFI premium 1/3, **ρ_F 0/3**. So the axis-3 directional claim holds in **one** model under correction, not three.

**3.4b The three models are not three replications.** Their per-question deltas are strongly correlated — r ≈ **0.73** for accuracy and ≈ **0.40** for H_sem — giving roughly **1.2 and 1.7 effective independent replications out of 3**. "Significant in all three models" is therefore close to one test reported three times. This is the single easiest way for a reviewer to halve the paper's apparent evidence; pre-empt it by reporting a model-as-random-effect analysis, or by stating plainly that the three models are correlated measurements on one question sample.

**3.5 The "capacity share 11 % → 82 %" story credits confident wrongness.** `fi_out_fixed` is uncorrelated with accuracy (Spearman −0.016 / +0.116 / +0.282). Among "confident" cells (H_sem < 0.2), **47 % (qwen) / 37 % / 38 % are wrong** (accuracy < 0.2). A model that confidently emits one wrong answer scores maximal "capacity realized". The 1.429-bit "capacity" is also arbitrary: the observed answer space is 1.39 (qwen) / **3.33 (llama)** / 2.81 (mistral) bits. And FI_out_fixed goes negative — it is not a share.

**3.6 The prompt-checker's headline defence is a sign error.** Reproduced exactly (n = 1852, n_ambiguous = 1022): AUROC(shorter = ambiguous) = **0.5431**. The published **0.4569** is the sign-flipped orientation. A baseline predictor takes its better orientation; "collapses below chance" must be withdrawn. Honest margin: **+0.124**, not +0.21.

**3.7 The probe's advantage is label efficiency, not detection quality.** Two protocols, matched carefully:

| protocol | TF-IDF word | TF-IDF char | first-word only | hidden-state head |
|---|---|---|---|---|
| **frozen, no holdout labels** (matches the head) | 0.544 | 0.562 | — | **0.667** |
| **in-domain 5-fold CV on holdout labels** | 0.655 | 0.663 | 0.650 | 0.667 |

Good news: under the *matched* protocol the head clearly beats text baselines (+0.10) — that is a real result the paper does not currently make. Bad news: with ~1.5 k in-domain labels a bag-of-words model, or literally **the question's first word**, matches it. The deliverable framing ("~10× cheaper than sampling") must be restated as zero-shot transfer, and both baselines must appear.

**3.8 Probe controls sit only at the weakest layer, and the null is wide.** `control_permuted` and `baseline_length` exist **only at the shallowest layer** in every `probe_results_*` file; the headline layers (21/24, AUROC .72–.83) have no control at all. A single permutation yields control AUROC ranging **0.372 → 0.583**, i.e. null SD ≈ .08 — comparable to the claimed margins. Best layer is selected post hoc over 4 layers × 2 head types. `main_body.tex` claims "We report permuted-label and prompt-length baselines for every head" — the data do not support that sentence.

**3.9 The framework's own flagship construct is the least predictable thing in the probe suite.** Best-over-layers AUROC for ρ_F: qwen .534/.555, llama .644, mistral .582, versus .72–.83 for accuracy and H_sem. The fragility head is at chance and this negative result is absent from the final deck.

**3.10 Reproducibility.** The correlation matrix `figures/v3_metric_corr.npy` that underpins the entire independence section has **no generating script in the repo**. The 150-question selection is not recorded as a manifest. `spread (Cao)` is an all-NaN row/column in that matrix (harmless in the current figure, which excludes it, but `np.nan_to_num(nan=0.0)` in `fig_independence` would silently render any undefined correlation as "independent").

**3.11 The "−0.86 bits" headline is a unit convention, not a measurement.** `aufi_in` caps unreachable thresholds at log₂(N+1) = 3.459 bits. At L0, **56–67 %** of the 21 thresholds are unreachable (no paraphrase attains them) versus **32–40 %** at L1 — so the AUFI difference is dominated by *how many thresholds are capped*, and its magnitude scales almost linearly with the arbitrary cap constant:

| cap convention | llama Δ | mistral Δ | qwen Δ |
|---|---|---|---|
| log₂(5+1) = 2.585 | −0.617 | −0.641 | −0.582 |
| log₂(N) = 3.322 | −0.817 | −0.834 | −0.759 |
| **log₂(N+1) = 3.459 (used)** | **−0.855** | **−0.871** | **−0.792** |
| log₂(20+1) = 4.392 | −1.119 | −1.119 | −1.019 |

The **sign and significance are robust** — this is not a sign-flip artefact (an internal reviewer claimed it was; that claim extrapolates to cap = 0, which is not a meaningful convention). But "ΔAUFI_in = −0.79…−0.87 **bits**" must not be presented as a measured quantity in bits. Report the delta *with the cap stated*, plus this sensitivity table, or report the censoring rate (56–67 % → 32–40 %) directly, which is the actual finding.

**3.12 The "FI_in(k) curve", named the primary deliverable, is one number.** The persisted `fi_in_curve_vals` is the **binary (T = 0)** curve. Across all 900 cells it takes at most two distinct values, and the only transition that ever fires is between k = 0 and k = 0.05: **688/900 (76.4 %) have exactly that one step and 212/900 (23.6 %) are completely flat**. Since FI_in(q,0) = 0 by construction, the 21-point curve encodes exactly one free number, −log₂(f_mean), and `aufi_in = 0.975·(−log₂ f_mean)` exactly. The Hazen "islands of function / stepped shape" analogue therefore has **no support in the persisted column**.
*Constructive:* the **graded** track does not have this problem — per-paraphrase F takes 11 distinct values (0, 0.1, …, 1.0), so a graded FI_in curve is a genuine curve. Plot and persist that one, and re-run the stepped-shape test on it with a null (see R7's permutation discipline).

**3.13 Scope.** Single dataset, single task format (short-answer QA over supplied evidence), single paraphrase generator, three same-size (7–8 B) open models. No general claim about "prompt sensitivity" is licensed.

---

## 4. Things that survive scrutiny (do not spend effort re-defending)

These attacks were run and **failed** — they are settled in the project's favour:

- **The ICC estimator is correctly specified.** Per-sample F really is binary 0/1 (`scoring/nli_with_gold.py:44-46`), so `SS_within = k·p̂(1−p̂)` is exact, not an approximation.
- **No level asymmetry in universe size.** Exactly 10 accepted paraphrases at both levels (170 cells L0, 169 L1); 1 singleton fallback in the entire corpus.
- **AUFI ≡ accuracy** confirmed: Spearman −0.9997 / −0.9992 / −0.9993. Demoting it was right.
- **The between-model ρ_F ordering is real and is not a decoding-noise artefact.** Mean within-paraphrase p(1−p) is the exact inverse of the ρ_F ranking (qwen .0247 < mistral .0378 < llama .0598), so a reviewer will allege the ranking is just decoding entropy. It is not: the *absolute* noise-corrected variance component σ²_B = (MS_B−MS_W)/k gives the **same order** (qwen .067 > mistral .039 > llama .022). Paired Wilcoxon p = 1e-11 / 7e-5 / 7e-5. **Report σ²_B pre-emptively.**
- **The three-factor structure passes the correct test.** Horn parallel analysis (2000 random matrices, n=136, p=13): observed eigenvalues 5.77 / 2.44 / 1.60 vs random 95th percentile 1.69 / 1.50 / 1.38 ⇒ exactly 3 factors retained. Varimax is clean (F1 dispersion, F2 ability, F3 sensitivity). Cumulative variance is **75.5 %**, not the 70 % printed.
  *Caveats to pre-empt:* the metric set is curated (6 of 13 are known-duplicate dispersion indices, 2 are accuracy and its exact monotone transform), so F1 and F2 are guaranteed; F3 has only two real indicators (ρ_F, ρ_u) and you cannot both define a factor by two variables and cite their correlation as independent corroboration; and **ESS_in loads −0.48 on F3**, contradicting "ESS_in ⊥ everything".
- **The x\* geometry result is real.** Null simulation (x\* = argmax F, F drawn from the true empirical F distribution, random unit embeddings, 4000 reps, d = 64 and 1024): null mean = **+0.002 ± 0.011**, sd 0.35. Observed mean = **−0.325** over 437 cells ⇒ ~18 SE from zero. *But:* it misses its own pre-registered target of ≤ −0.4, coverage is 437/894 = 49 % and undisclosed, and per-cell sd (0.35–0.42) equals the null sd, so only the mean carries signal — no per-question claim is licensed.
- **POSIX is faithfully implemented** (`metrics/posix.py` matches Chatterjee Eq. 4).
- **No dead ladder-code contamination** in the live specificity path.
- **The length control is adequate** for what it claims. Within-cell ρ(length, F) ≈ 0 over a range exceeding the manipulation.
- **The asymmetric paraphrase gate is defensible.** An internal reviewer flagged that L0 candidates are gold-checked against the union (mean 4.13 variants) while L1 uses target variants only (mean 1.49) — a 2.77× asymmetry between the two compared conditions. **This attack fails:** the asymmetry is disclosed and justified in `main_body.tex`, is forced by the design (single-gold L0 gating rejected 100 % of NLI-valid L0 paraphrases), and does not bind on the outcome — exactly 10 paraphrases are accepted at both levels for every question. Keep the disclosure; do not treat it as a defect.
- **There is no anti-dose-response in FI_spec.** An internal reviewer reported β = −0.127 (t = −4.45) for Δaccuracy on FI_spec bits, i.e. "more bits produce less gain". On verification the conditional estimate is **not interpretable** — it conditions on L0 accuracy, which is itself a function of m₀ (Lord's paradox). The unconditional estimate is −0.088 (cluster-robust t = −2.26) and is carried by a small subgroup. **The correct statement is weaker and still worth making:** the dial's *per-bit scale* is unvalidated — the design has only two points (0 and log₂m₀), so nothing establishes that the effect is linear in bits. Do not claim a dose–response you have not measured; add intermediate levels or drop the "bits of specificity" scale language.

---

## 5. Literature: what is already taken

The sweep found substantial prior art. **⚠ Verification status:** items marked ✅ were confirmed against a local PDF or are well-established; items marked ⚠ were reported by search agents and **must be verified before citing** — several have 2026 arXiv IDs that could not be re-checked (the session's web-search budget was exhausted).

### 5.1 Against the FI framing (contribution #2)
- ✅ **Hazen et al. 2007, PNAS** — already applies FI to natural-language strings scored by receiver response (§2.5). Fatal to "first application".
- ⚠ **Root-Bernstein 2024, PNAS letter e2318689121** (+ reply e2406598121) — the Wong/Cleland/Hazen 2023 "law of increasing functional information", the citation that licenses leaving biology, is **publicly contested**. Cite it aware of that; do not present it as settled.
- ⚠ **Zenil 2025** — attacks FI as observer-relative and as conflating rarity with function; the same argument that dismantled Assembly Theory. This transfers exactly to the generator-relativity problem in §2.5.
- ⚠ **Dembski & Marks 2009** — −log₂(p) as "endogenous information" is associated with intelligent-design literature. A reputational hazard worth knowing about before a public talk.
- ✅ **Chen et al. 2021 (pass@k)** — FI_in is a monotone reparameterisation of a pass rate. The project already concedes ρ(AUFI, accuracy) = −1.00.
- ⚠ **Polo et al. NeurIPS 2024 (PromptEval)** — already estimates the full *distribution* of performance across prompt variants with quantile guarantees. FI_in is (the log of) its survival function.

### 5.2 Against ρ_F (contribution #1)
- ✅ **Cox et al., AAAI 2025** — ρ_u = U_e/U_t, a per-question epistemic/total variance share explicitly "to quantify how much LLM uncertainty is attributed to prompt sensitivity". Already in the repo as a metric. This is the closest prior art and `main_body.tex` handles it well.
- ✅ **Romanou et al., BrittleBench (arXiv:2603.13285)** — applies the law of total variance with a random-effects ANOVA and reports an ICC-like Π_m = perturbation variance / total variance. `main_body.tex` already distinguishes it correctly (their inference is deterministic so the sampling term vanishes). Keep that paragraph; it is the paper's best piece of positioning.
- ⚠ **Żatuchin (arXiv:2607.13304)** — claimed explicit crossed random-effects / generalizability-theory decomposition separating within-prompt resampling from paraphrase facets, reporting ICCs. **If real, this is the most direct scoop of ρ_F. Verify first.**
- ⚠ **Pecher et al. (arXiv:2602.04297)** — "a significant portion of observed prompt sensitivity is attributable to prompt *underspecification*". This is the project's thesis, published first, plus linear probes on internal representations.

### 5.3 Against the probes (contribution #4)
- ✅ **Kossen et al. 2024, "Semantic Entropy Probes" (arXiv:2406.15927)** — linear probes on hidden states predicting semantic entropy. This is the dispersion head, already published. **Must be cited and compared.**
- ⚠ **Zhang et al. (arXiv:2509.13664), "Sparse Neurons Carry Strong Signals of Question Ambiguity"** — the vagueness head, with cross-dataset generalisation.
- ⚠ **Ramesh et al. (arXiv:2606.05486)**, ⚠ **Cencerrado et al. (arXiv:2509.10625, question-only linear probes predicting accuracy)** — the reliability head.

### 5.4 Against the empirical headline
- ✅ **AmbigQA (Min et al. 2020) §3.2** — set-F1 over all interpretations is the dataset's own protocol (§2.1).
- ✅ **Kim 2025, "DETAIL Matters" (arXiv:2512.02246)** — in the project's own PDF folder. A graded specificity ladder driving accuracy, with a doubling on weaker models. The FI_spec headline is not new.
- ⚠ **Keluskar et al., IEEE BigData 2024**; ⚠ **Su & Cardie 2026**; ⚠ **Huang et al. 2026** — ambiguous-vs-disambiguated accuracy on AmbigQA is established, including "upper bound" rows equivalent to the L1 condition.

### 5.5 Against H_sem as implemented
- ⚠ **McCabe et al. (arXiv:2509.14478)**, ⚠ **Nguyen et al., Findings of ACL 2025 (arXiv:2506.00245)** — semantic-alphabet-size estimation and small-k saturation. Directly relevant: plug-in entropy over k = 10 is downward-biased and \|A_q\| is an observed-richness estimate that the project itself notes it "cannot exceed what was measured". The Chao1 correction was tried and dropped in one sentence; that needs a sensitivity analysis, not a dismissal.
- ⚠ **Tomov et al. (arXiv:2511.04418), "The Illusion of Certainty: UQ for LLMs Fails under Ambiguity"** — most on-the-nose title in the sweep. If real, it directly addresses using H_sem on ambiguous questions.

---

## 6. What the honest contribution actually is

Strip out everything that does not survive, and this is what is left — it is not nothing, and it is publishable:

1. **A measurement model that unifies the prompt-sensitivity zoo, with analytic reductions.** Showing that S_τ ≡ H_sem/log₂\|A\|, Var[FI_out] ≡ Var[H_sem], FI_out_fixed ≡ log₂m₀ − H_sem, and that most published indices are functions of one pooled cluster distribution, is a **real service to the field** — and as *proofs* it is far stronger than as n=150 correlations.
2. **ρ_F: a noise-corrected, per-question, task-success variance share.** Distinct from Cox (gold-free, embedding readout, uncorrected) and from BrittleBench (model-level, deterministic inference). Real, if narrower than claimed, and the Methods section already argues it correctly.
3. **A demonstration that the field's standard treatment of ambiguous-question evaluation conflates a grading lottery with model ability.** This is currently the paper's biggest liability; turned around, it is its most interesting finding.
4. **Evidence that ambiguity is linearly decodable from a single forward pass and transfers zero-shot** to annotator-labelled questions where trained text baselines do not (0.667 vs 0.544/0.562).

**Note the shape of that list: it is a measurement-critique paper, not a new-metric paper.** That reframing is the single highest-leverage change available.

---

## 7. Recommendations

Ordered by (impact ÷ cost). Each gives **what is wrong → what to do → what the finished result must look like**.

### R1 — Union-gold control arm ⚠️ BLOCKING · cost: one re-scoring job, no generation

**Wrong.** The headline is arithmetically indistinguishable from a 1/m₀ lottery (§2.1); AmbigQA's own protocol scores against all interpretations; the Methods' justification for target-only scoring is logically incorrect.

**Do.** Re-score the **existing cached generations** against the union of *all* interpretations' answer variants, at **both** levels — which keeps the gold set constant across levels, satisfying the project's own fixed-gold guardrail.
- New script `scripts/rescore_union_gold.py`; reuse `scoring.nli_with_gold.f_score_batch_multi_gold` with `golds = ⋃ᵢ variants(interpretationᵢ)`.
- Emit `f_graded_union_per_paraphrase`, `f_graded_union_mean`; **do not modify `metrics/`**.
- **Verify the cluster response cache first.** The local `data/cache/llm_cache.sqlite` holds only 6 471 rows and they are all gateway-era (`kit.gpt-4.1`) — the v3 responses live on bwUniCluster. If that cache is gone this becomes a regeneration job and the cost changes by two orders of magnitude; check before scoping.

**Target result.** A 2×2 table {target-gold, union-gold} × {L0, L1} per model, plus the decomposition
`Δ_target = Δ_union + Δ_targeting`, plus the collision-split table from §2.1 (free, computable today).
- If **Δ_union ≈ 0**: state plainly that FI_spec measures *resolution of referential ambiguity for the grader*, not model improvement. That is a **more interesting and more defensible** claim than the current one, and it makes §2.1 your finding instead of your reviewer's.
- If **Δ_union > 0** significantly: that residual is the real ability effect and becomes the new headline, with the lottery share quantified and removed.

Either way the paper gains a control it currently cannot survive without.

### R2 — Replace the per-cell ICC with a partially-pooled hierarchical model ⚠️ BLOCKING · cost: analysis only

**Wrong.** Complete-case ρ_F deletes 34–55 % of cells non-randomly (§2.3), and the reported "stability" of .81/.92/.95 is **not a reliability** — it is a correlation between a statistic and a superset of itself. Verified directly: converting the stored rates to integer counts, `count₂₀ − count₁₀ ∈ [0,10]` for **every paraphrase in 100/100 cells in all three models (0/1000 violations)**, i.e. the k=20 arm literally contains the k=10 responses. `main_body.tex` says so explicitly ("a larger-k replication reuse[s] the responses of the smaller one exactly"). The inflation is exactly as predicted: √((1+r_disjoint)/2) gives 0.897/0.969/0.955 against the observed 0.875/0.971/0.964.

Two honest reliabilities, both computed here:

| | qwen | llama | mistral |
|---|---|---|---|
| **disjoint-sample** split-half (independent second k=10 replicate) | 0.843 | 0.475 | 0.758 |
| **disjoint-paraphrase** split-half, mean of 200 random 5/5 splits | **0.215** | **0.350** | **0.405** |
| → Spearman-Brown, full length | **0.35** | **0.52** | **0.58** |

*(A single 5/5 split is itself unstable — sd ≈ 0.05 across splits — so always average over many.)*

The **paraphrase** split-half is the one that matters, because ρ_F must generalise over paraphrase universes. At reliability 0.35–0.58, per-question ρ_F claims are not supportable, observed correlations are heavily attenuated, and the low cross-model agreement (.2–.45) cannot be attributed wholly to a "question × model interaction" — a large part of it is simply measurement error. Note also that **qwen has the highest mean ρ_F (.40) and the lowest reliability (.35)**.

**Do.** Fit one Bayesian binomial GLMM per (model, level): `correct ~ 1 + (1 | question) + (1 | question:paraphrase)`. The posterior for the paraphrase-level variance component is **defined for every cell**, including all-correct and all-wrong ones, borrows strength across cells, and yields honest per-cell uncertainty. Report:
- σ²_B (absolute, noise-corrected) **and** the ICC-style share ρ_F — the absolute component is comparable across models and pre-empts the decoding-noise objection (§4);
- coverage = 100 %, with the old complete-case numbers in an appendix for continuity;
- a true split-half reliability (split the 10 paraphrases, or the k samples into disjoint halves of 5) — never the k10/k20 superset comparison.

**Target result.** A table of ρ_F and σ²_B with credible intervals on **all 900 cells**, and the §2.3 orthogonality table recomputed under (a) complete case, (b) zero-imputation, (c) the hierarchical posterior. The paper states which it treats as primary **and why**, before looking at the answer.

### R3 — Stop saying "orthogonal"; test equivalence ⚠️ BLOCKING · cost: analysis only

**Wrong.** ".08" and ".03" are point estimates with no uncertainty, averaged over models (hiding llama's +0.224, CI [+0.088, +0.353]), and ".03" is the minimum over the dispersion family.

**Do.** (a) Pre-state an equivalence bound (e.g. \|ρ\| < 0.2 = "practically independent") and run TOST per model per level. (b) Correlate ρ_F with the **dispersion factor** (PC1), not with a hand-picked representative — and report the full family table from §2.3. (c) Report disattenuated correlations using the R2 reliability. (d) Report the correlation matrix **within level and within model**, as `main_body.tex` §Aggregation already promises — the deck's pooled numbers contradict the paper's stated protocol.

**Target result.** A sentence of the form: *"ρ_F shares at most X % of its reliable variance with ability and Y % with dispersion (90 % CI), on the interior of the accuracy range; we cannot establish independence tighter than ±0.2 at n = 136–198."* Plus: keep the Horn parallel analysis (§4) — it is the strongest independence evidence you have — and drop the octant test, which has no power (correlated Gaussians at ρ=0.7 fill all 8 octants ~98 % of the time).

### R4 — Convert "many costumes" from correlations into proofs 🔼 HIGH VALUE · cost: half a day

**Wrong.** Four of six "costumes" are algebraic identities presented as empirical convergence (§2.4); POSIX does not discriminate axis 3 from axis 2.

**Do.** Replace the correlation heatmap headline with a small **reduction table**: for each published index, one line of algebra showing what it is as a function of the pooled cluster distribution, with the exact identity where one exists (S_τ = H_sem/log₂\|A\|; Var[FI_out] = Var[H_sem]; FI_out_fixed = log₂m₀ − H_sem). Reserve *empirical* correlation claims for indices you did **not** re-implement from a shared input — in practice POSIX only — and restate that result honestly using the Williams test in §2.4. Also fix the S_τ = 0 degeneracy (14.6 % of cells, a_q ≤ 1) or exclude those cells with a stated rule.

**Target result.** A theorem-style subsection: *"Proposition 1. S_τ, Var[FI_out], \|A_q\| and 1−TVD are deterministic functions of the semantic cluster distribution; consequently the reported inter-metric agreements are identities, not evidence."* This is a **better** contribution than the current one and it is unattackable.

### R4b — Report axis 1 honestly: the graded curve, and bits with their convention 🔼 HIGH VALUE · cost: one day

**Wrong.** The persisted "primary deliverable" FI_in(k) curve is the binary one and encodes a single number (§3.12); the headline ΔAUFI in bits scales with an arbitrary cap constant (§3.11).

**Do.**
- Persist and plot the **graded** FI_in curve (per-paraphrase F takes 11 values, so it is a real curve). Delete the binary curve from the deliverable, or keep it explicitly as the degenerate case.
- Re-run the Hazen stepped-shape / "islands of function" test **on the graded curve against a null** (permute F within cell; report how often the null produces as many steps). The current test has no null and, on the persisted column, no signal.
- Never print "Δ = −0.86 bits" without the cap convention. Either report the sensitivity table from §3.11, or replace the claim with the underlying quantity that needs no convention: **the fraction of quality thresholds that no paraphrase reaches falls from 56–67 % at L0 to 32–40 % at L1.** That statement is convention-free, is the actual finding, and is more legible than the integral.
- Report the censoring rate wherever AUFI appears.

**Target result.** An axis-1 figure that shows a real curve with a bootstrap band, a stepped-shape test with a null, and one convention-free headline number.

### R5 — Fix the factual errors, and re-position the FI framing ⚠️ BLOCKING · cost: one day

**Wrong.** Six verified factual errors in the deck/talk track, several of which `main_body.tex` already avoids.

**Do — corrections (all verified in this review):**

| where | says | should say |
|---|---|---|
| `EXPLAINER:195` | "Both levels are closed-book" | uniform evidence, 900/900 rows, mean 17.9 snippets |
| `EXPLAINER:106-108` | ρ_u is "embedding geometry of the **prompts**" | response embeddings, same generations; that is ESS_in they are thinking of |
| slide 26 | "ΔH_sem < 0 — all 3 models, p ≤ 5e-9" | p ≤ 5e-9 holds for Δaccuracy/ΔAUFI only; mistral ΔH_sem p = .034 (BH q = .051) |
| slide 30 / `final_run_results.md:14` | length baseline ".457, below chance" | 0.543; the 0.4569 is the flipped orientation |
| slide 31 | "First application of Szostak/Hazen to prompts" | Hazen 2007 §"Functional Information of Letter Sequences" did natural language |
| independence figure | "70 % by top-3 eigenvalues" | 75.5 %; and cite Horn's parallel analysis instead |
| `EXPLAINER:266` (already flagged internally) | cross-model ".27–.30" | ≈.2–.45 |

**Do — re-position FI.** Define the quantity as **generator-relative**: write `FI_in^G(q,k)`, name G in the notation, state that bits are comparable only within a fixed G, and cite Hazen's own requirement (all configurations + degree of function for every configuration) as the reason. Then the honest novelty is: *"we instantiate Hazen's letter-sequence construction with an LLM as the receiver and a controlled meaning-preserving proposal distribution, and we characterise what the measure does and does not carry over."* Also state the FI_out = −log₂(2^H_sem/\|A_q\|) identity (§3.1) so the "one ruler" claim is true for 3 of 4 components, and stop claiming ρ_F is on the ruler.

**Target result.** Zero factual errors; an FI section that survives a reviewer who reads Hazen 2007.

### R6 — Second paraphrase generator 🔼 HIGH VALUE · cost: one cluster run

**Wrong.** FI_in is currently "the fraction of *Phi-4-style* rephrasings that work". Larger N samples that distribution more precisely; it does not widen it. Romanou et al. warn LLM paraphrasing may *standardise* language and make queries easier.

**Do.** Add a rule-based/template generator (syntactic transformations, question-word substitution, clefting, politeness/register shifts, filler insertion) as a second G. Recompute FI_in, ρ_F and σ²_B under both. This is the experiment that makes R5's generator-relativity claim *empirical* rather than a caveat.

**Target result.** A two-column table (FI_in^Phi4 vs FI_in^rule) with the cross-generator correlation of ρ_F. If ρ_F correlates well across generators, the construct is generator-robust and the paper is much stronger. If it does not, that is a **major negative finding about the whole field**, since every published index picks one perturbation family.

### R7 — Rebuild the probe evaluation 🔼 HIGH VALUE · cost: analysis only

**Wrong.** Controls only at the shallowest layer; post-hoc layer selection; sign-flipped baseline; missing text baselines; no operating-point analysis (§3.6–3.9).

**Do.**
- Compute `control_permuted` and `baseline_length` **at every layer**, and use ≥100 permutations to get a null *distribution*, not one draw.
- Select the layer inside a nested CV loop, or report all layers with a multiplicity-corrected best.
- Add three baselines to every head: **TF-IDF word, TF-IDF char, and first-word-only**, each under **both** protocols (frozen and in-domain CV) — the §3.7 table.
- Report PR-AUC and a confusion matrix at the shipped threshold alongside AUROC (base rate 0.585).
- Add the missing obvious competitor: **ask an LLM "is this question ambiguous?"** in one call, and report it.
- Report the fragility/ρ_F head's near-chance result **in the deck**, as a finding about ρ_F's nature.
- Cite and compare against Kossen et al. 2024 (semantic entropy probes) — the dispersion head is their result.

**Target result.** Every head reported as *AUROC [95 % CI] vs permuted null vs length vs TF-IDF vs first-word vs LLM-ask*, under a stated protocol. The claim becomes: *"prompt-token representations carry ambiguity signal that transfers zero-shot; text baselines require in-domain labels to match it."*

### R8 — Statistical hygiene 🔧 · cost: analysis only

> **✅ RESOLVED 2026-08-08.** `scripts/stats_hygiene.py` → `data/stats_hygiene.md` (declared 12-test
> family, Holm+BH, question-clustered CIs, cross-model dependence + pooled tests, split-half table);
> `FORKING_PATHS.md`; main_body.tex §Aggregation rewritten. Disclosed failure: qwen's Δ_union is
> BH-significant but not Holm-robust.

Report effect sizes with question-clustered CIs everywhere (not just p-values); apply Holm/BH across the declared family and report which claims fail (§3.4); **treat model as a random effect or state that the three models are correlated measurements, not replications (§3.4b)**; stop reporting `fi_out_fixed` and `H_sem` as two results (identical test, §2.4); replace the k10-vs-k20 "stability" with the disjoint-paraphrase split-half reliability (§R2); add a **pre-registration / forking-paths appendix** listing every analysis choice made after seeing data (AUFI demoted at ρ=−1.00; reformulation gain rejected at .98; `fi_out_mean` → `fi_out_fixed`, which *reverses a significant result*, §2.4; Chao1 tried then dropped; evidence dial withdrawn; H_sem chosen as the axis-3 representative). Reviewers forgive disclosed exploration; they do not forgive discovering it themselves.

### R9 — Reproducibility 🔧 · cost: hours

> **✅ RESOLVED 2026-08-08.** `scripts/make_metric_corr.py` (canonical matrix generator; original
> archived; factor headline is now 74.6 %/Horn 3 on 14 vars); `scripts/make_run_manifest.py` →
> `data/run_manifest.json`; `fig_independence` raises on NaN; rejection counts persisted since the
> R6 sidecars (the historical v3 gap is documented, not recoverable).

Commit the script that generates `figures/v3_metric_corr.npy` (there is none); emit a run manifest recording the 150 selected question IDs, seeds, model revisions and filter counts; persist paraphrase **rejection** counts per level (currently only accepted rows are stored, so the level-symmetry of the NLI/gold gate is unverifiable); remove `np.nan_to_num(nan=0.0)` from `fig_independence` or make it raise.

### R10 — Related work 🔧 · cost: one day (plus verification)

Add: Cox (AAAI 2025), Romanou/BrittleBench, Kossen SEP, Kim "DETAIL Matters", Polo/PromptEval, AmbigQA's evaluation protocol, Hazen's letter-sequence section, and the Root-Bernstein/Zenil critiques of the Wong 2023 law.
**Before citing anything from §5 marked ⚠, verify it exists and says what is claimed** — a chunk of the sweep's 2026-dated hits could not be re-checked and agent-reported citations are not trustworthy without confirmation.

### R11 — Strategic: re-centre the paper 🎯

**The current framing** ("we invented a new metric and a new axis") maximises exposure to §2.3, §2.5 and §5, where prior art is thickest and the claims are weakest.

**The recommended framing** — *"Prompt-sensitivity metrics are not measuring what they claim, and here is the measurement model that shows it"* — is supported by exactly the work already done:
1. Most published indices reduce **analytically** to semantic entropy (R4 — a proof, not a correlation).
2. The one quantity that does not (ρ_F) is only measurable on ~56 % of items, and its independence is bounded, not established (R2, R3).
3. The field's standard evaluation of ambiguous questions conflates a **grading lottery** with model ability (R1) — demonstrated on the field's own benchmark, with the authors' own collision flag as a natural experiment.
4. Ambiguity is nonetheless linearly decodable from one forward pass and transfers zero-shot (R7).

This reframing costs no new data collection beyond R1 and R6, turns the three findings that currently threaten the paper into its contributions, and lands in a category (measurement critique / position) where a single dataset and three 7–8 B models is an accepted scope.

---

## 8. Suggested order of work

| phase | items | gate |
|---|---|---|
| 0 (today, free) | collision-split table (§2.1); §2.3/§2.4/§3.4/§3.11/§3.12 recomputations; R5 corrections | decide R11 framing with Thanos |
| 1 | R1 (verify cluster cache **first**), R2 | Δ_union determines the headline |
| 2 | R3, R4, R4b, R7, R8 | paper-ready analysis |
| 3 | R6 (second generator), R9, R10 | hardening |
| 4 | write Results + Discussion | — |

**One caution for whoever implements this.** Every number in §2–§4 was recomputed independently for this review, but the §5 literature items marked ⚠ were not. Verify those before they enter the paper.
