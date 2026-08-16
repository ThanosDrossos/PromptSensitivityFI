# Independently verified findings (computed by me, not by subagents)

All numbers recomputed from `Code/PromptSensitivityFI/data/*.parquet` with the repo venv.

## V1. The headline accuracy gain is quantitatively reproduced by a pure guessing lottery
Lottery null: at L0 the model answers *some* valid reading; graded correct only if it happens to be the
seeded target. Prediction: acc_L0 ≈ acc_L1/m0, Δ = acc_L1·(1 − 1/m0).

| model | obs acc_L0 | lottery-pred acc_L0 | obs Δ | lottery-pred Δ |
|---|---|---|---|---|
| llama_3_1_8b | 0.180 | **0.180** | +0.238 | **+0.238** |
| mistral_7b_v03 | 0.229 | 0.199 | +0.247 | +0.277 |
| qwen_2_5_7b | 0.286 | 0.218 | +0.225 | +0.293 |

Observed gain is at or BELOW the lottery prediction in all three models. spearman(m0, Δ) = −0.17/−0.13/−0.03
(lottery predicts positive). => "disambiguation roughly doubles accuracy" is not evidence about specificity;
it is arithmetic of grading an ambiguous question against 1-of-m0 answers.

## V2. rho_F orthogonality is a coverage-selection artifact + a per-model average that hides a failure
Coverage 45.3% / 66.0% / 57.3%. Excluded cells are the degenerate ones (68–87% all-wrong, 13–32% all-right)
=> range restriction on accuracy.

| model | rho_F~acc (covered only) | 95% CI | rho_F~acc (impute 0 on zero-variance) |
|---|---|---|---|
| qwen | +0.029 | [−0.140, +0.196] | **+0.256** (p=7e-6) |
| llama | **+0.224** (p=.0015) | **[+0.088, +0.353]** — excludes 0 | **+0.477** (p=2e-18) |
| mistral | +0.001 | [−0.149, +0.150] | **+0.408** (p=2e-13) |

".08" ≈ the mean of (.029, .224, .001). Llama's CI excludes zero. Equivalence bound supported is only ±0.20.

## V3. rho_F is NOT orthogonal to the dispersion *family* — .03 is the minimum, not a representative
rho_F vs variation_ratio = +.41/+.28/+.36 ; vs consistency_mean (1−TVD) = −.40/−.17/−.36 ;
vs fi_out_var = +.14/+.26/+.41. Against the dispersion FACTOR (PC1 of the 6 dispersion metrics, 69–78% var):
**+0.172 (p=.046) / +0.085 / +0.256 (p=.0007)**. Deck reports .03 (H_sem only).
rho_u (Cox), the "independent convergent channel", loads on the dispersion factor at +.30/+.15/+.25 — it is
not a clean second channel.

## V4. fi_out_fixed is exactly −H_sem shifted by a per-question constant
`fi_out_fixed = log2(m0) − h_sem_mean` with residual 0.0e+00; m0 has exactly 1 distinct value per question.
So every paired L0→L1 claim about FI_out_fixed IS the H_sem claim. Verified: identical Wilcoxon p-values
(3.74e-07 vs 3.76e-07 llama; 0.0113 vs 0.0112 qwen; 0.034 vs 0.034 mistral). Reporting both double-counts.

## V5. The "capacity share" story credits confident wrongness
spearman(fi_out_fixed, accuracy) = −0.016 / +0.116 / +0.282. Among "confident" cells (H_sem<0.2),
**47% (qwen) / 37% / 38% are WRONG** (acc<0.2). A model that confidently emits one wrong answer scores
maximal "capacity realized". Also the 1.429-bit "capacity" is arbitrary: the observed answer space is
1.39 (qwen) / **3.33 (llama)** / 2.81 (mistral) bits, so "11% of capacity" compares H_sem to a ceiling
unrelated to the actual output space. FI_out_fixed also goes negative — not a share.

## V6. "significant in all three models" fails under any multiplicity correction
27 paired Wilcoxon tests over the headline metric set; BH q<0.05 leaves 16.
Failures include **mistral ΔH_sem q=0.051**, qwen ΔS_τ q=0.25, llama Δvariation_ratio q=0.059,
and Δrho_f in all three (q=.29–.88). The blanket "p ≤ 5e-9" applies only to Δaccuracy/ΔAUFI.

## V7. "rho_F correctly non-responsive to the dial" is underpowered, not established
Paired n = 36 / 70 / 58 (both levels covered). 95% CI on Δrho_F = [−0.106,+0.143] / [−0.009,+0.087] /
[−0.071,+0.104], against metric means of .40/.13/.23. The CIs admit changes of 25–70% of the metric's own
magnitude. Minimum detectable effect at 80% power = 0.172 / 0.067 / 0.122.
(The between-model ordering qwen .40 > mistral .23 > llama .13 IS solid: paired Wilcoxon p=1e-11, 7e-5, 7e-5.)

## V8. The OOD length baseline of ".457, below chance" is a sign error
Exact reproduction (n=1852, n_ambiguous=1022 — matches the paper): AUROC(shorter=ambiguous)=**0.5431**;
0.4569 is the sign-flipped orientation. A baseline predictor takes its better orientation.
Honest baseline = 0.543 ⇒ head margin **+0.124**, not +0.21. "Collapses below chance" must be withdrawn.

## V9. But the probe DOES beat a text-surface baseline OOD — this helps them and is currently missing
Trained TF-IDF logistic on the same L0-vs-L1 paraphrase texts the heads trained on, evaluated on the same
frozen holdout: word 1-2gram AUROC **0.565**, char 3-5gram **0.587**, vs hidden-state head 0.655–0.670.
=> the head is not a lexical detector. This baseline should be added; it is a stronger defense than the
length baseline they currently use. (PR-AUC context: prevalence 0.585; TF-IDF PR-AUC 0.635–0.656.)

## V10. Probe controls are placed only at the weakest layer, and the null is wide
`control_permuted` and `baseline_length` exist ONLY at the shallowest layer in every
probe_results_* file; the headline layers (21/24, AUROC .72–.83) have NO control. A single permutation gives
control AUROC ranging **0.372 → 0.583** across targets ⇒ null SD ≈ .08, comparable to the claimed margins.
Best-layer is selected post hoc over 4 layers × 2 head types.

## V11. rho_F — the paper's own novel axis — is the least predictable quantity in the probe suite
Best-over-layers AUROC for rho_f: qwen .534/.555, llama .644, mistral .582 (vs .72–.83 for accuracy/H_sem).
The framework's flagship construct is near chance from the prompt representation.

## V12. Estimator checks that PASSED (do not raise these)
- Per-sample F really is binary 0/1 (`scoring/nli_with_gold.py:44-46`), so SS_within = k·p(1−p) is exact.
  The ICC estimator is correctly specified.
- Universe size is exactly 10 at both levels (170 cells L0, 169 L1); 1 singleton fallback total. No
  level asymmetry in universe size.
- AUFI ≡ accuracy confirmed: spearman −0.9997 / −0.9992 / −0.9993.
- `nan_to_num(nan=0.0)` in fig_independence is a landmine but the 12 variables actually plotted contain no
  NaN. (`spread (Cao)` is all-NaN in v3_metric_corr.npy but is excluded from the figure.)

## V13. Graded F is nearly binary in practice
Of 2991 per-paraphrase F values: 53% exactly 0, 32% exactly 1 — only 15% intermediate. The "graded" track
adds little; FI_in curves and rho_F both rest on that 15%.

## V14. The "one ruler" framing covers 2 of 4 components (fixable)
Eq.2 is FI = −log2(|{c: F(c)≥k}|/|C|). FI_in and FI_spec are literal instances. FI_out = log2|A| − H_sem
is NOT — it is KL-from-uniform. rho_F is a variance share, and the paper's own Table 1 lists its unit as
"share", contradicting "every metric is an instance of Eq. 2".
FIX for FI_out: it IS on the ruler if you write the surviving count as the perplexity (Hill number of
order 1): 2^H_sem. Then FI_out = −log2(2^H_sem/|A_q|) = −log2(effective surviving fraction). State this.
FIX for rho_F: stop claiming it is on the ruler; it is a second-order quantity ABOUT the F that axis 1
counts. Say so.

## V15. rho_F is temperature-scoped and the model ranking survives (but must be defended)
MSW is estimated from k=10 samples at T=1.0; at T=0 within-variance → 0 and rho_F degenerates. So rho_F is a
(question × model × temperature) quantity. Mean within-paraphrase p(1−p): qwen .0247 < mistral .0378 <
llama .0598 — the EXACT INVERSE of the rho_F ranking, so a reviewer will say the ranking is a decoding-noise
ranking.
CHECKED: it is not. The absolute noise-corrected variance component σ²_B = (MSB−MSW)/k gives the SAME order
(qwen .067 > mistral .039 > llama .0215). Report σ²_B alongside the share to kill this objection pre-emptively.

## V16. The x* geometry result is REAL but misses its own target and hides coverage
Observed mean xstar_rho = −0.325 (llama −.32/−.30, mistral −.23/−.36, qwen −.37/−.36), n=437 cells.
Null simulation (x*=argmax F, F drawn from the real empirical F distribution, random unit embeddings,
4000 reps, d=64 and d=1024): null mean = +0.002 ± 0.011, sd 0.35. So the effect is ~18 SE from zero — genuine.
BUT: (a) the pre-registered target in `analysis/x_star.py` was mean rho ≤ **−0.4**; observed is −0.325, i.e.
the criterion FAILS; (b) coverage is 437/894 = 49% of cells, undisclosed; (c) per-cell sd (0.35–0.42) equals
the null sd (0.35), so ONLY the mean carries signal — no per-question claim is licensed.

## V17. Paper Methods vs deck/data inconsistencies
- Methods §Aggregation: "Spearman correlations computed WITHIN a specificity level". The deck's headline
  .08/.03 are pooled. Within-level, llama L0 rho_F~accuracy = **+0.259 (p=.019)**. The paper's own stated
  protocol contradicts the deck's number.
- Methods §probes: "We report permuted-label and prompt-length baselines for every head." In the data,
  `control_permuted` and `baseline_length` exist ONLY at the shallowest layer; the headline layers have none.
- Methods §Aggregation: "question pairs whose two levels yielded different numbers of paraphrases were
  excluded" — a no-op: every universe is exactly 10 at both levels.
- Methods §Clustering: Chao1 correction was tried and dropped for overestimating. Observed richness is
  downward-biased and k-dependent; this needs a sensitivity analysis, not a one-line dismissal.
- Methods §Levels DOES pre-empt the fixed-gold objection on design grounds, but never quantifies it — see V1.

## V18. The factor structure PASSES the proper test — but the test is weaker than it looks
Horn parallel analysis (2000 random matrices, n=136, p=13): observed eigenvalues 5.77 / 2.44 / 1.60 vs
random 95th-pct 1.69 / 1.50 / 1.38 => **exactly 3 factors retained**. This is the right test and they pass it.
Cumulative variance top-3 = **75.5%** (deck says 70%).
Varimax structure is clean:
  F1 dispersion: H_sem .98, S_τ .92, TVD .91, |A_q| .90, var-ratio .77, Var[FI_out] .71, FI_out_fixed −.72
  F2 ability:    accuracy .95, AUFI −.95, ΔFI premium .73
  F3 sensitivity: rho_F −.87, rho_u −.85, ESS_in −.48
Three problems a reviewer will raise anyway:
  (a) The metric SET is curated: 6 of 13 are known-duplicate dispersion indices and 2 are accuracy and its
      exact monotone transform (r = −1.00). F1 and F2 are guaranteed by construction. Only F3 is non-trivial.
  (b) F3 has exactly TWO real members (rho_F, rho_u). A 2-indicator factor is weakly identified — and the
      paper separately cites rho_F~rho_u = .67 as independent "convergent validity". You cannot both define
      a factor by two variables and cite their correlation as external corroboration of that factor.
  (c) **ESS_in loads −0.48 on F3**, contradicting the claim "ESS_in ⊥ all". It is a weak third sensitivity
      indicator, not a null diagnostic.
  (d) Cross-loadings exist: TVD −.33 and variation ratio −.38 on F3 — consistent with V3.
FIX: run the analysis on a non-curated metric set, or reframe as CONFIRMATORY factor analysis with fit
indices (CFI/RMSEA) and report the parallel analysis + loadings table instead of "70% of variance".

## V19. Operational note for any re-scoring experiment
`data/cache/llm_cache.sqlite` holds only 6,471 rows and they are all gateway-era (`kit.gpt-4.1`,
`meta-llama-3.1-8b-instruct` POSIX probes). The v3 responses are NOT in the local cache — they are on
bwUniCluster. Any re-scoring arm (e.g. the union-gold control) is a cluster job, but it is a RE-SCORING job,
not a regeneration job, IF the cluster cache is intact. Verify that before scoping.

## V20. "POSIX measures dispersion, not phrasing sensitivity — axis 2 stays empty" is NOT statistically supported
Slide 28 / contribution #3. Two problems.
(a) **Apples to oranges.** The reported POSIX~H_sem values (.63/.43/.61) are computed on all n=100 cells;
    the POSIX~rho_F values (.24/.30/.32) are necessarily computed on the rho_F-covered subset only
    (n = 42 / 62 / 62). Recomputed on the SAME subset, POSIX~H_sem drops to **.41 / .38 / .56**.
(b) **The difference is not significant.** Williams/Steiger test for dependent correlations on the common
    subset:

| model | n | POSIX~H_sem [95% CI] | POSIX~rho_F [95% CI] | difference |
|---|---|---|---|---|
| qwen | 42 | +0.409 [+0.119,+0.634] | +0.237 [−0.072,+0.505] | t=0.82, **p=0.42** |
| llama | 62 | +0.380 [+0.143,+0.575] | +0.297 [+0.051,+0.509] | t=0.48, **p=0.63** |
| mistral | 62 | +0.560 [+0.360,+0.710] | +0.318 [+0.074,+0.526] | t=1.97, **p=0.054** |

POSIX~rho_F CIs **exclude zero** for llama and mistral. So the data do not show that POSIX belongs to axis 3
rather than axis 2 — it correlates with both, and the two correlations are indistinguishable at n≈50.
=> Contribution claim #3 ("the phrasing axis stays empty") must be withdrawn or restated as
"POSIX correlates more strongly with dispersion, but at this sample size we cannot rule out that it also
captures formulation sensitivity."

## V21. AmbigQA's own evaluation protocol contradicts the L0 design (read from the local PDF, §3.1-3.2)
Min et al. 2020, §3.1: "output a set of semantically distinct and **equally plausible** answers y_1..y_n";
§3.2 defines correctness as a SET F1 over predicted (question, answer) pairs against the gold reference set:
c_i = max_j 1[y_i in Y_j] f(x_i, x_j); prec = Sum c_i / m; rec = Sum c_i / n.
=> The dataset's own protocol for an ambiguous question grades against ALL interpretations. Grading a single
response against ONE seeded interpretation is non-standard and depresses L0 by construction — exactly the
lottery of V1.

**The paper's stated defense is logically wrong.** Methods §Levels says crediting any interpretation "would
change the gold set between levels". It would NOT: scoring BOTH levels against the union keeps the gold set
identical across levels, which satisfies their own fixed-gold guardrail exactly. The second half of the
defense ("answering *some* reading cannot respond to disambiguation") is an empirical assumption, not a fact —
disambiguation could improve grounding and lift union-accuracy too.
=> The union-gold arm is the missing control, it is cheap (re-scoring only), and it is what makes the
headline interpretable: Delta_union is the non-tautological part of the effect; Delta_target − Delta_union is
the targeting/lottery part.

## V22. The bag-of-words objection — RESOLVED, both protocols matter (fleet's "FATAL P1" is overstated)
The review fleet trained TF-IDF with 5-fold CV *inside the holdout* (using holdout labels) and got .655,
concluding "the hidden state buys nothing". That is a different label budget from the frozen head. Matched:

| protocol | word 1-2gram | char 3-5gram | first-word only | hidden-state head |
|---|---|---|---|---|
| **A: frozen, no holdout labels** (matches the head) | 0.544 | 0.562 | — | **0.667** |
| **B: in-domain 5-fold CV on holdout labels** | 0.655 | 0.663 | 0.650 | 0.667 |

Correct verdict (MAJOR, not FATAL): the head's real advantage is **zero-shot transfer / label efficiency**,
not detection quality. With ~1.5k in-domain labels a bag-of-words model — or literally the question's first
word — matches it. Both baselines must be in the paper, and the claim must be restated accordingly.

## V23. rho_u is NOT an independent channel — the talk track is factually wrong about the repo's own code
`orchestrator.py`: "# rho_u on the per-paraphrase **response** embeddings"; `rho_u(response_embeddings)`.
`metrics/rho_u.py:5-18`: U_e/U_t over "N paraphrases x k sampled **responses** ... embeddings of the
*responses*".
=> rho_u is the SAME variance-ratio design on the SAME N x k generations that produce rho_F, differing only
in (a) embedding vs binary correctness readout and (b) NO noise correction.
`EXPLAINER_Three_Dimensions.md:106-108` claims "rho_u ... is computed from *embedding geometry of the
prompts*. They agree at rho = .67 — two different measurement channels". **False.** (The prompt-embedding
quantity is ESS_in, `ess_in_fn(prompt_embeddings)` — the talk track appears to have conflated the two.)
The **paper's Methods gets this right** ("epistemic share of the total variance of response embeddings",
plus the correct null expectation (N−1)/(Nk−1)). Fix = delete the deck claim, keep the paper's wording, and
downgrade .67 from "convergent validity across channels" to "two readouts of the same responses agree".

## V24. "One construct, many costumes" is, for 4 of 6 members, ALGEBRA — not empirical convergence
Verified over all 900 rows:
- `fi_out_var` ≡ `h_sem_var`, max abs difference **4.44e-16**. (FI_out = log2|A| − H_sem with |A| fixed per
  cell ⇒ Var[FI_out] = Var[H_sem] identically.) The reported ".70 correlation" is a correlation of a variable
  with itself.
- `s_tau_mean` ≡ `h_sem_mean / log2(a_q)`, **max abs diff 0.0000, Spearman +1.0000**. Errica's S_τ is
  normalised semantic entropy by definition. The reported ".94" is not a measurement.
- `a_q` is the normaliser inside both FI_out and S_τ; `variation_ratio` and `consistency_mean` (1−TVD) are
  computed from the SAME pooled `cluster_assignments` object as H_sem (orchestrator.py).
- Only POSIX is a genuinely separate measurement (separate teacher-forced pass) — and per V20 it does not
  discriminate axis 3 from axis 2.
- Side issue: `s_tau_mean` is exactly 0 in **14.6%** of cells, 94.7% of which have a_q ≤ 1 (degenerate
  normaliser), which inflates the apparent agreement.
**CONSTRUCTIVE REFRAME (this makes the contribution stronger, not weaker):** stop presenting these as
measured correlations. Prove them as one-line *analytic reductions* — "S_τ = H_sem/log2|A|; Var[FI_out] =
Var[H_sem]; FI_out_fixed = log2 m0 − H_sem" — in a small table. An analytic identity is a far stronger claim
than an n=150 correlation, and it is honest. Reserve the empirical correlation claim for POSIX and for any
index you did not implement yourself.

## V25. THE SMOKING GUN — the collision subgroup is a natural experiment that falsifies the causal reading
The pipeline already emits `target_collision` (the pinned target's answer is shared with another
interpretation, so **the lottery cannot be lost**). Splitting the paired L0→L1 deltas on that flag:

| subgroup | cells | questions | mean Δaccuracy | 95% CI (question-clustered) | Wilcoxon |
|---|---|---|---|---|---|
| lottery LIVE (no collision) | 405 | 135 | **+0.268** | [+0.208, +0.328] | p = 5e-34 |
| lottery PRE-WON (collision) | 45 | 15 | **−0.044** | [−0.135, +0.035] | p = 0.43 |

Difference +0.312, Mann-Whitney **p = 3.8e-07**. Where the grading lottery is removed by the data itself,
the headline effect disappears entirely. Combined with V1 (the lottery model predicts the observed L0
accuracy to within 0.000–0.07) and V21 (AmbigQA's own protocol scores against all interpretations), the
"specificity improves ability" reading is not supported.
Fairness caveat to state: n=15 collision questions, and collision questions may be systematically different
(two readings sharing an answer ⇒ possibly more redundant questions). It is suggestive, not decisive on its
own — but it converges with two independent lines of evidence.

## V26. The talk track claims the run was closed-book; it was not (paper is correct, deck is not)
`EXPLAINER_Three_Dimensions.md:195`: "Both levels are closed-book, so retrieved context cannot sneak in as a
confound." Actual v3 data: `context_mode == 'uniform_evidence'` for **900/900 rows**, mean **17.9** evidence
snippets per cell (min 2, max 32), zero rows with 0 snippets. The paper's Methods §Evidence describes the
evidence bundle correctly. Delete the sentence from the talk track.
Substantive consequence: with the evidence-dial numbers (Δ = +0.02 closed-book → +0.235 full evidence), the
finding's true scope is *answer selection from supplied evidence*, not model ability — which is exactly what
the lottery reading says.

## V27. VERIFIED FROM THE SOURCE PDF — Hazen et al. (2007) already applied FI to natural language
Read directly from `Hazen et al. - 2007 - Functional information and the emergence of biocomplexity.pdf`,
pp. 8575–8576. The paper has a titled section **"The Functional Information of Letter Sequences"**:
- configuration space = sequences of n letters (26^n);
- degree of function E_x = "the probability that a local fire department will understand and respond to the
  message" — i.e. **the probability that a natural-language string evokes the intended response from a
  receiver**;
- I(E_x) = −log2[M(E_x)/26^n]; worked example ~1000 of 26^10 ten-letter sequences ("FIREONMAIN",
  "MAINSTFIRE", "MAPLENMAIN") ⇒ **I ≈ 36 bits**;
- it even discusses degraded variants — "phonetic misspellings (FYRE or MANE), mistakes in grammar or usage
  (FIREOFMAIN) or typing errors" — yielding lower probability of response.

That is structurally the paper's own construction with the receiver swapped from a fire department to an LLM.
=> The deck's "First application of the Szostak/Hazen formalism to prompts" and the talk track's "A
literature search for FI applied to prompts returns nothing. That gap is the paper" **overclaim against the
project's own foundational citation.** A reviewer who opens Hazen 2007 — which the paper cites — finds this in
the first worked example. This is reputationally the worst kind of error.

**Also a hard methodological constraint from the same source.** Hazen: "rigorous analysis of the functional
information of a system with respect to a specified function x requires knowledge of two attributes: (i) all
possible configurations of the system ... and (ii) the degree of function x for every configuration."
FI_in satisfies NEITHER: U_q is 10 samples from Phi-4, not the configuration space. So FI_in is not Hazen's
I(E_x); it is a **generator-relative** survival rate.
**CONSTRUCTIVE FIX (turns the weakness into a contribution):** define the quantity explicitly as
FI_in^G(q,k) with the proposal distribution G in the notation, state that bits are comparable only within a
fixed G, and make generator-sensitivity an *experiment* (second, rule-based generator) rather than a
limitation bullet. Then the honest novelty is: "we instantiate Hazen's letter-sequence construction with an
LLM as the receiver and a controlled meaning-preserving proposal distribution, and we show what the measure
does and does not carry over."
