# The three-axes slide — talk track

> # ⚠️ CORRECTED 2026-08-07 — READ THIS BANNER FIRST
>
> The 2026-08-06 adversarial review (`../../REVIEW_2026-08-06_Adversarial.md`) and the
> R1–R3 fixes (`../../RESULTS_R1_R2_R3_2026-08-07.md`) superseded several headline
> numbers below. The R5-corrected claims are patched inline and marked **[R5]**; the
> current claim language lives in `../../PAPER_PROPOSAL_2026-08-07.md`. The four claims
> that changed most:
> 1. The accuracy headline is now **Δ_union = +0.06…+0.13** (union-gold scoring); the old
>    +0.22…+0.25 was 47–72 % grading lottery (quantified in the R1 arm; goes in Methods).
> 2. **"Orthogonal" is withdrawn** — say "distinct, non-redundant": all associations
>    ≤ |0.14| under the hierarchical estimator, equivalence establishable only to
>    0.33–0.45, and ρ_F has a consistent modest positive association with the
>    dispersion *family* (6/6 model×level, p = .031).
> 3. **The "costumes" are identities, not correlations** — `data/metric_reductions.md`.
> 4. **The novelty claim is generator-relative FI, not "first FI on prompts"** — Hazen
>    2007 §"The Functional Information of Letter Sequences" already did natural language.

**What this file is:** everything needed to present the "3 axes + the dial" slide to
someone who does not know the project, plus the verification status of every number on
it. Numbers recomputed 2026-07-27 from `data/specificity_v3_<model>.parquet`
(149 questions × 2 specificity levels × 3 models), `figures/v3_metric_corr.npy`,
`data/sensitivity_v2_k20_*.parquet`, `data/posix_arm_qwen_2_5_7b.parquet`.

---

## 1. The thesis of the slide, in one sentence

> Prompt sensitivity is not one number: the metric zoo collapses into **three
> statistically independent axes**, which we move with **one dataset-side dial**.

The slide's job is to convert "here are seven metrics" into "here are three questions,
each with one representative metric and a family of near-duplicates behind it." The
persuasive part is that the three axes are *empirically* independent — you cannot infer
one from another — and that most of the published literature lands inside a single one
of them (Axis 3).

### The unifying formula

Everything is the same information-theoretic ruler:

> **bits = −log₂( surviving fraction )**

- **FI_in** = −log₂(fraction of *paraphrases* that work) → Axis 1's bits-view
- **FI_out** = log₂|𝒜| − H_sem = −log₂(fraction of the *answer space* still in play) → Axis 3
- **FI_spec** = log₂(m₀/m_valid) = −log₂(fraction of *interpretations* surviving) → the dial

Different possibility spaces, one ruler. That is the intellectual coherence of the stack.

### Where the ruler comes from (the novelty claim)

- **Szostak (2003, Nature 423:689):** functional information = −log₂ P(a random sequence
  performs the function). His ATP-binding RNA aptamer: ~1 in 10¹¹ random 70-mers → ≈ **37 bits**.
  FI is a property of *the ensemble ranked by function*, not of one object.
- **Hazen, Griffin, Carothers & Szostak (2007, PNAS 104:8574):** **I(E_x) = −log₂[M(E_x)/N]**
  — M configurations reaching function ≥ E_x, out of N possible.
- **Wong, Cleland, Hazen et al. (2023, PNAS 120:e2310223120):** generalized it to *any*
  (configuration space, scalar function) pair under selection. **This is the step that
  licenses using it outside biology** — without it, "that's a biology formula" is fatal.
- **Our substitution:** biopolymer sequence → prompt string; degree of function → answer
  correctness; sequence space → the paraphrase universe of one query.

**[R5]** ~~A literature search for FI applied to prompts returns nothing. That gap is the paper.~~
**Corrected:** Hazen et al. (2007), pp. 8575–8576, has a titled section *"The Functional
Information of Letter Sequences"* — FI over natural-language strings scored by whether a
receiver responds as intended ("FIREONMAIN" ≈ 36 bits), including degraded variants. Our
construction is that construction with the receiver swapped for an LLM. The honest claim:
a **generator-relative** instantiation FI^G with a controlled meaning-preserving proposal
distribution — bits comparable only within a fixed G (Hazen's own requirement — all
configurations + degree of function for each — is not met by any sampled universe).

---

## 2. Axis by axis

### AXIS 1 — ABILITY · headline: the FI_in(k) curve + accuracy

**The question:** *how well does the model do, and how much phrasing luck does that take?*

Fix a query `q`. Let `U_q` be its paraphrase universe (LLM-generated, NLI-filtered so
meaning is genuinely held constant), `F(x) ∈ [0,1]` the degree of function of paraphrase
`x`, and `k` a threshold. With `N_k(q) = |{x ∈ U_q : F(x) ≥ k}|`:

> **FI_in(q, k) = −log₂( N_k(q) / |U_q| )**

| value | meaning |
|---|---|
| 0 bits | every paraphrase reaches quality `k` — robust |
| 1 bit | half the paraphrases work |
| log₂\|U_q\| | one "magic phrasing" only — maximal sensitivity |
| ∞ | nothing works at that threshold |

Report it as a **curve over k**, not a scalar (Hazen Figs. 2.1–2.2 predict a *stepped*
shape — "islands of function"; seeing steps in prompt space is a direct analogue).

**Why this box says "ability", not "sensitivity"** — the point most likely to be
challenged, so say it first: the curve's scalar summary `AUFI_in = ∫₀¹ FI_in(q,k) dk`
correlates with graded accuracy at **ρ = −1.00**. Under binary scoring
`AUFI_in = 0.975·(−log₂ accuracy)` exactly. So AUFI *is* accuracy in a log wrapper; it
goes to the appendix, and the curve stays as the primary visual because it is the
readable, bits-native picture of ability.

**Also in this family:** ΔFI reliability premium (ρ = .51 with accuracy — the curve's
tail; secondary), reformulation gain log₂(F_max/F̄) (ρ = .98 → rejected, same disguise).

**[R5 — corrected] Headline result:** the honest, convention-free statement is: *the
fraction of quality thresholds no paraphrase reaches falls from 56–67 % (ambiguous) to
32–40 % (disambiguated)* — never quote ΔAUFI in "bits" without its cap convention (the
delta ranges −0.57 to −1.10 across defensible caps; `data/axis1_graded_curve.md`). And
~~"accuracy roughly doubles"~~ is retired: under union-gold scoring (the dataset's own
protocol) the real gain is **+0.06 / +0.13 / +0.12** (BH-significant, 3/3 models); the
rest of the naive doubling was the 1-of-m₀ grading lottery (47–72 %), reported in Methods.

### AXIS 2 — FORMULATION SENSITIVITY · headline: ρ_F (functional ICC)

**The question:** *of the variation in whether the model succeeds, how much is caused by
which phrasing you happened to pick — as opposed to random decoding noise?*

This is **the novel axis** and the one the project is really about. A one-way
random-effects ICC(1) on the N×k correctness outcomes grouped by paraphrase:

    SS_between = k·Σᵢ(F̄ᵢ − F̄)²,   SS_within = Σᵢ k·F̄ᵢ(1 − F̄ᵢ)
    ρ_F = (MSB − MSW) / (MSB + (k−1)·MSW),   clamped to [0,1]

ρ_F = 0 → rephrasing is irrelevant, all wobble is sampling noise. ρ_F = 1 → success is
fully determined by phrasing.

**[R5 — corrected] Why it earns its own axis:** ~~⊥ ability (.08), ⊥ dispersion (.03)~~ —
those were complete-case, cross-model averages; the honest numbers
(`data/independence_target.md`): under the hierarchical estimator every within-model,
within-level association is **≤ |0.14|** (both gold sets), but equivalence is establishable
only to |ρ| < 0.33–0.45, and ρ_F has a consistent modest positive association with the
dispersion *family* (6/6 model×level positive, p = .031). Say **"distinct and
non-redundant," never "orthogonal."** The stronger separation evidence is the
**dissociation**: the dial moves accuracy in 3/3 models and ρ_F in 0/3 (properly powered,
n = 150, both golds). Compare AUFI's .999 with accuracy.

**[R5 — corrected] Convergent validity, one data pass, two readouts:** ρ_F is computed
from *behaviour* (correct/incorrect outcomes); **ρ_u (Cox et al. 2025)** is computed from
the embeddings of the ***responses*** — the *same* N×k generations, not prompt geometry
(that quantity is ESS_in; see `metrics/rho_u.py:5-8`, `orchestrator.py`). Their agreement
at **ρ = .67** is therefore an *upper bound* on true construct agreement (sampling error
is shared), not independent-channel corroboration. Do not say "two independent channels."

**Why published sensitivity scores don't belong here:** PSS / ProSA-style indices measure
*raw dispersion* of outcomes across prompts without separating decoding noise from
phrasing effects. That is why they load on Axis 3, not here. The noise correction is the
methodological contribution of this axis.

**[R5 — corrected] Reliability:** ~~k=10 vs k=20 gives ρ = .81/.92/.95~~ — that comparison
is invalid as a reliability: the k=20 set *contains* the k=10 responses (0/1000 violating
paraphrases), so it correlates a statistic with a superset of itself. The honest number is
the **disjoint-paraphrase split-half** (mean of 200 random 5/5 splits, Spearman-Brown):
**0.35 / 0.52 / 0.58** (qwen/llama/mistral). Per-question ρ_F claims are therefore not
supportable; population-level and model-level claims are. Cross-*model*
agreement is much weaker (≈ .2–.45 depending on level and pair) while accuracy transfers
at .77–.82. **Read that correctly:** sensitivity is not a pure property of the question;
it is a **(question × model) interaction** — one model's awkward phrasing is another's
easy one (Section_7 §7.3.4).

**Correctly non-responsive:** ρ_F barely moves with disambiguation (Qwen .397 → .403).
That is a feature. Disambiguation raises *ability*; the *share* of variance owed to
phrasing is a separate trait. A sensitivity metric should not behave like a difficulty
metric.

### AXIS 3 — OUTPUT DISPERSION · headline: H_sem (+ the FI_out_fixed reading)

**The question:** *how scattered are the model's answers — and how much of the available
precision does it actually use?*

Sample the model k times on one prompt, cluster responses by meaning (bidirectional NLI
entailment), take the entropy of that cluster distribution: **semantic entropy H_sem**
(Farquhar, Kossen, Kuhn & Gal, *Nature* 630:625, 2024). Then

> **FI_out(x) = log₂|𝒜_q| − H_sem(Y | X = x)**

(the `log N − H`, KL-from-uniform form). High = the prompt pins one meaning down;
0 = the model spreads over everything it knows. The alternative definition
`−log₂ P[correct]` was **rejected** (Section_7 §7.4.1): its reading inverts, scoring bad
prompts as high-information.

**[R5 — corrected] "One object, five readouts" — identities, not correlations.** Presenting
these as measured agreements was wrong in both directions: S_τ **≡** H_sem/log₂|𝒜|
(max abs err 1.7e-16) and Var[FI_out] **≡** Var[H_sem] (4.4e-16) are *exact relabelings*;
|𝒜_q|, variation ratio and 1−TVD are functionals of the *same* pooled cluster
distribution. Say: *"five published indices are provably functions of one object — the
semantic cluster distribution"* (`data/metric_reductions.md` has the Proposition). The one
independently measured index, POSIX ψ, correlates with **both** dispersion and ρ_F, and
the difference is not significant (Williams p = .42/.63/.054) — it does not discriminate
between the axes. Never say "the phrasing axis stays empty."

**⚠ The trap you must pre-empt — the moving yardstick.** Raw `fi_out_mean` *falls* with
disambiguation (Llama 2.47 → 2.08), which looks like it contradicts the hypothesis. It
doesn't: `|𝒜_q|` is estimated per cell from observed clusters and shrinks too
(19.9 → 11.8), so the ruler shortens with the thing measured. Hold the reference fixed
(`fi_out_fixed`) and the expected direction appears everywhere:

| model | FI_out_fixed L0 → L1 | H_sem L0 → L1 | share of capacity realized |
|---|---|---|---|
| Llama-3.1-8B | 0.16 → 0.59 | 1.27 → 0.84 | 11% → 41% |
| Mistral-7B-v0.3 | 0.55 → 0.69 | 0.88 → 0.74 | 38% → 49% |
| Qwen-2.5-7B | 1.05 → 1.17 | 0.38 → 0.26 | 74% → 82% |

**Always use `fi_out_fixed` for cross-level claims.** H_sem falls in all three models,
which is the same finding without any yardstick problem.

**"FI_out_fixed = its calibration reading against FI_spec"** — the last column above is
the payoff. Capacity = mean log₂(m₀) = **1.429 bits**, i.e. exactly the specificity the
dial supplies. The share tells you **how much of the precision it was handed the model
actually converts into a concentrated answer**. Llama wastes most of it when the question
is vague (11%) and improves to 41% once disambiguated; Qwen is well-calibrated from the
start (74% → 82%). That is a per-model calibration statement no single-axis metric gives.

### THE DIAL — FI_spec (dataset-side)

**The question:** *how much ambiguity does the question text itself remove?* — no model
involved. This is the **x-axis of the experiment, not an outcome.**

> **FI_spec = log₂( m₀ / m_valid )**

with `m₀` = number of valid interpretations of the original ambiguous question, and
`m_valid` = number the current wording still admits.

| level | question | m_valid | FI_spec |
|---|---|---|---|
| 0 | original ambiguous `Q` | m₀ | log₂(m₀/m₀) = **0 bits** |
| 1 | disambiguated `Q_i` | 1 | **log₂ m₀** bits |

**Source of m₀:** AmbigQA (Min, Michael, Hajishirzi & Zettlemoyer, EMNLP 2020) — real
Google queries annotated with their distinct valid interpretations and each one's answer.
(About half of Natural Questions are ambiguous — that is the practical motivation for the
whole axis.) In our data m₀ averages 2.97, so level 1 carries **1.43 bits** (range 1.0–3.32).

**The guardrail that makes the whole design valid — state this unprompted:** the gold
answer `a_i` is **fixed across both levels**; only the question text changes. Without it,
disambiguation would move the target and any accuracy gain would be a grading artifact.
**[R5 — corrected]** ~~Both levels are closed-book~~ — **false**: every v3 cell runs with
the uniform evidence bundle (`context_mode = uniform_evidence`, 900/900 rows, mean 17.9
snippets). Say instead: *both levels receive the identical evidence bundle, so retrieved
context cannot differ between levels.* And since the R1 arm: the target-only grading is
itself a lottery worth 47–72 % of the naive gain — the primary effect is now **Δ_union**
(+0.06…+0.13), scored against the union of all interpretations' answers at both levels.

**Why "model-free" matters:** FI_spec is identical across all three models by
construction, so it is a true independent variable. **ESS_in** (effective sample size of
the paraphrase universe) sits here as a sanity check on the universe text itself and is
⊥ everything (|ρ| ≤ .17 with all axes; .27 with ρ_u).

---

## 3. The one-minute delivery

> "Prompt sensitivity has been reported as a zoo of metrics. We show it collapses into
> three independent questions. **Axis 1, ability:** how well does the model do — the
> FI_in curve is the bits-view; its scalar turns out to be accuracy in a log wrapper, so
> we keep the curve and demote the scalar. **Axis 2, formulation sensitivity:** what
> share of success variance is caused by *phrasing* rather than noise — that is ρ_F, it's
> orthogonal to the other two (.08 and .03), and it's confirmed through a second,
> embedding-based channel at .67. **Axis 3, output dispersion:** how scattered the answers
> are — semantic entropy, and essentially every published sensitivity index we tested is
> a costume for it, at .60 to .94. Then **the dial:** AmbigQA lets us add a measured
> number of bits of specificity to the question, model-free, with the gold answer held
> fixed. Turn the dial 1.4 bits and ability roughly doubles, phrasing-luck cost drops by
> ~0.9 bits, answers concentrate — and the fraction of that precision each model actually
> exploits, from 11% to 82%, is a calibration profile that no single metric would show."

---

## 4. Questions you will get

**"Isn't FI_in just accuracy?"** Its *integral* is (ρ = −1.00) — we say so on the slide
and demote AUFI to the appendix. The curve is retained as the presentation of the
construct, and the sensitivity claim rests on ρ_F, which is .08 with accuracy.

**"Why an ICC?"** Because a raw spread across paraphrases confounds two sources: the
phrasing and the sampler. ICC(1) is the standard variance decomposition that separates
them, and it is the F-space analogue of Cox's ρ_u, which is why the two channels agree.

**"Is 149 questions enough?"** For the population-level claims, yes — the level effects
are significant in all three models (e.g. FI_out_fixed p = 6e-7 for Llama). For
*per-question* claims it is thin: cross-model agreement of ρ_F is ~.2–.45, and coverage
is partial because ρ_F is undefined on all-correct/all-wrong cells (no variance ⇒
sensitivity unmeasurable). Both are reported, never hidden.

**"Why is ρ_F undefined for some questions?"** If every paraphrase always succeeds or
always fails, there is no variance to decompose. That is principled, not a bug — but it
means coverage must always be reported alongside.

**"Does this transfer across models?"** Ability does (.77–.82). Sensitivity does much
less (~.2–.45) — which is the finding, not a defect: it is a (question × model) trait.

**"Where does POSIX ψ = .60 come from?"** A separate 100-question arm on Qwen
(`data/posix_arm_qwen_2_5_7b.parquet`), because POSIX needs per-token log-probabilities
that the main run did not store; ψ vs H_sem = .63, ψ vs S_τ = .59. The slide labels it
"new arm" for exactly this reason.

---

## 5. Verification status of every number on the slide

| slide claim | verified value | status |
|---|---|---|
| AUFI ≡ accuracy, ρ = −1.00 | −1.00 | ✅ |
| reformulation gain rejected, ρ = .98 | .981 | ✅ |
| ΔFI premium (secondary) | .51 with accuracy | ✅ |
| ρ_u (Cox) embedding twin, ρ = .67 | +.67 | ✅ |
| ρ_F ⊥ ability (.08), ⊥ dispersion (.03) | +.08, +.03 | ✅ |
| k10↔k20 stability .81–.95 | .81 / .92 / .95 | ✅ |
| S_τ .94 · TVD .91 · \|A_q\| .91 · var-ratio .75 · Var[FI_out] .70 | .94 / .91 / .91 / .75 / .70 | ✅ |
| POSIX ψ .60 (new arm) | .63 vs H_sem (n=100, Qwen only) | ✅ |
| FI_spec model-free, identical across models | mean 1.429 bits in all 3 | ✅ |
| ESS_in ⊥ all | \|ρ\| ≤ .17 (except .27 with ρ_u) | ✅ |
| **cross-model .27–.30** | **within-level .39/.42/.46 (L0), .17/.30/.37 (L1); pooled .25–.39** | ⚠ **soften** |

**The one fix:** the cross-model band is not .27–.30. Level 1 happens to average .28,
which is likely where the number came from, but level 0 runs .39–.46. Say instead:
**"cross-model ≈ .2–.45, versus .77–.82 for accuracy → a (question × model) trait."**
The interpretation on the slide is unchanged and still correct; only the printed range
needs widening. Everything else can be defended as printed.
