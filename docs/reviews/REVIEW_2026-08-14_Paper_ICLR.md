# Adversarial review of the paper draft — ICLR reviewer perspective

**Under review:** `Paper/SensitivityFunctionalInformationPaper/{main.tex, main_body.tex, appendix.tex, references.bib}`
(compiled `main.pdf`, 30 pages, clean build, zero LaTeX warnings, zero overfull boxes, no dangling citations)
**Date:** 2026-08-14 · **Stance:** hostile but fair, ICLR standards, everything checked against the artifacts
**Predecessor:** `REVIEW_2026-08-06_Adversarial.md`. R1–R9 all landed; this review deliberately does *not*
re-report anything that review already fixed.

**How it was produced.** Seven independent reviewers were run over seven dimensions — statistics, design,
construct validity, novelty/citations, probes, number verification, venue fit — each followed by an
adversarial verifier instructed to *refute* its own dimension's findings from the paper text, the artifacts
and the code (14 agents, ~2.3 M tokens). Findings that were refuted were dropped; findings that were
downgraded appear here in their downgraded form. §9 lists what was checked and cleared, so the same ground
does not get re-litigated. The load-bearing findings were additionally reproduced by hand — §2.5 with a
fresh recovery simulation through the project's own estimator, with the script inlined so it can be re-run.

---

## 1. Verdict

**Score: 4/10 — reject as framed, major revision away from a defensible accept.**
Confidence: high on the technical findings (all recomputed), medium on the framing judgement.

This is a serious, unusually self-critical measurement paper, and the current draft is markedly more honest
than any previous artifact in the project. The statistical hygiene — declared test family, Holm + BH,
question-clustered CIs, models treated as correlated measurements, a mechanical-coupling null that
disqualifies the paper's own most tempting correlation — is better than most published work in this area.
Five things nevertheless block it, and all five are fixable without new data.

1. **The flagship reduction is definitional, and the identity argument proves less than it claims.**
   "Five published dispersion indices … agreeing to within 4.4 × 10⁻¹⁶" resolves to two cited indices from
   a single source — one of them *redefined* onto the shared cluster distribution because token logprobs
   were unavailable — plus one uncited statistic and two of the paper's own constructs. Worse, "functionals
   of one distribution" is not "one measurement": on the paper's own primary test H_sem responds in Qwen at
   p = .011 while S_τ — its *exact algebraic restatement* — does not (p = .213). (§2.1)
2. **"The third axis is measured by no existing index" is contradicted by the paper's own data.**
   ρ_F agrees with Cox's ρ_u at **+0.24 to +0.53** under the primary estimator (6/6 strata) and with
   Cao/Sclar's spread at **+0.18 to +0.37**; under the complete-case estimator both run **+0.41 to +0.81**.
   ρ_u was computed, promised in Methods as "a second channel on the same construct", and never reported. (§2.2)
3. **The zero-shot probe holdout contains all 150 training questions** — 12.8 % of the positive class, all
   positives. The clean numbers exist in the repo and the paper quotes the other artifact. (§2.3)
4. **The width dial's ρ_F result is an undisclosed complete-case subset (n = 22 for the only significant
   model), and the arms were not filter-identical** as Methods asserts. (§2.4)
5. **The "properly powered null" that anchors the other half of the dissociation is a shrinkage artifact.**
   Running the pipeline's own estimator on simulated data with a true Δρ_F of **+0.20** — larger than the
   whole between-model range in the study — returns +0.033 / +0.049 / +0.041, i.e. *inside the 0.04 bound
   the paper says excludes such a change*. 34–55 % of cells are pinned to a single prior value and only
   36 / 70 / 58 question pairs carry information at both levels. (§2.5)

---

## 2. Reject-level issues

### 2.1 The flagship reduction is a definition, and "one distribution" ≠ "one measurement"

> **Abstract:** "five published dispersion indices are arithmetic restatements of the latter, agreeing to
> within 4.4 × 10⁻¹⁶"
> **§4.1:** "The dispersion indices are one measurement, and this is arithmetic rather than correlation.
> Evaluated on every cell of the grid, S_τ equals H_sem / log₂|A_q| … to a maximum absolute deviation of
> 4.4 × 10⁻¹⁶"

**(a) The identity is the implementation.** `prompt_sensitivity/metrics/errica.py:40-51`:

```python
def s_tau_freeform(cluster_assignment, a_q_size):
    """S_τ(x) = H_sem(Y|X=x) / log2|A_q|  in [0, 1]."""
    h = entropy_from_assignment(cluster_assignment)
    return float(h / math.log2(a_q_size))
```

S_τ *is defined* as `H_sem / log₂|A_q|`. Confirming it equals `H_sem / log₂|A_q|` to 1.67 × 10⁻¹⁶ confirms
floating-point associativity. `FI_out^fixed ≡ log₂m₀ − H_sem` has residual exactly 0.0 — it is Eq. 9.
`Var[FI_out] ≡ Var[H_sem]` is one line of algebra on Eq. 8, since |A_q| is constant within a cell.

**(b) The substitution behind it is undisclosed.** The same file's docstring:

> "Free-form pre-cluster: we use the H_sem-derived MC variant (cluster proportions as the empirical
> distribution), **because the gateway does not expose full-vocab token logprobs needed for the original
> token-entropy formulation**."

The paper describes Errica's index correctly at §2.3 ("the normalized entropy of the *predictive
distribution*") and then computes something else. Appendix A.3 states the honest version — "The two
quantities separate only on open-ended generation, where the answer classes are not given and must be
recovered by semantic clustering" — which **contradicts §4.1**, because the entire grid *is* open-ended
generation. The identity holds there only because the re-implementation forced it.

**(c) The count is inflated.** The five named in §5.1 are normalized response entropy, the variance of
FI_out, the observed answer-space size, the variation ratio, and total-variation consistency. Var[FI_out]
and FI_out are *this paper's* constructs; |A_q| is a nuisance parameter, not an index; the variation ratio
is credited in `variation_ratio.py` to Lu et al. 2024, and **`lu2024prompts` is in `references.bib` and
cited zero times in the paper**. Only S_τ and 1−TVD carry a citation, and they are one author's paired
proposal. Three of the five were never checked numerically at all: `data/metric_reductions.md` records
|A_q|, the variation ratio and 1−TVD as *"functionals of P (same pooled clustering) — by construction"*.
The abstract's tolerance comes from three quantities and is applied to five.

**(d) The deepest problem: an algebraic identity is not a measurement identity.** "Functions of one pooled
cluster distribution" does not entail "the same measurement" — mean and variance are both functions of a
distribution and measure different things. The paper's own data show this, and the cleanest demonstration is
the exact identity itself. `S_τ ≡ H_sem / log₂|A_q|` holds to 1.7 × 10⁻¹⁶, but dividing by a *per-cell
varying* normaliser is not a monotone transform across cells, so the two are not the same statistic.
Recomputed from `data/specificity_v3_*.parquet` (n = 150 questions per model, paired L0→L1 — the paper's own
primary test):

| model | H_sem | \|A_q\| | S_τ | variation ratio | 1−TVD |
|---|---|---|---|---|---|
| Qwen | −0.124 (p = **.011**) | −1.60 (p = 8.9e-5) | −0.017 (p = **.213**) | −0.064 (p = .006) | +0.084 (p = .002) |
| Llama | −0.433 (p = 3.7e-7) | −8.12 (p = 3.6e-8) | −0.053 (p = 2.8e-4) | −0.048 (p = .042) | +0.120 (p = 2.8e-6) |
| Mistral | −0.139 (p = **.034**) | −1.76 (p = .052) | −0.029 (p = .057) | −0.018 (p = **.483**) | +0.040 (p = .163) |

Within-level Spearman with H_sem: |A_q| +0.87 to +0.94, S_τ +0.89 to +0.96, variation ratio **+0.71 to
+0.81**, 1−TVD −0.88 to −0.94. High, informative, and **not 1.0**.

So on the paper's own headline test the five give different answers: in Qwen, H_sem responds (p = .011) and
S_τ — its *exact algebraic restatement* — does not (p = .213); in Mistral, H_sem responds (p = .034) and
the variation ratio does not (p = .483). "The dispersion indices are one measurement" is therefore too
strong. What is true, and is all the paper needs, is: **they are functions of one clustering, so their
mutual agreement is not convergent evidence**. Two of them (FI_out^fixed, and S_τ up to a varying
normaliser) are affine restatements of H_sem; the rest are distinct functionals that happen to correlate.
The weaker claim is unimpeachable and costs the paper nothing.

**What the honest version looks like.** Two published indices from one source (Errica's S_τ and 1−TVD) plus
one further published statistic (Lu's variation ratio) plus two quantities introduced here are all
*functionals of a single pooled cluster distribution*; two of the five are exact affine identities with
H_sem and the rest are distinct functionals of the same object. Correlations among them are therefore not
convergent evidence. That is still a real service to the field, and it does not need 10⁻¹⁶ or the word
"five".

---

### 2.2 The "new" axis is where the paper's own data show convergence with existing indices

> **Abstract:** "The third axis is measured by no existing index"
> **§5.2:** "we supply the axis that the existing indices do not measure"

Recomputed two ways. Under the **hierarchical estimator the paper declares primary** (all 150 cells per
stratum, no complete-case selection), joining `data/rho_f_hier_union_*.parquet` to
`data/specificity_v3_*.parquet`:

| stratum | ρ_F ↔ **ρ_u (Cox)** | ρ_F ↔ **spread (Cao/Sclar)** | ρ_F ↔ H_sem | ρ_F ↔ accuracy |
|---|---|---|---|---|
| Qwen L0 / L1 | +0.455 / +0.240 | +0.213 / +0.175 | −0.011 / −0.215 | −0.163 / −0.126 |
| Llama L0 / L1 | +0.432 / +0.529 | +0.244 / +0.370 | −0.141 / −0.047 | +0.050 / +0.092 |
| Mistral L0 / L1 | +0.517 / +0.505 | +0.282 / +0.339 | −0.064 / −0.067 | +0.063 / +0.064 |

Under the complete-case MoM estimator (which is what feeds Figure 2's matrix): ρ_F ↔ spread **+0.49 to
+0.79**, ρ_F ↔ ρ_u **+0.41 to +0.81**, 6/6 strata each; in `figures/v3_metric_corr.npy` both sit at
**+0.671**, ρ_F's two largest associations with anything, against +0.081 for accuracy and +0.033 for H_sem.

So: two *published* indices track ρ_F, in every stratum, under both estimators, more strongly than anything
else in the fourteen. This is not a hostile reading — it is **convergent validity, and it is good news**.
The problem is that the paper claims the opposite, and then never reports the numbers:

- **ρ_u is promised and dropped.** §3.2.2: "We compute ρ_u alongside ρ_F as a second, embedding-based
  channel on the same construct." It appears **nowhere** in §4. A convergent-validity channel is defined,
  motivated over half a page, computed, and then withheld — while the abstract asserts the axis is empty.
- **Figure 2 shows the axis occupied.** Its "formulation sensitivity" block is `rho_F`, `rho_u (Cox)`,
  `spread (Cao)`, `ESS_in` — three of them pre-existing quantities — under a caption saying every metric
  loads on the factor of its assigned axis. The figure depicts what the text denies.
- **Incremental validity is never tested.** `spread = max(F) − min(F)` needs no ICC, no beta-binomial and
  no k = 10 budget. Before ρ_F is a contribution it must beat spread and ρ_u on the paper's own utility
  criterion (out-of-sample rephrasing payoff). That regression is a re-analysis of existing data and is not
  in the repo.

**Fix.** Change the claim to "measured by no existing index *with the within-prompt sampling term
removed*"; report the ρ_F ↔ ρ_u and ρ_F ↔ spread correlations in §4.4 as convergent validity; add the
incremental-validity test. If ρ_F does not beat `spread` out of sample, that is the most important result in
the paper and it should say so.

---

### 2.3 The zero-shot probe holdout contains every training question

> **§3.6:** "evaluated zero-shot on an annotator-labelled holdout **whose labels were never used for
> training**" · **§4.5:** "…**and were never used in training**, reach an AUROC of 0.670 to 0.678"

```bash
uv run python -c "
import pandas as pd, glob
seen=set()
for f in glob.glob('data/specificity_v3_*.parquet'):
    seen |= set(pd.read_parquet(f)['question_id'].astype(str))
h=pd.read_parquet('data/vagueness_holdout_qwen_2_5_7b.parquet',
                  columns=['question_id','ambiguous']).drop_duplicates('question_id')
ov=h[h.question_id.astype(str).isin(seen)]
print(len(seen), len(h), len(ov), int(ov.ambiguous.sum()))"
# -> 150  2002  150  150
```

Two evaluation paths exist and disagree:

| script | excludes training questions? | n | head AUROC | quoted where |
|---|---|---|---|---|
| `scripts/eval_vagueness_holdout.py:42-61` (`keep = ~meta["question_id"].isin(seen)`) | **yes** | 1,852 | 0.667 / 0.655 / 0.670 | `data/final_run_results.md` |
| `scripts/probe_eval_hardened.py:336-384` (`eval_ood`) | **no** | 2,002 | 0.678 / 0.670 / 0.678 | `data/probe_eval_hardened.md` → **the paper, Fig. 3, abstract** |

The confusion matrices confirm it arithmetically: 957+481+215+349 = **2,002**; positives 1,172 = 1,022 +
**150**. `probe_eval_hardened.py`'s own module docstring says "OUT-OF-DISTRIBUTION (1,852
annotator-labelled questions…)" — the function returns 2,002. All 150 contaminated questions are positives,
so their L0 prompt is literally a training example carrying the label "ambiguous".
`dump_vagueness_holdout.py` even documents that the overlap "is excluded at EVAL time, laptop-side"; the
hardened rewrite dropped that step.

The margin over frozen baselines survives decontamination (+0.08 to +0.10), so **the qualitative claim
stands** — but contribution 3 is "zero-shot transfer", and a reviewer who finds a contaminated number in
exactly the place the paper claims to be strictest will not extend credit anywhere else.

**Fix.** One line in `eval_ood`. Re-run, requote 0.667 / 0.655 / 0.670, regenerate Figure 3, and reconcile
the two scripts so they cannot diverge again.

---

### 2.4 The width dial's ρ_F result is an undisclosed complete-case subset, and the arms were not identical

The width dial is the positive control that completes the double dissociation — the paper's self-declared
"strongest evidence". Three problems.

**(a) n = 22.** Table 4's ρ_F rows are means over cells where MoM ρ_F is defined in *all three arms
simultaneously*. From `data/width_dial_cells.parquet`:

| model | paired n (of 100) | paired means (**the paper's table**) | unpaired means (each arm on its own covered cells) | monotone? |
|---|---|---|---|---|
| Qwen | **22** | 0.356 → 0.463 → 0.519 | 0.334 → 0.389 → 0.395 | yes |
| Llama | 54 | 0.113 → 0.135 → 0.138 | 0.098 → **0.131** → **0.114** | **no** |
| Mistral | 46 | 0.211 → 0.230 → 0.276 | 0.195 → 0.197 → 0.226 | yes |

"ρ_F increases monotonically across the three arms in all three models, significantly in Qwen (p = 0.004)"
therefore rests on **22 of 100 cells** for the only significant model, and Llama's monotonicity is produced
by the restriction. The project knows this — `scripts/make_paper_figures.py:58-61`: *"computed on the cells
covered in ALL three arms (paired), which is NOT the same as averaging each arm over its own covered set —
**the unpaired version is non-monotone for Llama**."* Table 4's caption says only "on covered cells" and
gives no n.

**This is the selection the paper condemns.** §4.4: "The method-of-moments form is undefined exactly on the
all-right and all-wrong cells, so its missingness is outcome-dependent." Requiring definedness in three arms
is that selection cubed, and it is not mild — mean accuracy in the paired-covered subset versus all cells is
**0.577 vs 0.418 (Qwen), 0.418 vs 0.297 (Llama), 0.595 vs 0.380 (Mistral)**. Coverage also differs *across
the compared arms* (Qwen 32 / 42 / 40 of 100 for narrow / production / wide), so the paired set is not a
neutral common denominator — and the narrow arm's lower coverage is itself a consequence of its smaller
universes, i.e. of problem (b) below.

*In fairness:* the caption does say "on covered cells" and the estimator switch is announced with an
identifiability argument, so neither is hidden. What is missing is the effective n — the caption header
reads "50 questions, both levels pooled", which a reader takes as 100 — and the direction of the
disagreement with the primary estimator. On the raw per-arm hierarchical fit Qwen's ρ_F *falls* across the
dial (0.677 → 0.503 → 0.516, one-sided p = 1) and Mistral's is non-monotone; the paper says it does not
report that fit and why, but not which way it goes. (Under N-matching the hierarchical fit for Qwen rises,
0.677 → 0.732, p = 2.8e-8, so the underlying result is probably fine — that number belongs in the paper.)

**(b) The gates were not identical.** §3.3.5: "All filters, thresholds, and caps are identical across arms;
only the proposal distribution changes." The NLI gate is adaptive —
`paraphrases/pipeline.py:352-361` relaxes the bidirectional threshold from 0.90 to
`nli.fallback_threshold = 0.85` whenever ten candidates cannot be met. From `data/width_dial_analysis.md`:

| arm | nli_reject | dedup_reject | **fallback_nli_frac** | mean \|U\| | personas / T |
|---|---|---|---|---|---|
| narrow | 0.07 | **0.844** | **0.66** | **6.7** | 3 / 0.5 |
| production | — (**no sidecar**, pre-R6 cache) | — | — | 10.0 | 8 / 0.8 |
| wide | 0.636 | 0.231 | 0.01 | 10.0 | 8 / 1.0 |
| swap | 0.546 | 0.297 | 0.01 | 10.0 | 8 / 0.8 |

The narrow arm ran at the *relaxed* equivalence threshold in **66 %** of universes versus **1 %** in the
wide arm, and produced universes a third smaller. The arms differ in realized gate stringency and in N, not
only in width — and "only the proposal distribution changes" is the identification assumption that licenses
reading narrow→wide as a width effect. Compounding it, the production arm has **no rejection sidecar at
all**, so the one contrast the paper calls usable ("mostly narrow to production") has one arm whose gate
behaviour is unmeasured.

The |U| gap has a second consequence the paper does not address: **every pooled-cluster metric is
|U|-dependent**, because |A_q| counts distinct clusters over the pooled responses of *all* paraphrases in a
cell — 67 responses in the narrow arm against 100 elsewhere. Measured across arms, |A_q| duly rises
2.72 → 3.38, 12.14 → 16.58, 9.55 → 11.88 from narrow to production (all p ≤ .0023), and on the N-matched
production→wide contrast it does not move at all (p = .81 / .12 / .37). So the apparent narrow→wide
dispersion response is a sample-size artifact — which is the right conclusion for the paper, but it means
the *same* artifact contaminates every narrow-arm comparison, including the headline ρ_F one. The paper
discloses that |A_q| is downward-biased in k (§3.4, Clustering); it does not disclose that it is also
downward-biased in |U|, which is the axis the width dial varies. This is the strongest argument for making
the N-matched analysis primary rather than a robustness note.

**(c) The defence against the N confound is not reproducible.** §4.3 cites "matching paraphrase counts
across arms by subsampling over five seeds". `scripts/width_dial_analysis.py:362` N-matches exactly two
columns — `sigma2_B` and `rho_f_hier` — never `rho_f_mom`, the estimator the table and the sentence use;
`load_arm_cells` takes the hard-coded default `seed = 42` (line 162) and there is no multi-seed loop
anywhere in `prompt_sensitivity/`. The quoted range (p = 0.001–0.013) appears in no committed artifact;
re-running the committed functions over `rho_f_mom` at seeds 0–4 gives p = 0.003–0.016 — close, but on
**19–21 paired cells**. Similarly, §4.3's "a simulation confirms this is a regime problem rather than a
small-sample bias" has **no committed simulation** — the claim traces only to a narrative sentence in
`RESULTS_R6_width_dial_2026-08-08.md:54`.

**Fix.** Give n per row of Table 4; print the unpaired means and state Llama's non-monotonicity; disclose
the adaptive NLI fallback and its per-arm rate in Methods; either regenerate the production sidecar or say
it is missing; commit the five-seed N-matched MoM pass and the identifiability simulation, or drop both
sentences.

---

### 2.5 The "properly powered null" is a shrinkage artifact — a true Δρ_F of +0.20 would report as +0.04

> **§4.2:** "Formulation sensitivity does not respond to the manipulation at all. … **This is a properly
> powered null, not an underpowered one: the confidence intervals exclude changes larger than 0.04 in
> magnitude in every model.**" · **§5.1:** "leaves formulation sensitivity untouched within confidence
> intervals that exclude changes larger than 0.04"

This is the load-bearing sentence of the specificity half of the double dissociation, and it does not
survive.

**Mechanism.** `fit_hierarchical_rho_f` fits **one** Beta prior on ρ across all 300 cells of a model, both
levels pooled (`stats_hygiene.py:99`), and Table 2 uses the posterior **means** (`rho_mean`), not the draws.
Degenerate cells — 54.7 % / 34.0 % / 42.7 % of the grid — carry no information and are therefore pinned to
essentially one number: Qwen's 164 degenerate cells all lie in **[0.4921, 0.4966]**, a range of 0.0045. Their
paired L1−L0 difference is ≈ 0 by construction. Only **36 / 70 / 58 of the question pairs are informative at
both levels** — which is exactly the complete-case n the hierarchical estimator was introduced to escape.
The estimator did not recover the missing information; it replaced it with a constant, and a constant
cannot move.

**Recovery check.** Simulating the paper's own design — per-cell mean success held at its observed value,
true ρ = ρ₀ at L0 and ρ₀ + Δ at L1, the full paraphrase × sample grid resimulated, and the **pipeline's own**
`fit_hierarchical_rho_f` run over all 300 cells:

| model | true Δρ_F = +0.10 → reported | true Δρ_F = +0.20 → reported |
|---|---|---|
| Qwen (ρ₀ = 0.49) | +0.015 | **+0.033** |
| Llama (ρ₀ = 0.13) | +0.044 | **+0.049** |
| Mistral (ρ₀ = 0.25) | +0.005 | **+0.041** |

Single seed, so the exact digits move a little between runs (Qwen's Δ = +0.20 cell lands between +0.033 and
+0.040 across draws, and Mistral's Δ = +0.10 estimate is noisy); the Δ = +0.20 row is the stable one and the
conclusion does not depend on the seed. Reproduce with:

```python
# run from Code/PromptSensitivityFI with `uv run python -`
import numpy as np, pandas as pd
from prompt_sensitivity.analysis.rho_f_hierarchical import fit_hierarchical_rho_f
K, rng = 10, np.random.default_rng(0)
for m, rho0 in [('qwen_2_5_7b', .49), ('llama_3_1_8b', .13), ('mistral_7b_v03', .25)]:
    d = pd.read_parquet(f'data/specificity_v3_{m}.parquet').sort_values(['question_id', 'spec_level'])
    for delta in (0.10, 0.20):
        rows = []
        for mu, n, lv in zip(d.f_graded_mean, d.f_graded_per_paraphrase.map(len), d.spec_level):
            r = np.clip(rho0 + (delta if lv == 1 else 0), 1e-4, .999); mu = np.clip(mu, 1e-3, 1 - 1e-3)
            p = rng.beta(mu * (1 / r - 1), (1 - mu) * (1 / r - 1), size=max(n, 1))
            rows.append((rng.binomial(K, p) / K).tolist())
        est = pd.DataFrame({'q': d.question_id.values, 'l': d.spec_level.values,
                            'r': fit_hierarchical_rho_f(rows, K).rho_mean})
        w = est.pivot_table(index='q', columns='l', values='r').dropna()
        print(m, f'true +{delta:.2f} -> reported {float((w[1] - w[0]).mean()):+.4f}')
```

A true effect of **+0.20** — larger than the entire between-model range of ρ_F in this study
(0.09 to 0.44) — would be reported by this estimator as +0.033 to +0.049, i.e. **at or inside the very bound
the paper says excludes it**. Back-solving the attenuation, the 0.04 posterior-mean bound corresponds to an
**estimand-scale bound of roughly 0.10 (Llama) to 0.22 (Qwen)** — comparable to or larger than the width
dial's own effect, which is the paper's positive control. The interval is a statement about the shrunken
statistic, not about the estimand. The observed null (−0.002 / +0.013 / +0.013) is therefore consistent with
no effect *and* with a large one, and "properly powered" cannot be claimed.

One further detail compounds it. **41 / 22 / 37 of the 150 paired deltas are exactly zero** — the
signature of degenerate cells pinned to the same prior value at both levels — and SciPy's Wilcoxon default
(`zero_method='wilcox'`) discards them. The quoted p-values therefore rest on **109 / 128 / 113** pairs, not
the "full n = 150 questions per model" the text claims.

**This does not necessarily overturn the conclusion** — ρ_F plausibly is flat under the specificity dial,
and it is the theoretically expected result. What is overturned is the *evidential status*: the paper
currently sells this null as positive evidence for discriminant validity, and it is not.

**Fix (analysis only, no new data).** Three options, in order of preference: (i) run the same recovery
simulation and report the estimator's minimum detectable Δρ_F — a good number to have regardless;
(ii) report the paired test on the multiple-imputation **draws** rather than the posterior means (the module
docstring already warns that "correlating shrunken point estimates understates uncertainty"); (iii) report
the complete-case MoM paired test on the 36 / 70 / 58 informative pairs alongside, with its own CI, and let
the reader see both. Until one of those exists, the sentence must read "a null that we cannot distinguish
from an underpowered one at this design size".

---

## 3. Major issues

### 3.1 The headline effect is a null for 60 % of the questions, and the moderator is not reported

`data/seed_audit.md` §L3(iii) — computed, documented, absent from the paper:

| model | Δ_union, pinned reading = first-listed (n = 60) | pinned reading = later (n = 90) | gap |
|---|---|---|---|
| Qwen | **+0.175** | **−0.011** | +0.186 |
| Llama | +0.175 | +0.092 | +0.083 |
| Mistral | +0.184 | +0.075 | +0.109 |

Spearman(Δ_union, pinned index) = −0.213 (p = .01) / −0.158 (p = .05) / −0.148 (p = .07). For Qwen the
headline +0.064 is carried entirely by the 40 % of questions where a SHA-256 hash happened to pin AmbigQA's
first-listed (canonical) reading. The seed audit's own verdict: *"The heterogeneity by reading rank is a
substantive result and should be reported rather than averaged away."*

This is scientifically more interesting than the marginal mean — it says disambiguating *toward the
canonical reading* helps and disambiguating toward an obscure one does not — and it bounds the practical
advice the paper gives. The paper is right that the *estimand* is seed-stable (±0.003–0.007 against a CI
half-width of ±0.06); that is a different question from homogeneity, and only the former is addressed. Note
also that the moderator is specific to the **primary union endpoint** — it is flat and partly reversed under
target gold, which is itself informative about what the union effect is picking up.

### 3.2 The dataset filter conditions on the L1 treatment condition; the stated defence is a non-sequitur

> **§3.3.2:** "we applied a dataset-side filter that keeps only questions whose **target** answer appears
> verbatim in the snippet bundle; 52 % of the ambiguous questions pass. The filter **uses no model output
> and therefore introduces no selection on model knowledge**."

`specificity/build_levels.py:184-197`: `target_in_evidence()` resolves `choose_target_idx(...)` and returns
`any(a.lower() in bundle for a in q.interpretations[idx].answers)` — it tests **only the pinned target's**
answers, and the pinned target is exactly what defines level 1. The evidence bundle is level-invariant
(guardrail 2), so the retained 52 % is precisely the set on which L1's gold is guaranteed present in the
prompt, while the other readings — which is what L0 needs under union gold — carry no such guarantee. The
conclusion ("no selection on model knowledge") answers a different objection than the one that matters:
this is selection on the *contrast*, not on knowledge.

Related and undiscussed: the evidence snippets are *the annotators' own search results for the original
question*, so there is no reason to expect them to cover all readings equally. That is the single most
plausible alternative explanation for §3.1's reading-rank moderator — the mechanism would be
evidence-reading alignment rather than specificity — and the data to check it (per-reading answer presence
in the bundle) already exist. This deserves a paragraph.

Funnel, for the record (`data/run_manifest.json`): 2,002 → 1,172 ambiguous → **609 pass evidence
coverage** → **first 150 in dataset order**, no stratification. The analysed sample is 25 % of the eligible
pool. Disclosed, but the paper should characterise the 459 discarded eligible questions.

### 3.3 Inference discipline is applied asymmetrically (3.3–3.3e)

The next five items are one theme: the paper's statistical discipline is real, well documented, and applied
much more strictly to the claims it wants to defend than to the claims it wants to assert.

**3.3 — A double standard on accepting nulls.** The paper is admirably strict about null claims where it is
defending a bound, and not strict at all where it is asserting one.

- **Strict (good):** §4.1 refuses "orthogonal" because "the sample supports equivalence bounds only of
  |ρ| < 0.33 to 0.47"; §4.2 defends the ρ_F null by showing the CI "excludes changes larger than 0.04".
- **Not strict:** §4.3 concludes "competence does not respond to width … and neither does dispersion" from
  **Friedman p = 0.68 / 0.53 / 0.54** and **0.47 / 0.57 / 0.62** on n = 100, with no equivalence bound. A
  non-significant Friedman is not evidence of absence, and the arm's own preregistration (P3/P4) asked for
  "the delta with CI". The largest arm-mean gap (≤ 0.02) is quoted for competence but no bound at all is
  given for dispersion, although the directed narrow→wide H_sem change is −0.053 / −0.048 / −0.076 bits.
- **Worse, the test choice is asymmetric within the same table.** The axis that is *supposed* to move is
  tested with a directional one-sided Wilcoxon; the axes that are supposed to stay flat are tested with an
  omnibus Friedman. Apply the same directional test to dispersion and Qwen's H_sem falls from narrow to wide
  at **one-sided p = 0.034** — a *smaller* p than σ²_B's 0.054 in the same model, which the paper reports as
  a positive result — on n = 100 and surviving N-matching. So "widening the paraphrase generator … leaves
  competence and dispersion flat" is not supported in the model that carries the ρ_F result; it is an
  omnibus test masking a directional movement. Use one test for the whole table.
- **Not strict:** §4.1 concludes POSIX "loads on dispersion and on formulation sensitivity about equally,
  and the difference is not significant in any model (p = 0.42 / 0.63 / 0.054)". n = 42 / 62 / 62
  (`metric_reductions.md`) and is not stated. Failing to reject a difference of dependent correlations at
  n = 42 is no information. The artifact's own wording — "POSIX is not axis-diagnostic **at this sample
  size**" — is the correct one, and the qualifier is dropped in the paper.

**Also:** even taken at face value, the 0.04 bound is 9 % of Qwen's ρ_F level and **44 % of Llama's**. State
it relative to each model's own level. (§2.5 shows the bound cannot be taken at face value at all.)

### 3.3b "All inference follows the declared family" is false, and the width arm has no error control

§4 opens: "all inference follows the declared family of Section 3.4." The declared family is twelve tests
and covers **only the specificity arm**. Reported in Results but outside it: three Steiger tests for POSIX;
36 equivalence bounds; one sign test; 897 per-cell islands permutation tests; six σ²_B level tests; six
one-sided width tests plus six Friedman plus the manipulation check plus six swap Spearmans plus the
N-matched subsample tests; 48 probe permutation tests; six predictive-validity correlations. Applying the
paper's own BH rule *within the width arm's six primary tests* leaves **one** significant — which, to be
fair, is what §4.3 already says in words ("significantly in Qwen"), so this formalises the Results rather
than overturning them. The fix is not to put everything in one family — it is to say which claims carry
error control and which are descriptive, and the paper currently implies the former for all of them.

### 3.3c The width arm's estimator substitution was a deviation from preregistration, and it changes the answer

`scripts/width_dial_analysis.py` preregisters, in its own docstring: "**P2 rho_F (hierarchical) increases
with width — one-sided narrow < wide.**" The paper reports P2b (MoM on covered cells) instead. To be clear,
**the substitution itself is disclosed**: Table 4's caption names the MoM estimator and gives the
identifiability reason, and §4.3 carries two explicit estimator notes. What is *not* disclosed is that the
replaced estimator was the preregistered one, and that **the substitution changes which model supports the
prediction**:

| | Qwen | Llama | Mistral |
|---|---|---|---|
| **P2, preregistered (hierarchical)** | p = 1 (decrease) | p = 1.5e-8 | p = 6.2e-7 |
| **P2b, reported (MoM, covered)** | **p = 0.0035** | p = 0.079 | p = 0.051 |

The paper's narrative — "the dial moves ρ_F most in the model with the most phrasing sensitivity to amplify
(Qwen) and least in the model with the least (Llama)" — is available only under the substituted estimator.
Under the preregistered one the ordering is exactly reversed. A deviation from preregistration that flips
the result needs to be labelled as one, with both rows shown. `FORKING_PATHS.md` compounds this: it lists
twelve forks, does not contain this one, and its fork 2 asserts that "all headline orderings hold under
both" estimators and that "the width-dial double dissociation" is decision-robust — both contradicted by
the R6 artifact.

### 3.3d N-matching is reported where it helps and omitted where it hurts

The narrow arm's universes are 33 % smaller, so N-matching is the key robustness check. §4.3 reports it for
ρ_F, where it holds. `data/width_dial_analysis.md` also contains it for σ²_B, where it does not:

| model | σ²_B narrow<wide, unmatched | **N-matched** |
|---|---|---|
| Qwen | 0.054 | 0.022 (better) |
| Llama | 0.021 | **0.051** (loses significance) |
| Mistral | 0.060 | **0.214** (loses it decisively) |

σ²_B is the endpoint the paper elevates as "not deflated by a model's decoding entropy and therefore the
quantity on which models are compared". Its width response is the one that does not survive the confound
control, and that row is not in the paper. `RESULTS_R6_width_dial_2026-08-08.md` also notes the single-seed
N-matched p-values "wobble across seeds (e.g. llama σ²_B .020–.569)". *In fairness:* the narrow < wide
**direction** still holds 3/3 for σ²_B after matching — what is lost is individual significance in two
models. Report the matched row with five-seed ranges and say exactly that.

### 3.3e The one consistent cross-axis association is complete-case, reported one sentence after "primary estimator"

> **§4.1:** "Under the primary estimator, no cross-axis association exceeds |ρ| = 0.14 … One association is
> consistently non-zero: ρ_F correlates positively with the dispersion factor **in all six model-by-level
> strata under both gold sets** (sign test p = .031; disattenuated 0.15 to 0.42)."

The 6/6 pattern and the sign test are the **complete-case** column. Under the hierarchical estimator named
one sentence earlier, `independence_union.md` gives −0.004, −0.067, +0.048, +0.111, +0.080, +0.126 → **4/6
positive**; `independence_target.md` gives 5/6. The equivalence bounds quoted in the same sentence
(0.33–0.47) are likewise complete-case Fisher intervals (`independence_analysis.py:246,261`) while the
|0.14| is hierarchical — and the association is quoted *disattenuated* while the bound is quoted
*attenuated*, which is not like-for-like. Three estimators and two attenuation conventions in two sentences.
The bound error at least runs **in the paper's favour** — the hierarchical CIs support |ρ| < 0.34, tighter
than the quoted 0.47 — so this is a labelling fix, not a retraction.

The underlying point — ρ_F is modestly associated with dispersion and the paper should not say orthogonal —
is correct and worth keeping. It just needs one estimator per sentence.

### 3.4 Neither "out-of-sample" validity check leaves the original paraphrase set

> **§4.4 / §5.1:** "ρ_F estimated from the first ten samples predicts the payoff of rephrasing … on the
> disjoint second half of the k = 20 arm … Whether rephrasing a question will pay off is therefore
> decidable, at the population level, before trying."

Both checks resample *decoding draws* over the **same ten paraphrases**: the greedy check is the T = 0 pass
over the identical universe; the payoff check is samples 10–19 of the k = 20 arm over the identical
universe. What is demonstrated is that the between-paraphrase signal is stable across decoding draws —
which is what ρ_F is defined to isolate, and is close to a split-half reliability. That the payoff
correlations (+0.61 / +0.39 / +0.70) sit in the same band as the split-half reliabilities
(0.38 / 0.52 / 0.57) is consistent with exactly that reading.

A practitioner reads "whether rephrasing will pay off" as *paraphrases you have not tried*. The design
never tests that, and it could: hold out five paraphrases, estimate ρ_F on the other five, predict payoff on
the held-out five. Pure re-analysis.

Additionally the target `F_max − F̄` is the statistic `FORKING_PATHS.md` #10 retired as uninterpretable
("its expectation rises mechanically with k … without a per-k null it is uninterpretable"). Predicting a
statistic you declared uninterpretable needs a sentence of justification.

### 3.5 Gold-set discipline breaks down in Table 5 and in the primary family

Recomputed from `data/rho_f_hier_{union,target}_*.parquet` (n = 300 cells each):

| quantity | **union gold** (the declared primary) | **target gold** | printed in the paper |
|---|---|---|---|
| ρ_F hierarchical | 0.4439 / 0.0913 / 0.2395 | 0.4872 / 0.1305 / 0.2521 | both, correctly labelled ✓ |
| σ²_B | 0.0374 / 0.0179 / 0.0283 | **0.0307 / 0.0143 / 0.0226** | target, **unlabelled** |
| MoM coverage | **59.7 / 82.7 / 77.7 %** | **45.3 / 66.0 / 57.3 %** | target, **unlabelled** |
| Spearman(MoM, hier) | 0.903 / 0.945 / 0.848 | 0.867 / 0.893 / 0.856 | "**0.64 to 0.72**" — neither |

Three consequences:
1. Under the gold set the paper itself declares primary, MoM covers **60–83 %** of cells, not "half to two
   thirds" / "45 to 66 percent" — and that coverage deficit is the paper's stated justification for
   introducing the hierarchical estimator at all. The argument is still fine; the number is the wrong one.
2. "agrees with the method-of-moments form where both exist (Spearman **0.64 to 0.72**)" does not reproduce
   from any committed artifact. The true agreement is **0.85 to 0.95** — the error runs *against* the
   paper's interest. The string traces only to `RESULTS_R1_R2_R3_2026-08-07.md:94`, a superseded internal note.
3. `scripts/stats_hygiene.py:95-106` fits the hierarchical ρ_F on `specificity_v3_{m}.parquet` (target gold)
   and reads `union_gold_{m}.parquet` only for the accuracy endpoint. So the ρ_F row of
   Table 2 — inside a family the paper introduces as union-gold — **is target gold**, and §4.2's "the same
   null replicates under target gold" is backwards. (The null does replicate under union: −0.0131 /
   +0.0126 / +0.0134, p = .54 / .43 / .41. Report that as the primary row.)

The mechanical-coupling null is likewise target-gold only (`mechanical_coupling_null.py:140-141`), which
should be stated where it is used.

### 3.6 One correlation counted four times, in a paper about counting one measurement five times

> **§4.1:** "with all 24 observed correlations inside the null band" · **§5.3:** "reproduces all
> twenty-four observed complete-case associations"

`data/mechanical_null.parquet` has 24 rows = 3 models × 2 levels × **4 assumed true-ρ values**;
`r_observed_cc` takes **6 distinct values**, each compared against four bands. `data/mechanical_null.md`
says it correctly: *"**6/6** observed complete-case correlations sit inside the mechanical null band."* Given
the paper's central contribution, this is an unfortunate place to multiply evidence by four. Write
**"6/6, robust across four assumed values of the true ρ"** — which is a *stronger* and shorter sentence.

### 3.7 Table 5's predictive-validity rows use the estimator the same section disowns, and have no script

| row | value | issue |
|---|---|---|
| Agreement between gold sets | +0.53 / +0.69 / +0.68 | traces to `RESULTS_R1_R2_R3:43`; **no script in the repo** |
| Predicts greedy-pass disagreement | +0.51 / +0.40 / +0.54 | `RHO_F_CONSTRUCT_VALIDITY:36`; **no script**; no n |
| Predicts rephrasing payoff | +0.61 / +0.39 / +0.70 | `RHO_F_CONSTRUCT_VALIDITY:44-46`; **no script**; **n = 42 / 62 / 62**, complete-case MoM |

`grep -rl payoff prompt_sensitivity/` returns only `make_ksri_deck.py` (a slide string). The two
predictive-validity rows — the strongest utility evidence in the paper — are complete-case MoM correlations
on 42–62 cells, computed with the estimator §4.4 declares unusable ("its missingness is outcome-dependent…
We therefore report the hierarchical estimator throughout"). Either re-run them on the hierarchical
estimates or state the inconsistency. (That sentence is inaccurate in a second way: Table 4's ρ_F rows are
MoM. Soften to "except in the per-arm width analysis, where the fit is not identifiable on the narrow arm".)

### 3.8 FI_out^fixed is negative in up to 42 % of cells — a surviving fraction greater than one

> **§3.2.3:** "Equation 9 is not clamped, because a model that disperses over more meanings than the
> question admits is informative in itself."

Recomputed from `data/specificity_v3_*.parquet`:

| model | FI_out^fixed < 0 | at L0 | minimum | mean \|A_q\| observed | mean log₂m₀ |
|---|---|---|---|---|---|
| Qwen | 2.3 % | 2.7 % | −1.21 | 3.6 | 1.43 |
| **Llama** | **31.7 %** | **42.0 %** | −2.07 | 15.8 | 1.43 |
| Mistral | 23.3 % | 26.7 % | −2.16 | 12.3 | 1.43 |

The cause is structural: `2^H_sem` counts **model answer clusters** (including wrong answers, 3.6–15.8 of
them) while `m₀` counts **annotator readings of the question** (mean 2.7). Their ratio is not a surviving
fraction and its negative log is not a functional-information value — Hazen's Eq. 1 is non-negative by
construction. "Informative in itself" converts a formal breakdown into an interpretation, and 42 % is the
modal regime for the model carrying the largest H_sem effect. Table 1 compounds it by listing this row's
possibility space as "semantic answer space" when Eq. 9 uses the interpretation count.

The paper effectively already concedes the point by excluding FI_out^fixed from the inference family and
calling it a unit conversion. Say the same thing in §3.2.3 — **log₂m₀ is a reference constant for unit
conversion, not the possibility space of the answer distribution** — and report the negative rate.

### 3.9 FI_spec's bits do no work, and the dose test the paper calls impossible is available and null

> **§3.2.4:** "FI_spec is the third instance of Equation 3" · **§5.3:** "A specificity ladder with more than
> two rungs would identify a dose-response curve in bits, **which a two-point design cannot**."

`metrics/fi_spec.py:10-11` makes the degeneracy explicit: L0 → `m_valid = m₀` → 0 bits; L1 → `m_valid = 1` →
log₂m₀ bits. FI_spec ≡ level × log₂m₀ — a dataset constant times a binary indicator. It appears in no §4
number. (The paper does state the construction openly at §3.3.1; the issue is the claim that it is a
*measurement*.)

But the dose test **is** available *between* questions: at L1, FI_spec ranges over 1.0–3.32 bits, and
`data/seed_audit.md` already contains the regression — Spearman(Δ_union, m₀) = **−0.105 (p = .20) /
−0.126 (p = .12) / −0.049 (p = .55)**. Removing more bits of ambiguity does **not** buy more competence; the
association is null and, if anything, wrong-signed. That is the minimal validity requirement for stating a
manipulation in bits, it is a legitimate and interesting negative result, and the paper currently claims the
test is impossible while its own artifact contains it.

### 3.10 "Exactly three axes" — the disconfirming re-run is computed and not disclosed

> **§4.1 conclusion:** "*the measurement framework has exactly three distinct axes*"

`data/factor_audit.md` Q2: with one representative per identity class (7 variables), **Horn retains 2**, not
3, and "the clean structure dissolves (accuracy splits +0.72/−0.52 across two factors)". The paper hedges the
*mechanism* correctly — "Because seven of the fourteen inputs are provable functions of one clustering, we
treat this as consistent with a three-axis reading rather than as established by it" — and then never prints
the number it computed that settles its own hedge, concluding "exactly three" anyway.
`SUPERVISOR_FEEDBACK_2026-08-11.md` states the required framing: *"the factor analysis proves the
ASSIGNMENT, never the COUNT … Do NOT say 'PCA+Horn proves exactly 3'."* That instruction was not carried
into the text.

(Q3 — Horn retains 4 once `log₂m₀` is added — is *not* a fair counter-number, since log₂m₀ carries the
manipulated variable rather than a fourth outcome metric. Mention it, don't lean on it.)

### 3.11 The two halves of the double dissociation are not commensurable

| | specificity arm | width arm |
|---|---|---|
| sample | 150 questions | **positions 0–49** of the same 150 (verified; described only as "a 50-question subsample") |
| ρ_F estimator | hierarchical, all cells | **method of moments**, covered-in-all-arms (n = 22/54/46) |
| test | two-sided Wilcoxon | **one-sided** Wilcoxon |
| multiplicity | declared 12-test family, Holm + BH | none declared |
| dispersion representative | H_sem | H_sem (but \|A_q\| moves at p = 6e-9, §2.1d) |

A double dissociation is an argument that *one instrument* responds to one manipulation and not the other.
When the instrument, the sample, the cell set, the test and the multiplicity regime all change between the
halves, the two nulls and the two effects are not directly comparable. The claim is still probably true —
but it should be stated with the asymmetry visible, and ideally the specificity null should be recomputed on
the same 50 questions with the same MoM estimator, which costs nothing.

Relatedly, the abstract, Discussion and Conclusion state the width result flatly ("raises the phrasing
share", "rises in all three models") where §4.3 correctly hedges it to "significantly in Qwen". The
direction is 3/3 in point estimates, so this is a hedging gap rather than an error — but the qualifier
belongs in the abstract.

### 3.12 The probe section: four smaller breaks, one of which is a Methods misstatement

- **The headline head's null is not what Methods says it is.** §3.6: "Permuted-label controls are full
  refitted permutation null distributions at every probed layer." `probe_eval_hardened.py:57-62` sets
  `"vagueness": (True, False)` — `perm_ok = False` — so the vagueness head uses `flip_control`, which by its
  own docstring "rescore[s] the SAME oof" and refits nothing. The parquets confirm `null_kind == "flip"` for
  every vagueness row, and `"(null at selected layer only)"` for every logistic row in every head. The flip
  control is arguably the *correct* null for a within-question label — the sentence just needs to say so.
- **The in-distribution 0.873 is a within-question L0-vs-L1 contrast and the paper never says so.**
  `feedback/heads.py:82`: `out["vagueness"] = (out.spec_level == 0).astype(float)`. The label *is* the
  experimental manipulation: positives and negatives are the ten L0 and ten L1 paraphrases of the same
  question, sharing an identical evidence block. Folds are grouped by `question_id`, so this is a
  presentation gap and not leakage — but a reader takes "separates underspecified from well-specified
  questions at 0.873" as a population detection rate, and it is not one. One sentence fixes it.
- **The one non-null fragility number is not the statistic every other head is quoted from.** The paper
  gives Llama 0.63 (p = 0.038); that is the fixed-layer 5-fold OOF at the selected layer. The **nested-CV
  value is 0.580**, and vagueness (0.874/0.873/0.873) and dispersion (0.671/0.758/0.765) are both quoted
  from `nested_cv`. Its p = 0.038 is also the 1/26 floor of a 25-draw null, refit with `n_splits=3` against a
  statistic fit with `n_splits=5`. Quoting 0.580 makes the paper's own negative result cleaner.
- **The reliability head has no baseline at all.** `probe_eval_hardened.py:279-289` gates the four text
  baselines behind `if binary:`, and reliability is `(False, True)`. So §3.6's "Text-surface baselines …
  are evaluated under protocols matched to the heads" is false for one of the four heads. Its pooled
  0.35–0.51 also masks a condition asymmetry: within the underspecified condition — the one a user is
  actually in — it reaches only ≈0.23–0.35.
- **The base rate is never stated.** The holdout prevalence is 0.59 (`probe_eval_hardened.md`, "PR-AUC 0.725
  (0.59)"). §4.5 reports "precision of about 0.66" while flagging 67–73 % of questions; without the base
  rate, that reads as a working detector rather than a ~7-point lift over flagging everything. Results is
  otherwise honest here; the Contributions paragraph ("a warning before the model answers") omits the
  operating point entirely. Note also that 0.65 is a hard-coded literal (`heads.py:236`) that no script
  selects or validates — "shipped decision threshold" overstates it.

### 3.13 The Reproducibility statement is false in three checkable ways

> "The repository contains … **the analysis scripts that produce every table and figure in this paper, and
> the per-cell result files from which they are computed.**"

```bash
git ls-files data | wc -l     # -> 10, none of them a parquet
grep -n parquet .gitignore    # -> data/*.parquet ; data/**/*.parquet
```

1. **No per-cell result files are in the repository.** All parquets are gitignored by rule.
2. **None of the artifacts the Results section quotes are committed.** `data/stats_hygiene.md` (the declared
   source of truth), `metric_reductions.md`, `width_dial_analysis.md`, `probe_eval_hardened.md`,
   `independence_*.md`, `mechanical_null.md`, `factor_audit.md`, `seed_audit.md`, `axis1_graded_curve.md`,
   `run_manifest.json` and `FORKING_PATHS.md` are all **untracked**.
3. **Not every figure comes from a script.** `scripts/make_paper_figures.py` hard-codes Figures 1 and 3 as
   Python literals (`SPEC`, `WIDTH_RHOF`, `PROBE_IN`, `PROBE_OOD`) despite a docstring claiming "Every
   number is either read from a committed artifact". Only Figure 2 loads a file.

And a dangling promise: §3.4 says "Analysis decisions … are disclosed in the **forking-paths appendix**".
There is none. `appendix.tex` has Derivations, Estimator Conventions, and an empty "Additional Material"
section containing only a `%% TODO` comment — which renders as an empty Section C on page 30 of the
compiled PDF. `FORKING_PATHS.md` exists, is excellent, and is in neither the paper nor the repo.

---

## 4. Minor issues and presentation

- **Length and structure.** Main text runs to page 22 (Intro p.2, Background p.3, **Methods pp.5–14**,
  Results pp.14–19, Discussion p.19, Conclusion p.21); references pp.23–28; appendix pp.29–30. The ICLR
  template's limit is 9 pages. *This is not a venue violation* — the title page self-identifies as a KIT
  seminar submission and `\iclrfinalcopy` deliberately disables anonymous mode. It is a writing problem:
  nine pages of Methods before the first result pushes the load-bearing hedges (the estimator switch,
  POSIX's non-assignability, the factor-count caveat) far from the claims they qualify, which is precisely
  how the overstatements in §2 survived into the abstract. §3.1–3.2 also restate Hazen three times.
- **Three `%% TODO-LIT` comments remain** (lines 783, 790, 914), each marking a literature claim printed
  **with no citation** — including "the closest published comparison finds LLM paraphrases *more* lexically
  and syntactically diverse than crowdworker paraphrases". Anyone following the GitHub link sees them.
- **The L0 gold-preservation filter is an OR where Methods says AND.** §3.3.3: "at level 0 it is the union
  over all interpretations, **because a faithful rephrasing of an ambiguous question must preserve every
  reading**." `paraphrases/constraint_filter.py:339-340`: "a candidate PASSES if it preserves **ANY** of the
  accepted answers (logical OR over `gold_answers`)". The OR is defensible (single-gold rejected 100 % of
  valid L0 paraphrases), but the stated *reason* describes an AND. Since nine of the ten L0 prompts per cell
  are generated, FI_spec = 0 bits is verified for the original text and asserted for the rest; the
  bidirectional NLI gate carries that load and should be named as doing so.
- **`fi_in.py` compares against the grid without a tolerance.** Line 41 (`sum(1 for s in scores if s >= k)`)
  and line 98 use bare `>=` against `np.linspace(0, 1, 21)`, whose elements include 0.30000000000000004,
  0.6000000000000001 and 0.7000000000000001. Graded F values are exact multiples of 1/10, so a paraphrase
  with F = 0.3, 0.6 or 0.7 fails a threshold it exactly attains at 3 of 21 grid points.
  `sensitivity_v2.py:75` and `axis1_graded_curve.py:69` both use an epsilon; only the module feeding the
  FI_in curve and AUFI does not. Numerically negligible, but the codebase is internally inconsistent about
  the definition in Eq. 4.
- **Uncited-but-relevant entries in the paper's own `references.bib`:** `lu2024prompts` (source of the
  variation ratio *and* of a published sensitivity-vs-accuracy correlation), `kim2025detail` (prompt
  specificity as an IV — bears on gap 3), `sorensen2022information` (information-theoretic prompt
  quantification — bears on the FI framing), `kobalczyk2025active`, `lin2024generating`, `liu2026understanding`,
  `yang2026shared`. Fifteen entries are active and uncited. The bib's own comments prescribe fixes that were
  not applied (e.g. on `nikitin2024kernel`: "position our reduction as collapse DOWNWARD").
- **The three gap claims are stated as unqualified universals** and each is qualified by something already
  in the paper or the bib: gap 3 ("specificity is not manipulated as an independent variable") is
  contradicted 27 lines earlier by the paper's own description of `keluskar2024llms`; "no formal reduction
  has been published" is qualified by `nikitin2024kernel`, cited two pages later for a weaker point.
- **The 14 metrics are never enumerated** in the paper. Given that the retained factor count depends on the
  composition of that set, it must be inspectable. `ESS_in` appears in Figure 2 and nowhere in the text, and
  its own module docstring warns it "mostly tracks context length, not paraphrase diversity".
- **"13.8 % of cells" (S_τ degeneracy)** is the unweighted mean of 26.3 / 6.7 / 8.3 %. Report the range;
  Qwen — the model carrying the width-dial result — is at 26 %.
- **The split-half reliability (0.38 / 0.52 / 0.57) is target-gold, method-of-moments, and conditioned on
  cells informative in both halves** — three mismatches with the hierarchical union-gold quantity it is used
  to bound. Under the primary union gold the same statistic is **0.363 / 0.509 / 0.495**
  (`independence_union.md`), so Mistral is 0.50, not 0.57. §4.3's "at the ceiling set by the instrument's own
  split-half reliability" also does not hold for Llama, where the swap agreement (+0.27) sits well below its
  reliability (0.52).
- **The probe logistic permutation nulls use 25 draws** (`probe_eval_hardened.py`, `n_perm_logistic=25`), so
  **p = 0.038 = 1/26 is the smallest attainable value** — Llama's fragility p is that floor rather than a
  measured value, and the accepted nulls (0.42, 0.50) rest on 25 draws. The mass-mean nulls use 200
  (floor 0.005). The permutation counts appear nowhere in the paper; disclose them. (This does not undermine
  the fragility null itself — the observed scores sit at the null mean, not just short of significance.)
- **Estimator mixing in one sentence.** §4.1's "no cross-axis association exceeds |ρ| = 0.14" is the
  *hierarchical* estimate; "the sample supports equivalence bounds only of |ρ| < 0.33 to 0.47" is the
  *complete-case* bound. Both true; presenting them as one comparison is not.
- **"disattenuated 0.15 to 0.42 … under both gold sets"** is the `independence_target.md` range. Under the
  primary union gold it is **0.08 to 0.50** — and 0.50 exceeds the upper equivalence bound quoted in the
  same paragraph.
- **Table 4's caption** says "Both quantities increase with width in every model". σ²_B is **non-monotone in
  Qwen (0.0256 → 0.0258) and Mistral (0.0210 → 0.0205 → 0.0249)**; only narrow < wide is tested.
- **Figure 2 plots |Spearman|**, hiding that FI_out^fixed correlates −1 with H_sem, which makes the
  dispersion block look more coherent than the signed matrix would. Note the absolute value in the caption.
- **The length control is never mentioned.** L1 questions are 43 % longer (46.4 → 66.5 chars, verified), and
  "is the effect just longer prompts?" is the first alternative any reviewer raises. The project has an
  excellent answer sitting in `final_run_results.md` — within-cell ρ(length, F) = −0.046 / −0.021 / +0.038
  over a range exceeding the manipulation in 95 % of cells, and ρ(Δlength, Δaccuracy) ≈ 0 — and the paper
  uses none of it. This is free credibility being left on the table.
- **Nine cited works are dated 2026** and print as bare arXiv preprints. The project has verified them;
  the bib should carry venues where they exist.

---

## 5. What survives scrutiny — stop re-defending these

Recomputed and correct. Several of these were attacked during this review and held.

- The **ρ_F estimator** matches Eqs. 5–7 exactly (`metrics/sensitivity_v2.py:56-72`), and the binary-outcome
  SS_within identity in Appendix A.1 is exact.
- **AUFI ≡ accuracy**: exactly −1.0000 binary, and −0.9985 to −0.9997 graded across all six strata. Appendix
  A.2's "the graded track softens the identity but does not remove it" is literally true. *Suggestion, not
  correction:* print the graded number — it strengthens the reduction and justifies demoting AUFI.
- **ΔFI's axis assignment is data-confirmed**, not a construct decision: within-stratum Spearman with
  competence averages +0.53 against +0.30 with ρ_F, and it loads +0.74 / +0.27. (It is still never reported
  as a number — a page-cost issue, §4.)
- **The design totals reconcile exactly**: each `specificity_v3_*.parquet` is (300, 42) → 900 cells, 89,730
  scored responses under a disclosed ten-paraphrase cap. The "299" in `axis1_graded_curve.md` is that
  analysis's own subset.
- **The hierarchical estimator** recovers known ρ in simulation (0.041 / 0.194 / 0.513 for true
  0.05 / 0.2 / 0.5), agrees with MoM at 0.85–0.95, and is deterministic across refits to six decimals. The
  flat-μ-prior trap was found, fixed and regression-tested.
- **The width dial's underlying result is probably not an estimator artifact.** The raw per-arm
  hierarchical fit is non-monotone, but the paper announces that it does not report that fit and gives an
  identifiability argument; under N-matching the hierarchical fit for Qwen *rises*, 0.677 → 0.732
  (p = 2.8e-8). What is missing is disclosure of the direction, not the switch (§2.4).
- **The model ordering Qwen > Mistral > Llama** holds under both gold sets, both estimators, both
  generators, and on σ²_B (which is not deflated by decoding entropy). The most robust result in the paper.
- **The ρ_F null under the specificity dial is the theoretically right prediction**, and framing it as
  discriminant validity rather than failure is correct. (What does not survive is the claim that it is
  *adequately powered* — see §2.5. The prediction is good; the evidence for it needs rebuilding.)
- **The union-gold reframing** and the lottery decomposition are exactly right, and putting the
  decomposition in Methods rather than Results is the correct editorial call.
- **The mechanical-coupling null** (Gulliford) is the right instrument, correctly interpreted.
- **POSIX's non-assignability is reported prominently and honestly** in §4.1, with the exact p-values and
  the Steiger citation — this was checked as a possible burial and it is not one.
- **The swap arm** is a real contribution: per-cell ρ_F agreement +0.41 / +0.27 / +0.51 at the reliability
  ceiling, σ²_B +0.64–0.80, ordering preserved. "Levels are generator-relative, structure is
  generator-robust" is earned.
- **The fragility-head null** is reported honestly and prominently in both Results and Discussion.
- **The FI priority claim** ("first use of functional information to evaluate language models") is hedged,
  and the project's own verified literature sweep supports it. The withdrawn Szostak/Hazen claim stayed
  withdrawn, and the Hazen letter-sequence precedent is cited in Methods. This was attacked and it held.
- **Related-work engineering** is unusually good: the G-theory lineage, the Corona finite-U reframing, the
  Cox ρ_u name-collision paragraph, the Kossen/Zhang/Razavi probe positioning, the Nikitin "one
  instantiation" caveat.
- **The LaTeX compiles clean**: no warnings, no overfull boxes, no undefined references, 61 citations all
  resolved.

---

## 6. What the honest contribution is

After every deduction, what remains is publishable, and it is not what the abstract says.

1. **A reduction result about measurement practice.** Under a common semantic-clustering instantiation, the
   dispersion quantities in circulation are functionals of one pooled cluster distribution, two of them
   exact affine identities with H_sem. Reported correlations among them are therefore not convergent
   evidence. This is a real service. It is analytic, it does not need functional information to state, and
   it should be a proposition with a scope ("the dispersion indices we computed"), not an agreement to 10⁻¹⁶.
2. **A per-question, gold-referenced, noise-corrected sensitivity share with honest measurement
   properties.** ρ_F itself is standard G-theory; what is new is the estimand plus a hierarchical estimator
   defined on every cell, a reliability figure, a mechanical-coupling null, and a generator-swap ablation
   that appears to be first in this literature. Its agreement with Cox's ρ_u and Cao's spread should be
   presented as convergent validity, and the incremental-validity test against them run.
3. **A double dissociation between a question-side and a generator-side dial** — currently the weakest of
   the three, because *both* halves need rebuilding: the width half rests on a 22–54-cell outcome-selected
   subset under a substituted estimator, and the specificity half rests on a null whose bound is a shrinkage
   artifact. The design is right and the prediction is right; the evidence is thinner than the paper says.
   If it holds after A5 and A6, it is the strongest construct argument available and better than any
   correlation.
4. **A negative result worth more than it is given**: formulation sensitivity is *not* linearly readable
   from a single prompt representation, with proper nulls. It is currently buried under a probe
   contribution that largely replicates Kossen and Zhang.

What is *not* a contribution, and should stop being framed as one: functional information as the ruler that
generated the framework. FI generated exactly the two axes that turned out to be redundant with quantities
the field already had, and did not generate the one novel axis — which §3.1 concedes is "not an instance" of
Eq. 3. The Contributions paragraph nonetheless asserts the universal ("Every quantity in the model is −log₂
of a surviving fraction") and reaches three possibility spaces only by substituting the *manipulated
variable* for axis 2. Table 1 gets this right in its body and its note; the Contributions and Conclusion do
not. The defensible version is that FI supplies a common *reporting unit* that makes the redundancy
legible — a paragraph, not a framing.

---

## 7. Recommendations, in order

| # | action | cost | fixes |
|---|---|---|---|
| **A1** | Re-run `eval_ood` excluding the 150 training questions; requote 0.667/0.655/0.670; regenerate Fig. 3 | 1 line + one run | §2.3 |
| **A2** | Table 4: n per row, unpaired means, Llama's non-monotonicity; disclose the adaptive NLI fallback (0.90→0.85, 66 % vs 1 %) and the missing production sidecar; commit the 5-seed N-matched MoM pass and the identifiability simulation or drop those sentences | half a day | §2.4 |
| **A3** | Restate the reduction as a scoped proposition; fix "five published indices"; disclose the S_τ substitution; drop 4.4e-16 from the abstract; reconcile §4.1 with §4.3 on \|A_q\| | writing + one table | §2.1 |
| **A4** | Report ρ_F ↔ ρ_u and ρ_F ↔ spread as convergent validity; run the incremental-validity test against `spread`; soften "measured by no existing index" | one afternoon | §2.2 |
| **A5** | Run the Δρ_F recovery simulation and report the estimator's minimum detectable effect; test on the MI draws or report the complete-case MoM test alongside; withdraw "properly powered null" until one of these exists | one afternoon | §2.5 |
| **A6** | Show both P2 rows (preregistered hierarchical and reported MoM) and state the deviation; add the N-matched σ²_B row | writing + one table | §3.3c, §3.3d |
| **B1** | Report the reading-rank moderator as a result; check per-reading answer presence in the evidence bundle as the alternative explanation | half a day | §3.1, §3.2 |
| **B2** | Label every gold set in Table 5; fix coverage to the union values; fix "0.64 to 0.72" → 0.85–0.95; make the ρ_F endpoint union-gold and fix the inverted "replicates under target gold" | half a day | §3.5 |
| **B3** | Add equivalence bounds (or "no information") to every claimed null: width Friedman, POSIX Williams; state POSIX's n; use one test direction across the whole width table; say which claims carry error control and which are descriptive | writing + small analysis | §3.3, §3.3b |
| **B3b** | Fix the "6/6 under both gold sets" sentence: one estimator per sentence, and either 4/6 (hierarchical) or "complete-case, 6/6" | one paragraph | §3.3e |
| **B4** | Add the held-out-paraphrase payoff prediction (5 vs 5) so the utility claim leaves the original universe | one afternoon | §3.4 |
| **B5** | "24" → "6/6, robust across four assumed true ρ" | one line | §3.6 |
| **B6** | Commit the scripts behind Table 5's three unscripted rows; re-run the two predictive rows on hierarchical ρ_F; add n | one day | §3.7 |
| **B7** | Report FI_out^fixed's negative rate; restate log₂m₀ as a unit-conversion constant; fix Table 1's possibility-space label | writing only | §3.8 |
| **B8** | Run and report the between-question FI_spec dose regression (null — say so) and delete "which a two-point design cannot" | one hour | §3.9 |
| **B9** | Print factor_audit Q2 (Horn = 2 de-duplicated); adopt "assignment, not count" | writing only | §3.10 |
| **B10** | Recompute the specificity null on the width arm's 50 questions with MoM so the dissociation halves are commensurable; move the width qualifier into the abstract | one afternoon | §3.11 |
| **B11** | Probes: say the null is a flip control; say the label is `spec_level`; quote fragility from `nested_cv` (0.580); note the missing reliability baseline; state the 0.59 base rate | writing only | §3.12 |
| **C1** | `git add -f` the twelve artifacts; put `FORKING_PATHS.md` in the appendix; make `make_paper_figures.py` read artifacts instead of literals; populate or delete Appendix C | half a day | §3.13 |
| **C2** | Resolve the three `TODO-LIT`s; cite `lu2024prompts`, `kim2025detail`, `sorensen2022information`; narrow the three gap claims | one day | §4 |
| **C3** | Add the length control (it is already computed and it is convincing) | one hour | §4 |
| **C4** | Add an epsilon in `fi_in.py`; fix the OR/AND wording in §3.3.3 | one hour | §4 |
| **C5** | Cut toward 9 pages — Methods is nine pages by itself | two days | §4 |

**If only four things get done:** A1, A2, A3, A4. A1 and A2 are where a checkable number does not survive
its own protocol; A3 fixes the abstract's most falsifiable sentence; A4 turns the paper's most exposed claim
into supporting evidence.

---

## 8. Score card

| criterion | score | note |
|---|---|---|
| Originality | 4/10 | the estimand and the swap ablation are new; the ruler is a relabelling and the probes replicate Kossen/Zhang |
| Technical quality | 5/10 | inference machinery better documented than most published work here; but both halves of the central dissociation and the probe headline have protocol breaks |
| Clarity | 7/10 | very well written; 2.4× over budget, and Methods buries Results |
| Significance | 5/10 | the reduction matters to the field; the new axis is not yet shown to beat a one-line statistic |
| Reproducibility | 3/10 | statement false as written; no data and no source-of-truth artifacts committed |
| **Overall** | **4/10** | reject as framed, clear path to accept — the same verdict as 2026-08-06 for entirely different reasons, which is itself progress |

**Strongest case for reject.** *The paper's headline is that published sensitivity indices are one
measurement, but the reduction is verified against implementations the authors redefined onto a shared
clustering, two of the "five published indices" are the authors' own, and the premise fails on the paper's
own primary test — H_sem responds in Qwen at p = .011 while S_τ, its exact algebraic restatement, does not.
Its second claim, a new axis no index measures, is contradicted by its own correlation matrix, where two
published indices track the new one more closely than anything else in the study and one of them was
computed and never reported. Its third contribution, a zero-shot probe, is evaluated on a holdout containing
every training question. The double dissociation that is supposed to establish the axes as separate
properties fails on both sides: the width half rests on 22 of 100 cells selected by the same
outcome-dependent criterion the paper elsewhere calls uninterpretable, under an estimator substituted for
the preregistered one in a way that reverses which model supports the prediction, in arms whose equivalence
gate fired at 0.85 in 66 % of one arm and 1 % of another despite Methods asserting they were identical; and
the specificity half's "properly powered null" is a shrinkage artifact — the authors' own estimator returns
+0.03 to +0.05 when the true effect is +0.20, inside the bound they claim excludes it. And the framing
device, functional information, is conceded by the paper itself not to apply to the one axis it claims as
new.*

**Strongest case for accept.** *This is the most methodologically self-aware measurement paper in the
prompt-sensitivity literature. It shows — algebraically, not by correlation — that a family of dispersion
indices is one object, which invalidates a class of convergence arguments the field has been making. It
supplies the first noise-corrected, per-question, gold-referenced sensitivity share and, unusually, reports
its reliability, its coverage, a mechanical-coupling null that disqualifies its own most tempting
correlation, and a preregistered double dissociation with a positive control. It runs the first
paraphraser-swap ablation in this literature and finds levels generator-relative while structure is not. It
reports a clean, properly-nulled negative result about its own probe, and it withdrew a novelty claim
against itself. The errors are disclosure and framing errors on top of an unusually honest empirical core,
and every one is fixable without new data.*

---

## 9. Checked and cleared — do not re-litigate

These were raised during this review and did **not** survive verification. Recorded so the same ground is
not covered again.

| claim | verdict |
|---|---|
| "The graded track softens the identity" is empirically false | **Refuted.** Binary is exactly −1.0000; graded is −0.9985 to −0.9997. The hedge is literally true. |
| ΔFI is really an axis-2 quantity mis-assigned to axis 1 | **Refuted.** Competence +0.53 vs ρ_F +0.30 within stratum; loading +0.74 vs +0.27. |
| The width dial reverses under the primary hierarchical estimator | **Refuted.** The raw per-arm fit is non-identifiable and is not reported; N-matched Qwen rises 0.677 → 0.732 (p = 2.8e-8). |
| Design totals (897 vs 900 cells) do not reconcile | **Refuted.** Each parquet is (300, 42); 900 exactly. |
| The paper violates ICLR anonymity / page limits as a venue | **Refuted.** It self-identifies as a KIT seminar paper and disables anonymous mode deliberately. The length critique survives as a writing critique only. |
| The "first use of functional information to evaluate language models" claim is the withdrawn one in disguise | **Refuted.** Hedged, and supported by the project's own verified sweep; the Hazen precedent is cited in Methods. |
| POSIX's non-assignability is buried | **Refuted.** Reported prominently in §4.1 with p-values and the Steiger citation. |
| The probe's in-distribution result leaks across folds | **Refuted.** Folds are grouped by `question_id`; both levels of a test question are held out together. The label-disclosure gap (§3.12) is real; the leakage is not. |
| The fragility head's layer was chosen post hoc | **Refuted.** Selected inside nested CV. What survives is that the quoted 0.63 is not the nested-CV statistic every other head is quoted from. |
| The width arm's estimator substitution was hidden | **Refuted.** Table 4's caption states it with the identifiability reason. What survives (§3.3c) is that the replaced estimator was the *preregistered* one and that `FORKING_PATHS.md` fork 2 asserts the opposite. |
| The width arm's dispersion response is as large as the specificity effect | **Refuted.** On the directed narrow→wide contrast H_sem moves −0.053 / −0.048 / −0.076 bits against a specificity effect of −0.124 / −0.433 / −0.139. The *reporting* asymmetry (§3.3) survives; the magnitude claim does not. |
| The width arm's conclusions collapse under BH | **Refuted as a consequence.** BH within the arm leaves one model significant — which is what §4.3 already says in words. |
| N-matching kills two of three width effects | **Refuted.** The narrow < wide direction holds 3/3 for σ²_B after matching; only individual significance is lost (§3.3d). |
| The complete-case equivalence bound inflates the paper's claim | **Refuted in direction.** The hierarchical CIs support \|ρ\| < 0.34, *tighter* than the quoted 0.47 — a labelling error that runs against the paper's own interest. |
| The 25-draw permutation floor undermines the fragility null | **Refuted.** The observed scores sit at the null mean, not just short of significance. The undisclosed draw counts (§4) are still worth fixing. |
| factor_audit Q3 (Horn = 4 with log₂m₀) disproves "three axes" | **Refuted as evidence.** log₂m₀ carries the manipulated variable, not a fourth outcome axis. Q2 (Horn = 2) is the real counter-number. |

---

*Every number attributed to a recomputation in this document was produced first-hand from the committed
artifacts or the implementation on 2026-08-14. Where this review and `three_axes_framing` or the older
result documents disagree, prefer this review.*
