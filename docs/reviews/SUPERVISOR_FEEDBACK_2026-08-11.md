# Supervisor feedback after the KSRI talk — three points, answered with new computations

Artifacts produced: `data/factor_audit.md`, `data/seed_audit.md`,
`data/probe_redundancy_audit.md` (scripts: `scripts/factor_audit.py`,
`scripts/seed_audit.py`, `scripts/probe_redundancy_audit.py`). Full test suite passes.

---

## 1 · "Check the PCA + Horn. Which metrics went in, and why should the factors be the three you specified?"

### What went in

14 metrics, computed per cell, correlated **within** each of the 6 model × level
strata (Spearman, pairwise-complete), then averaged element-wise over the strata so
the manipulation is never smuggled into the matrix:

| a-priori axis | metrics fed in |
|---|---|
| competence (3) | accuracy, AUFI (graded), FI premium |
| formulation sensitivity (4) | ρ_F, ρ_u (Cox), spread (Cao), ESS_in |
| output dispersion (7) | H_sem, S_τ, FI_out_fixed, Var[FI_out], TVD-sens, variation ratio, \|A_q\| |

POSIX is **not** in the matrix (computed on a subset only); its non-discrimination
comes from the separate Williams test. FI_spec is **not** in it either — see §3.

### What came out — the assignment is perfect

Varimax rotation of the first three components. F1 = dispersion, F2 = competence,
F3 = sensitivity, and **14 of 14 metrics load dominantly on the factor of the axis
they were assigned to a priori**. Each representative owns its factor:

- H_sem **.98** on F1 · accuracy **.93** on F2 · ρ_F **.90** on F3
- convergent evidence: ρ_u **.82** and spread **.64** land on F3 with ρ_F, exactly as the
  collapse story predicts; S_τ .92, \|A_q\| .90, TVD .91 land on F1 with H_sem.

That is the direct answer to "why should it be the ones we specified": we did not read
three factors off the data and name them afterwards — the three axes were declared
first, and every one of the 14 metrics independently sorted itself into the right box.

### Horn is not a close call

Eigenvalues 5.95 / 2.88 / **1.62** / 0.94 …; Horn's 95th-percentile thresholds
1.72 / 1.53 / **1.40** / 1.31. Factor 3 clears its threshold by 0.22; factor 4 misses
by 0.37. Re-run across 10 simulation seeds × 4 assumed sample sizes (n = 80/136/150/299):
**retains 3 every single time**. So the count is not seed-fragile.

### The honest caveat — and it changes how we should phrase the contribution

I ran the check the hostile version of this question implies: **de-duplicate the input
set** (one representative per identity class + every genuinely separate measurement =
7 variables) and re-run. Result: **Horn retains 2, not 3**, and the clean structure
dissolves (accuracy splits +.72/−.52 across two factors).

Why: after removing provable duplicates, the dispersion axis has exactly **one**
indicator left (H_sem). Factor analysis cannot identify a factor from a single
indicator — you need ≥3 — so the de-duplicated run is under-determined *by
construction*. It is not evidence against three axes; but it does prove that the
**number 3 is partly a function of how many aliases we fed in**, because we fed in
seven dispersion aliases and three competence ones.

**Recommended restatement** (and this is a strengthening, not a retreat):

> The factor analysis establishes the **assignment**, not the count: every published
> index sorts onto the axis we claim for it, and no fourth dimension appears. The
> **count** rests on the two pieces of evidence that do not depend on the input
> composition: the algebraic reductions (five indices are exact functions of one
> quantity — proof, not correlation) and the experimental dissociation (each axis
> moves under its own dial and only its own).

Say it in that order and the "you got three because you put in three groups"
objection is answered before it is raised. The current phrasing ("PCA + Horn retains
exactly 3 factors") invites the objection instead.

---

## 2 · "Replicate the measurements for different seeds — is that necessary? Can we do it?"

Partly necessary, mostly already done, and one layer is worth a cheap cluster job.
There are **four** seed layers with very different costs:

| layer | what it changes | cost | status |
|---|---|---|---|
| **L1 analysis** — bootstrap, Horn sims, split-half draws, CV folds, subsample matching | nothing about the data | free | **done now, stable** |
| **L2 sampling** — which k = 10 generations per prompt | the measured success rates | cache-only re-score, **no new generation** | **recommended, cheap** |
| **L3 target reading** — which annotator reading is pinned at L1 | the L1 prompt *and* the target gold | full re-generation of the L1 half | **quantified instead** |
| **L4 generator** — who writes the paraphrases | the whole paraphrase universe | done | **already done (R6 swap arm)** |

### L1 — varied, and it moves nothing (`data/seed_audit.md`)

- Bootstrap seed (5 seeds): effect estimates are means and carry **no seed at all**;
  only CI endpoints move, max drift **0.002–0.008**.
- Split-half draw seed (5 × 200 splits): reliability .38/.52/.57 → spread **≤ 0.02**.
- Hierarchical ρ_F estimator: deterministic (grid + Nelder–Mead from a fixed start),
  identical to 6 decimals across repeat fits — **no seed enters the point estimates**.
- Horn: retains 3 across 10 seeds (above).

### L3 — the interesting one, now quantified rather than assumed

The target seed picks which of a question's m₀ readings becomes L1. I checked whether
that choice matters, and **it does**:

| model | Δ_union when index 0 was pinned | when a later index was pinned | gap |
|---|---|---|---|
| Qwen2.5-7B | **+0.175** (n = 60) | **−0.011** (n = 90) | +0.186 |
| Llama-3.1-8B | +0.175 | +0.092 | +0.083 |
| Mistral-7B | +0.184 | +0.075 | +0.109 |

AmbigQA lists readings in canonical order, so this says: **disambiguating to the
obvious reading buys a lot; disambiguating to an obscure one buys nearly nothing**
(for Qwen, nothing at all). That is a substantive moderator we should report, not a
defect — but it does mean the layer is not innocuous.

Does it threaten the headline? No, and here is the number. A new seed redraws each
question's reading uniformly, so what varies is the *share* of questions landing on
index 0: expected .399, SD across seeds **.039** (150 near-independent Bernoulli
draws). Propagating that through the gaps above:

| model | headline Δ_union | implied SD across target seeds | 95 % CI half-width |
|---|---|---|---|
| Qwen2.5-7B | +0.064 | **±0.007** | ±0.064 |
| Llama-3.1-8B | +0.125 | **±0.003** | ±0.060 |
| Mistral-7B | +0.119 | **±0.004** | ±0.063 |

**The seed-induced spread is ~10× smaller than the sampling CI.** A target-seed
replication would land well inside the interval we already report — so it is a
legitimate robustness item, but not one that can overturn a result.

### What I'd actually run

**L2, as a cache-only cluster job.** The k = 20 arm already contains a *disjoint
second half* of samples (fresh seeds, samples 10–19) for the covered cells; re-scoring
those from the cache reproduces every headline on an independent sample with **zero new
generation**. That is the strongest seed replication available per unit of compute.
The local cache holds only the June smoke run (45 MB), so it needs the cluster cache.

---

## 3 · "Why 4 probe heads if 3 axes suffice? Where does FI_spec sit in the PCA?"

### Was FI_spec in the PCA? No — and it structurally cannot be

Correlations are computed **within** a specificity level. At L0, FI_spec = log₂(m₀/m₀)
= **0 bits for every question** — zero variance, correlation undefined. So it never
entered the matrix, and no version of the analysis could have included it.

Its question-level carrier **log₂(m₀)** (number of annotator-listed readings; identical
at both levels, equal to FI_spec at L1) *is* defined throughout, so I added that:

- direct correlations with the three representatives: **−.19** with accuracy, **−.09**
  with ρ_F, **+.13** with H_sem — all weak;
- in the 15-variable factor solution its largest loading is **−.50** (competence), i.e.
  it does not sit inside any axis;
- and **Horn then retains 4 factors, not 3** — adding it adds a dimension rather than
  collapsing into one.

So the premise "it probably falls into one of the three" is not what the data shows.

### Is the fourth head redundant? Three tests say no (`data/probe_redundancy_audit.md`)

All four heads retrained on identical features (hidden state, last prompt token, layer
0.5), identical question-grouped folds:

| test | result |
|---|---|
| **Direction.** \|cosine\| between the underspecification weight vector and each axis head | **.19–.29** (competence), .006–.08 (dispersion), .04–.19 (sensitivity) — nearly orthogonal |
| **Reconstruction.** 3 axis heads' predictions → underspecification | **.63 / .69 / .66** vs **.86 / .87 / .87** direct |
| **Construct-level** (capacity-matched: the *true measured* axis values, not probe estimates) | **.64 / .70 / .65** — same answer, so it is not an artifact of the direct head having more parameters |

**Mean cost of dropping the head: −0.21 AUROC.** The three axes simply do not contain
the information; underspecification is an *input* property, the axes are *outcome*
properties.

### Where I agree with him, and what I changed

He is **conceptually right and the slide was wrong**: listing "4 variables in the linear
heads" implies a four-dimensional measurement model, which contradicts our own
contribution. FI_spec is **the dial, not an axis**.

He is **empirically wrong that it is therefore redundant**: it is the only head that
works well (.87 vs the axis heads' .77 / .51 / chance), the only one with external
labels, and the only deployable artifact (C3).

So I did **not** retrain a 3-head model — dropping the head would delete contribution
C3 to fix a labelling problem. The fix is one line of framing:

> ~~"4 variables in the linear heads: competence, formulation sensitivity, output
> dispersion, underspecification (FI_spec)"~~
>
> **"Three axis heads — can the measurement model's outcomes be anticipated? (dispersion
> partly, competence weakly, formulation sensitivity not at all) — plus the prompt
> checker, which reads the input property FI_spec and is the one that transfers."**

One honesty note to keep with it: the in-domain .87 is inflated by L1 prompts simply
being longer (length alone scores .756 in-domain). The defensible number remains the
zero-shot holdout: **.67–.68 head vs .55–.59 frozen text baselines**, where length
collapses to .545.

---

## Recommended edits to the deck (not applied — your `final_copy` is untouched)

1. **Result 1 slide**: change "No fourth axis: PCA + Horn retains exactly 3 factors" →
   "PCA + Horn: every index sorts onto its axis, no fourth dimension appears" and let
   the identities + dials carry the count. (§1)
2. **Prompt-checker slide**: replace the "4 Variables" list with "3 axis heads + the
   prompt checker (input property)". (§3)
3. **Backup, new**: the reading-rank moderator table from §2 — it is a genuine finding
   and pre-empts the target-seed question.
4. **Limitations**: add one line that a sampling-seed replication (L2) is available
   cache-only and is the next robustness run.
