# The AmbigQA pipeline, end to end — and the proposed final-run changes

**Audience:** someone who does not know this project. **Date:** 2026-08-02.
Every number and example below comes from the committed data of the completed
"v3" run (149 questions × 2 levels × 3 models) or from this repo's code.

---

## 0. What the project measures, in one paragraph

When you ask a language model a question, three separate things can go wrong
or right: the model may simply not know the answer (**ability**), it may know
it but only answer correctly for *some phrasings* of the same question
(**formulation sensitivity**), and its answers may scatter over many meanings
rather than committing to one (**output dispersion**). This project measures
all three as separate, bit-valued axes — using one ruler, *functional
information* (FI): **bits = −log₂(the fraction of possibilities that still
work)**, borrowed from molecular biology (Szostak 2003; Hazen et al. 2007) and
licensed for non-biological systems by Wong et al. (2023). The experiment
moves a fourth, dataset-side dial — **how specific the question itself is** —
and watches the three axes respond.

---

## 1. The data: AmbigQA

[AmbigQA](https://arxiv.org/abs/2004.10645) (Min et al., EMNLP 2020) takes
real Google queries and has human annotators mark the ambiguous ones, listing
every distinct **interpretation** with its own **disambiguated question** and
**answer**. Roughly half of real queries turn out ambiguous.

Running example (a real row of our data):

> **Ambiguous question Q:** "Who won the mayor race in St. Petersburg,
> Florida?" — the annotators found **m₀ = 3** valid interpretations (different
> election years), each with its own answer.
> **One interpretation Q_i:** "Who won the **2017** mayor race in St.
> Petersburg, Florida?" → answer: Rick Kriseman.

Everything the pipeline treats as ground truth — the interpretations, the
disambiguated question texts, the answers, m₀ — is **annotator-written**. The
dataset also ships, per question, the annotators' own Wikipedia search-result
snippets (`used_queries`), which we use as evidence (step 3).

---

## 2. Step 1 — build the specificity levels (the dial)

File: `prompt_sensitivity/specificity/build_levels.py`

For each question we build one cell per **specificity level**:

| level | question text | m_valid (readings the wording admits) | FI_spec = log₂(m₀/m_valid) |
|---|---|---|---|
| L0 | the ambiguous Q (annotated) | m₀ = 3 | 0 bits |
| L1 | the target interpretation's Q_i (annotated) | 1 | log₂ 3 = **1.58 bits** |

FI_spec is the dial: *how many bits of interpretation-narrowing the question
text itself carries*. It is *model-free* — identical for every model by
construction — which is what makes it a valid manipulated variable (an x-axis,
not an outcome).

### 2.1 What happens to the multiple interpretations? (one is pinned)

Every question in the study has several annotated interpretations — that is
the inclusion criterion. Exactly **one** of them is pinned as the
**measurement target**, chosen deterministically (sha256 of question-id +
seed: no cherry-picking, and every job, model, and rerun picks the same one).
The full real example:

> **Q (L0):** "Who won the mayor race in St. Petersburg, Florida?" (m₀ = 3)
> - [0] "…the **2017** mayor race…" → Kriseman / Rick Kriseman ← **TARGET**
> - [1] "…the **2013** mayor race…" → Kriseman / Rick Kriseman
> - [2] "…the **2009** mayor race…" → Foster / Bill Foster

Scoring at BOTH levels credits only the **target's** answer variants
(semantic NLI match against {"Kriseman", "Rick Kriseman"}); the OR in
"multi-gold scoring" ranges over the target's *surface variants*, never over
interpretations:

| model's answer at L0 | a valid reading? | scores |
|---|---|---|
| "Rick Kriseman" | yes (2017 — and also 2013) | **1** |
| "Bill Foster" | yes (2009) | **0** |
| "Ken Welch" (2021, not annotated) | no | 0 |

So an L0 answer can be a perfectly *valid reading* and still score 0 — by
design. Crediting every interpretation would (a) let the gold set change
between levels (union at L0, one answer at L1), reintroducing exactly the
ground-truth drift the guardrail exists to prevent, and (b) measure a
different construct — "can the model answer *some* reading" cannot respond to
disambiguation, whereas "does added wording steer the model to the *intended*
reading" is the thing FI_spec manipulates. This is why L0 accuracy sits near
the pick-the-right-reading baseline (~1/m₀) and rises when L1 pins the
reading — e.g. the Mussolini cell (person-vs-party): at L0 the model
consistently answered a *different valid* interpretation (F = 0), at L1 F = 1.

The non-target interpretations are **not discarded** — they do three jobs:
they set m₀ (the dial's denominator, here log₂3 = 1.58 bits); their answer
**union** is the validity constraint for L0 *paraphrases* (a faithful
rephrasing of an ambiguous question must preserve *any* reading's answer —
judging L0 paraphrases against only the target rejected 100% of valid ones, a
bug found and fixed 2026-07); and the model's drift across readings at L0 is
exactly what axis 3 (H_sem) registers as dispersion.

One honest quirk this example exposes: interpretations can **collide** on the
same answer (Kriseman won 2013 *and* 2017). At L0, "Kriseman" then scores 1
even if the model meant 2013 — fixed-gold scoring credits the surface answer,
not the intended reading. Collisions inflate L0 accuracy, which *shrinks* the
measured L0→L1 gain — i.e. the bias is conservative, against our own
hypothesis, never for it.

**The two guardrails** (the design's load-bearing walls):

1. **Fixed gold.** The scoring answer is the *target interpretation's* answer
   at **both** levels. Only the question text changes. Without this,
   disambiguation would move the target and any accuracy gain would be a
   grading artifact. (The target interpretation is chosen deterministically
   via a seeded hash, so every job picks the same one.)
2. **Identical evidence.** Both levels (and every paraphrase) see the *same*
   evidence block, so specificity is the only manipulated variable.

## 3. Step 2 — uniform evidence

Closed-book turned out to be a wall: 84% of questions hit a knowledge floor
(the 7B models simply don't know). So every cell gets the question's own
annotator search snippets as context — identical across levels and
paraphrases (guardrail 2). A dataset-side filter keeps only questions whose
target answer appears verbatim in the snippets (52% pass) — model-free, so it
introduces no selection-on-model-knowledge bias. Answerability now comes from
*reading*, not *recall*, which matches how AmbigQA was built (open-book).

## 4. Step 3 — paraphrase universes (for the sensitivity axis)

File: `prompt_sensitivity/paraphrases/pipeline.py`

For each (question, level), a separate 14B generator model (Phi-4 — never used
as an eval model, to avoid a self-grading confound) writes ~10
meaning-preserving rephrasings, filtered twice:

- **NLI equivalence**: bidirectional entailment with the original — meaning
  (and thus specificity) stays constant *within* a level;
- **gold-preservation judge**: a paraphrase must still admit the right
  answer(s). At L0 "right" means *any* interpretation's answer (an ambiguous
  question's answer set is the union); at L1 only the target's.

Real L0 paraphrases from the example: "St. Petersburg, Florida mayor race
winner?", "Who emerged as the victor in the mayoral election in St.
Petersburg, Florida?" — same meaning, different surface form.

These generated texts are **stimuli**, not data: nothing generated is ever
scored as ground truth.

## 5. Step 4 — evaluation cells

File: `prompt_sensitivity/scripts/run_specificity.py` (checkpointed per cell;
resumes across 30-minute cluster windows)

One **cell** = (question, level, model). For each of the N=10 paraphrases the
eval model answers k=10 times at temperature; every answer is scored against
the **fixed gold** with NLI-based semantic matching (never exact string match
— Hua et al. 2025 show exact match inflates "sensitivity" with grading
artifacts). This yields the per-paraphrase **graded function**
F(x) = P(correct | phrased as x) ∈ [0,1] — 3 models × 149 questions × 2
levels × 10 paraphrases × 10 samples.

## 6. Step 5 — the three axes + the dial, computed per cell

| axis | headline metric | formula / reading |
|---|---|---|
| 1 · Ability | **FI_in(k) curve** + accuracy | FI_in(q,k) = −log₂(share of paraphrases with F ≥ k): "how much phrasing luck does quality k take?" Its scalar integral AUFI turned out to be accuracy in a log wrapper (ρ = −1.00) → appendix; the curve stays. |
| 2 · Formulation sensitivity | **ρ_F** (functional ICC) | share of success variance caused by *which phrasing you picked* vs decoding noise. ⊥ accuracy (.08), ⊥ dispersion (.03); agrees with an independent embedding-based channel (ρ_u) at .67; stable k=10↔k=20 (.81–.95). |
| 3 · Output dispersion | **H_sem** (+ FI_out_fixed) | semantic entropy of NLI-clustered answers; FI_out_fixed = log₂(m₀) − H_sem = how much of the supplied precision the model *realizes*. Nearly every published sensitivity index is this family in costume (S_τ .94, TVD .91, POSIX .60–.63…). |
| the dial | **FI_spec** | log₂(m₀/m_valid), dataset-side (step 2). |

Worked example, real numbers (qwen):

| | L0 (ambiguous) | L1 (2017-specific) |
|---|---|---|
| FI_spec | 0 | 1.58 bits |
| graded accuracy | 0.98 | 0.92 |
| ρ_F | 0.00 | 0.21 |
| H_sem | 0.09 | 0.29 |

(This particular question is *easy with evidence* at both levels — the
population-level result over 149 questions is what carries the finding:
accuracy roughly doubles, FI_in drops ~0.8 bits, H_sem falls, all three
models, p ≤ 5e-9.) The second flavor of question also matters: "Who is the
administrator of the Small Business Administration?" scores F=0 at both
levels — a **floor** cell, where ρ_F is *undefined* (no variance to
decompose); coverage is always reported rather than hidden.

## 7. Step 6 — hidden states, probes, and the feedback model

- `dump_hidden_states.py` re-runs every prompt for **one forward pass** and
  stores the last-prompt-token hidden state at 4 depths (before any
  generation — the "TBG" state; cache-only, bit-identical prompts).
- `train_fi_probes.py` fits **linear** heads on those states with
  **GroupKFold by question** — paraphrases of one question never straddle
  train/test, so every reported number is performance on **unseen questions**
  — plus permuted-label and length-only controls.
- The user-facing **feedback model** (`feedback/heads.py`, demoed in the
  Streamlit app) asks, from a single forward pass of *your* prompt: is it
  vague (AUROC **.85–.87** vs length-baseline .755)? will answers scatter
  (.69–.78)? how likely is a correct answer (calibrated, ECE .04–.08)? is it
  phrasing-fragile (≈ chance — honestly labeled experimental)?

What does **not** exist yet: any evaluation on a *different dataset* — that
is the main gap the final run addresses.

---

## 8. Proposed changes (the final run) — status per item

### 8.1 C1 — a middle rung for the dial *(design decision pending)*

Today the dial has two points per question (0 and log₂ m₀). The proposal adds
a **partially disambiguated** L_mid for questions with m₀ ≥ 3 (613 exist), so
FI_spec takes a value strictly between — a *within-question dose-response*.
Three candidate mechanisms, differing in what gets constructed:

| option | L_mid text comes from | admitted-set label quality | caveat |
|---|---|---|---|
| **(1) across-question dose** — no L_mid at all | — (uses existing v3 data: L1's FI_spec already spans 1.0–3.32 bits across questions) | annotated | dose varies *between* questions (m₀ may correlate with difficulty; checkable) |
| **(2) mechanical disjunction** | template over **annotator** Q_i texts ("…in 2001, or in 2003?") | exact **by construction** | register: explicitly enumerated ambiguity ≠ vague wording |
| **(3) LLM rewrite** *(built, incl. judge gates + human-review file)* | Phi-4, gated | LLM-judged + human eyeball — **categorically weaker than annotation** | reviewer surface: a machine-written question text on the x-axis |

Current recommendation on record: (1) always (free), (2) if a within-question
mid-level is wanted, (3) shelved.

### 8.2 C6 — evidence dial *(approved, built)*

Rerun 50 questions at evidence fractions 0.0 and 0.5 (f = 1.0 is the existing
v3 data). Gives the second axis of a (question-specificity × evidence-amount)
surface, with the same guardrails — the trimmed evidence stays identical
across levels and paraphrases.

### 8.3 C3 — POSIX for all three models *(approved, built)*

POSIX (Chatterjee et al. 2024) needs token log-probabilities and is priced at
~13 s/cell on cached generations, so it runs as a **comparison subset** (the
same 50 questions as the existing qwen arm, now also llama + mistral) — enough
for the cross-model correlation row ("POSIX belongs to the dispersion
family"), not an everywhere-metric.

### 8.4 Cross-dataset check: CondAmbigQA *(mode decision pending)*

[CondAmbigQA-2K](https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA-2K)
(EMNLP 2025) was inspected by loading it: 2,000 questions; per question a list
of `properties = {condition, groundtruth, citations}` and 20 retrieved
Wikipedia passages. 1,451 questions have ≥ 2 conditions (the analogue of
interpretations). **Two structural differences from AmbigQA**, found by
checking the columns: the gold answers are **long-form** (median 36 words —
our short-span NLI scoring doesn't transfer as-is), and the `condition` is a
~35-word context paragraph, so its "disambiguation" means **appending
context**, not rewording the question. Hence two wiring modes:

- **(a) safe** — no scoring against their gold: frozen **vagueness-head
  transfer** (embed `question` vs `question + condition`; one forward pass
  each) + the gold-free dispersion metrics (H_sem, S_τ, TVD). Tests the
  deliverable on a truly unseen dataset with zero adaptations.
- **(b) full** — replication with disclosed deviations: judge-based scoring
  against the long gold + an NLI leak-gate on each condition.

Also free and already possible: the **830 NQ questions AmbigQA annotators
marked *non-ambiguous*** form a held-out test-set for the vagueness head
(labels from human judgment, questions never seen in training).

### 8.5 Paper analyses *(done, committed)*

Three independence arguments for "you need all three axes", already computed:
**constructive counterexamples** (accuracy pinned at 0.5 while ρ_F goes 0→1;
accuracy pinned while H_sem doubles; H_sem pinned while accuracy goes 1→0),
**factor analysis** (the 14-metric correlation matrix yields exactly three
factors — dispersion family / ability / ρ_F+ρ_u — explaining 70% of variance),
and **octant occupancy** (median-splitting the three axes: all 8 high/low
combinations are populated in every model × level — no axis predicts
another). Plus bootstrap CIs for ρ_F (506 cells, median 95% width 0.18).

### 8.6 Launch mechanics *(ready)*

`bash cluster/submit_final_run.sh {ml-smoke|ml-prep|ml-eval|ml-dump|posix|dial}`
— chained 30-minute GPU windows, per-cell resume, every artifact pulled by
`run.sh pull`. The ML phases include a **human hard gate**: a review file of
every generated mid-level rewrite must be read and approved before full
compute runs (only relevant if option 3 of §8.1 is ever chosen).

---

## 9. Open decisions

1. **§8.1** — mid-level mechanism: (1) across-question only, (2) mechanical
   disjunction, or (3) LLM rewrite?
2. **§8.4** — CondAmbigQA mode: (a) safe / (b) full?

Everything else is approved and built; the run starts with the two decisions
above.
