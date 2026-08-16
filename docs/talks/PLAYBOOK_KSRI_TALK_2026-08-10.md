# Talk playbook — KSRI seminar, 2026-08-10

Matched to **your** final deck: `20260809_KSRI_Seminar_Prompt_Sensitivity_final_copy.pptx`
(33 slides: 19 main + Backup divider + 13 backup). Audience: KSRI/WIN postdocs, first contact
with the project. Target: 13–14 min main path + Q&A.

Every number here is from the verified artifacts (`data/stats_hygiene.md` is the source of
truth for inference). Structure per slide: **Say** (talk track) · **Must land** (the point that
has to survive even if you improvise) · **If asked** (likely questions + answers).

---

## Part 1 · Slide-by-slide

### 1 · Title (~20 s)
**Say:** "Everyone in LLM evaluation talks about prompt sensitivity as if it were one number.
I'll show you it is three separate measurements, that most published indices are secretly the
same one, that the missing one can be measured per question, and that each measurement can be
validated by its own intervention."
**Must land:** the one-liner: *three measurements, not one number.*

### 2 · Agenda (~15 s)
**Say:** "Structure mirrors a paper: background, gap, methods, results, contributions." Skip fast.

### 3 · Motivation: One question, three phrasings (~75 s)
**Say:** walk left to right. "A user asks about The Godfather. There are many equivalent
phrasings — verified equivalent, same meaning. The user picks one, plus a level of detail,
without knowing either matters. The model answers one correctly and two wrongly. Nothing marks
the losing phrasings in advance." Then the three consequence chips: trial-and-error prompt
engineering, silent accuracy loss, unmeasured per-question reliability.
**Must land:** *phrasing, not knowledge, often decides the outcome — and nobody can see it coming.*
**If asked** "is the Godfather example real data?": "Illustrative — but the pattern is real and
quantified later: in our grid, some phrasings of the same question succeed and others fail, and
ρ_F measures exactly how much of that is systematic." Do not claim these three specific answers
were measured.

### 4 · Background: three families (~60 s)
**Say:** organize, don't enumerate. "The literature measures prompt sensitivity in three ways.
Family one: how much the *answers* vary — agreement, consistency, semantic entropy. Family two:
how much benchmark *scores* drop under perturbation — model-level. Family three, from IR
evaluation and psychometrics: variance decomposition — which share of variance comes from which
source." Close on your chevron: "a rich zoo — many measurements, or one with many names? That
question gets a proof later."
**Must land:** the three *hows* (compare answers / re-score perturbed benchmarks / decompose variance).
**If asked** "what about per-question success measures?" (a postdoc who knows ProSA/Cao):
"Good catch — ProSA and worst-prompt-performance do score success per question. What none of
them does is separate the wording effect from sampling noise: single greedy decode, T = 0, or
raw spread. That separation is exactly the gap we fill." (This is the verified defence; the
slide families are about the dominant framing, not a universal negative.)

### 5 · Functional information (~60 s)
**Say:** "One ruler for everything: functional information — minus log two of the fraction of
possibilities that achieve a function. Rare success, many bits. Szostak and Hazen built it for
molecules; we instantiate it twice for prompts. FI_in over the *input* space: of all rephrasings
of a question, what fraction works? FI_out over the *output* space: how sharply does one prompt
narrow the answer distribution — log of the answer-space size minus semantic entropy. To our
knowledge, the first use of FI to evaluate LLMs."
**Must land:** one ruler, two spaces (input = rephrasings, output = answers).
**If asked** "isn't this just renaming accuracy/entropy?": "Exactly — and that is a *result*, not
an accident: AUFI is provably accuracy, FI_out is provably log m₀ minus H_sem. FI is the lens
that makes those identities visible; the contributions are the measurement model and ρ_F, not
the formalism." **If asked** "first use of FI on language?": "Hazen himself scored letter
sequences with human receivers in 2007 — the construction isn't new; applying it to *evaluate
LLMs* is, to our knowledge." (The slide says "to our knowledge" — keep that hedge out loud.)

### 6 · Research cycle (~45 s)
**Say:** the four boxes in order: practical problem (users can't tell if phrasing/detail will
decide the outcome) → research question ("how can sensitivity to phrasing and to semantic change
through added context be quantified, per question?") → research problem (indices describe the
response distribution, mostly one quantity, none use FI) → research answer (three-axis model,
each axis validated by its own intervention, plus a prompt checker).
**Must land:** the RQ, verbatim — it anchors the contributions later.
**Note:** your RP box says "None use functional information" — safe (verified: no FI-for-LLM
precedent). The phrase "mostly one quantity" is the Result-1 proof; say "as I'll *prove* later".

### 7 · Methods divider (~5 s) — "Three parts: the testbed, the design, the instrument."

### 8 · The testbed: AmbigQA (~60 s)
**Say:** "Real Google queries. Annotators found that many are ambiguous — they enumerated the
valid readings, wrote one disambiguated rewrite per reading, attached gold answers and their own
evidence snippets. The Nun: two films, 2018 and 2013 — two readings, one bit of missing
specificity. That gives us three things for free: a *human-written, model-free* specificity dial
(ambiguous versus disambiguated version); an honest grading protocol (several valid answers);
and evidence so answering is reading, not recall. 2,002 validation questions, 150 after filters."
**Must land:** FI_spec = log₂ m₀ bits *from annotations* — the dial is not model-generated.
**If asked** "why only 150?": "Two filters: at least two annotated readings, and the evidence
must contain the gold answer — so failure means reading/phrasing, not missing knowledge. 2,002 →
1,172 ambiguous → 609 with covering evidence → first 150, deterministic order."

### 9 · Design (~75 s)
**Say:** left to right: "AmbigQA, two filters, 150 questions. Each exists at two specificity
levels. Per question and level we build ten verified rephrasings. Three open models answer each
ten times. Everything is scored against two gold sets — the pinned target reading AND the union
of all valid readings. Roughly 90,000 graded answers per gold set. Two dials and only two:
specificity acts on the question, generator width acts on the rephrasings. Everything else —
evidence text, gold answers, the generator — is held constant."
**Must land:** *only two things ever vary by design.* And say the union/target sentence here —
it pays off on slide 14.
**If asked** "why hold evidence constant?": "A closed-book pilot left most questions
unanswerable — with evidence, failure is attributable to reading and phrasing, not recall; and
evidence must never become a second, uncontrolled dial."

### 10 · The instrument (~45 s)
**Say:** "The filters ARE the instrument. Phi-4 drafts up to 120 candidates; gate one keeps
meaning constant — bidirectional NLI entailment at 0.9; gate two keeps answerability constant —
a gold-preservation judge; dedup, cap at ten. Whatever varies downstream is form, not content.
And every rejection is logged — the gates themselves become a finding later."
**Must land:** meaning held constant *by construction*; FI_in's possibility space is exactly this universe.
**If asked** "NLI on questions — is that even in-distribution?": "No — entailment is defined on
statements. We treat it as a high-precision semantic-match heuristic, backed by the gold judge,
with an unquantified error rate. It's a disclosed instrument limitation, shared by everyone
filtering paraphrases."

### 11 · The measurement model (~90 s — THE slide)
**Say:** row by row. "Competence: how well — graded accuracy, union gold. Formulation
sensitivity: does the *wording* decide success — ρ_F, the share of success variance attributable
to phrasing, an intraclass correlation with a hierarchical estimator. Output dispersion: how
scattered — H_sem, semantic entropy. And the manipulated variable FI_spec — that's the dial, not
an outcome. The last column is the validation plan: each axis has its own dial, and the claim
will be that each dial moves its axis and nothing else."
**Must land:** ρ_F's definition in one sentence: *of the variance in whether the model succeeds,
the share caused by how you phrase it, after subtracting sampling noise.*
**If asked** "competence vs accuracy?": "Competence is the construct; accuracy is its estimator —
and only with three corrections: union gold (not the pinned-reading lottery), mean over the
universe (not one phrasing), k = 10 samples (not one decode)."

### 12 · Results divider (~5 s) — "Four results: a proof, two dials, and an artifact."

### 13 · Result 1: the zoo collapses (~75 s)
**Say:** "Every published index lands on one of our three axes. Right column: five indices are
*exact identities* of H_sem — error at machine precision; that's algebra, not correlation.
Left: AUFI is analytically accuracy. Middle: the convergent family — Cox's ρ_u is the same ICC
estimand on embeddings without gold, agreement about .67; Cao's spread is the uncorrected
variant; the islands statistic tracks ρ_F at .83 to .93. POSIX is the one genuinely separate
measurement and it cannot be assigned — it loads on both axes equally. And no fourth axis: PCA
plus Horn's parallel analysis retains exactly three factors, 74.6 % of variance in the top three."
**Must land:** *identities, not correlations* — we prove the collapse, we don't estimate it.
**If asked** "PCA or Horn — what did you do?": "Both, they're complementary: eigendecomposition
of the mean within-stratum Spearman matrix gives the 74.6 %; Horn is the retention rule — it
simulates same-sized random data and keeps only components whose eigenvalue beats noise; exactly
three survive. And note the deliberate order: theory proposed three, algebra removed eleven,
Horn confirmed no fourth, and the dials — next slides — validate causally." **If asked** about
"ρ = −1" on the slide: "minus one at two decimals; the exact value is −.9997 graded, and the
identity is exact under binary scoring."

### 14 · Result 2: specificity dial (~90 s)
**Say (IMPORTANT — the union/target explanation is NOT on the slide, do it verbally):**
"Left panel, two bars per model. Gray: scored against the annotator's pinned reading — the
effect looks like +22 to +25 points. Green: scored against the union of ALL valid readings —
the dataset's own protocol — and the honest effect is +6 to +13 points. The gap is what we call
the grading lottery: scoring one pinned reading rewards *guessing the annotator's reading* —
47 to 72 % of the naive effect is protocol artifact, not ability. Middle: dispersion falls, but
Holm-robust in one model only. Right: ρ_F does not move — and that's a *powered* null: n = 150,
confidence intervals exclude anything beyond ±0.04, replicated under both gold sets. The dial
that lifts accuracy by up to 13 points leaves formulation sensitivity exactly flat."
**Must land:** honest effect +6–13 pts (BH 3/3, Holm 2/3) *and* ρ_F flat by design of the construct.
**If asked** "why is qwen's effect only BH-significant?": "Family of 12 tests; qwen's +.064
survives Benjamini–Hochberg (.045) but not Holm (.15). The pooled per-question test across
models is p = 3×10⁻⁵. We report both corrections and don't oversell." Backup: slide 25 (full
table + CIs), slide 29 (lottery decomposition).

### 15 · Result 3: width dial (~90 s)
**Say:** "The construct predicts its own dial: ρ_F decomposes variance over the paraphrase
distribution, so *widening* that distribution must raise it — and must not touch competence or
dispersion. Preregistered before any data. Left: ρ_F rises with width in all three models —
significant in Qwen, and coherently so: the most phrasing-sensitive model has the most to
amplify. Right: accuracy flat; H_sem flat too (Friedman p ≥ .47 everywhere). Together with the
previous slide, that is a double dissociation: each axis moves under its own dial and only its
own. And the swap strip: we replaced generator AND judge with OLMo-2 — a different model family —
per-cell agreement +.41/+.27/+.51, consistent with the reliability ceiling, ranking preserved.
First paraphraser-swap ablation in this literature: levels are generator-relative, structure is
generator-robust."
**Must land:** the double dissociation sentence + "first swap ablation in this literature"
(verified clean: PTEB lists it as open).
**If asked** "which estimator on the left chart?": "Method of moments on covered cells — the
per-arm hierarchical fit is not identifiable on the narrow arm (seven paraphrases, mostly
degenerate cells); trend p = .004/.079/.051, and Qwen is stable over five matching seeds."
**If asked** "why is the wide arm barely wider than production?": "The NLI gate censors it —
64 % rejections. The equivalence filter, not the generator, bounds realizable width; every
NLI-filtered evaluation inherits that. Backup slide 31 has the numbers."

### 16 · The prompt checker (~75 s)
**Say:** "Everything so far costs ~200 model calls per question — diagnostic, not usable at
typing time. So: one forward pass, no generation; take the hidden state at the last prompt
token, mid-depth; a linear head reads it. We train four heads — one per row of the measurement
model: competence, formulation sensitivity, dispersion, and the underspecification head for the
dial. The chart is the underspecification head: in-domain it reaches .87 AUROC versus .76 for
the best text baseline. The blue bars are the real claim: the *frozen* head on annotator-labelled
questions — labels from a different mechanism, never trained on — holds .67–.68 while frozen
text baselines drop to .55–.59. And one honest null: ρ_F itself is NOT linearly readable —
chance-level probes."
**Must land:** warning *before* any answer is generated; the claim is zero-shot transfer and
label efficiency, not raw detection ("with in-domain labels, bag-of-words catches up").
**If asked** "why four heads, not three?": "The table has four rows: three outcome axes plus the
manipulated variable. The dial head is the actionable one — a user can fix underspecification —
and the only one with external labels to transfer to." **If asked** "and the width dial?":
"Width is a property of the *instrument*, set by us — known by construction, no deployment
meaning. Specificity is a property of the *user's question* — unknown and fixable. That's why
one gets a head and the other doesn't." **If asked** operating point: "At the shipped 0.65
threshold it flags 67–73 % at precision ≈ .66 against a .59 base rate — a ranking signal, not a
hard gate." Backup: slide 26.

### 17 · Contributions (~45 s)
**Say:** "C1: a measurement model that tidies the zoo — three questions, three representatives,
and the collapse is proved, not correlated. C2: ρ_F, the missing axis — the share of success
variance attributable to phrasing, gold-referenced, noise-corrected, validated by its own dial
and by the literature's first paraphraser-swap ablation. C3: a usable artifact — 'your prompt is
underspecified' from one forward pass, transferring without target-domain labels. Together they
answer the research question from the cycle."
**Must land:** each contribution maps onto one clause of the RQ (what to quantify / how, validly / actionably).

### 18 · Limitations and Future Work (~30 s)
**Say:** pick three, don't read: "One dataset; sensitivity operationalized through
meaning-preserving paraphrases only — the mildest perturbation family, so effects are lower
bounds; generators are LLMs — a human-written universe is untested; three mid-size models. Going
forward: measurement models as the audit layer for LLM evaluation, prompt-QA gates that consult
ρ_F — 'is rephrasing worth it?' — and a real dose-response ladder for specificity."
**⚠ Watch-outs on this slide:** (1) typo "liekly" → "likely" (fix before presenting if you can);
(2) "Longer prompts are more likely to be censored by the NLI gate" — we never *measured*
length-vs-censoring; the measured facts are wide-arm NLI rejection 64 % and narrow-arm dedup
rejection 84 %. If probed, soften to "the gates censor, plausibly length-dependent — measured
per-arm, not per-length."

### 19 · Thank you (~end)
Then drive Q&A from backup (see Part 3).

---

## Part 2 · Backup slides — when to pull which

| # | Slide | Pull when… |
|---|---|---|
| 21 | What is missing (gap) | someone asks "what exactly was the gap?" ⚠ **carries the OLD wording** — see Watch-outs |
| 22 | Key takeaways | closing summary requested; also a good final answer to "so what?" |
| 23 | Result 4: ρ_F as instrument | reliability/validity questions — coverage, split-half, payoff prediction chart |
| 24 | RQ answered | "did you actually answer your question?" — three-line summary |
| 25 | Endpoint table | any request for CIs / exact p-values / Holm-BH detail on the specificity dial |
| 26 | Probe full numbers | probe details: per-head AUROC, operating point, controls |
| 27 | Problems we faced | "what went wrong?" — the lottery, estimator traps, gate censoring; strong slide, shows rigor |
| 28 | Pivots since January | supervisor-style "how did the project evolve?"; the dose-response → measurement-model story |
| 29 | Grading lottery | anyone challenges the +6–13 vs +22–25 discrepancy — the decomposition table |
| 30 | Statistical hygiene | stats-minded pushback: family, corrections, correlated models, forking paths |
| 31 | Width dial numbers | estimator/censoring detail for Result 3 |
| 32 | Related-work positioning | "how is this different from X?" ⚠ header says "Validated by intervention" — see Watch-outs |
| 33 | Formulas | anyone wants FI_in / ρ_F / H_sem / FI_spec formally |

**⚠ Watch-out, backup 21:** this is the older gap wording with three universal negatives. Two are
attackable as written: ProSA/BrittleBench/Cao DO reference task success per question (they just
don't separate sampling noise), and POSIX IS shown to move under few-shot exemplars (a training
knob, not a planted dial). If you show this slide, add verbally: "…more precisely: none separates
the wording share of per-question success from sampling noise, and none is validated by a
*planted* dial with controls or a paraphrase-generator swap." Then it's bulletproof.

**⚠ Watch-out, backup 32:** column header "Validated by intervention" — same POSIX loophole.
Say "validated by its *own* dial" out loud. Also: ρ_u's noise-corrected cell shows ✓ on your
version; the verified reading is that Cox compares against a 1/n baseline and does *not* subtract
the within-term — if pressed, concede the column is generous to Cox and the difference is the
gold reference + the subtraction.

---

## Part 3 · Q&A arsenal (hardest questions first)

**"Żatuchin / Messing already did this variance decomposition."**
"Yes — and we cite them. Żatuchin decomposes wording vs resampling for *sentiment about brands*,
corpus-level; Messing for *aggregate benchmark scores*. Neither delivers a per-question wording
share of gold-referenced task success under sampling. The estimator is textbook G-theory —
Urbano 2013 — the *estimand* is what's new."

**"Cox's ρ_u is the same thing."**
"Same ICC idea, per question — closest prior art, cited. Two differences: ρ_u reads response
*embeddings* with no gold answer, so it measures uncertainty share, not success share; and it
benchmarks against 1/n rather than subtracting the within-phrasing term. Empirically they
converge at about .67 — which is validation for both, and they are not the same measurement."

**"Semantic entropy / probes are already published (Kuhn, Farquhar, Kossen, Zhang)."**
"Adopted, not claimed: H_sem *is* semantic entropy — that's the point of the collapse result.
Kossen probes semantic entropy from hidden states, Zhang detects ambiguity on AmbigQA — both
cited; our probe delta is the paraphrase-distribution targets and the frozen zero-shot
evaluation against protocol-matched text baselines."

**"Your reliability is only .38–.57 — isn't ρ_F useless?"**
"It caps *per-question point claims*, which we never make. Population, ranking, and dial claims
average over 150 questions. It's a 10-item questionnaire: you compare groups with it, you don't
diagnose individuals. Spearman-Brown says 4–6× more paraphrases would push it to .8 — a compute
choice, not a construct failure. And it predicts rephrasing payoff out of sample at +.39 to +.70."

**"Only 3 factors because you built the metrics that way?"**
"Partly — five indices are *identities*, so their co-loading is arithmetic. That's why the factor
analysis is deliberately the weakest leg: theory proposed three axes, algebra collapsed the rest,
Horn found no fourth dimension, and the interventions did the real validation — each axis moves
under its own dial only. Imposed by theory, confirmed by data, validated causally."

**"Everything depends on your paraphrase generator."**
"Correct, and FI is explicitly generator-relative — bits are only comparable within a fixed G.
That's why we ran the swap: generator AND judge replaced by OLMo-2, different family — per-cell
structure and model ranking survive; levels shift. Levels are G-relative, structure is G-robust.
No published metric had tested even that."

**"Two points aren't a dose-response."**
"Agreed — that's exactly the pivot from the January design. Direction is causal (human-written,
within-question); per-bit dosage is not identified. The multi-level ladder is named future work."

**"Why should IS people care?"**
"Because it's a measurement-model problem, not an NLP problem: LLM components in business
processes have phrasing-dependent reliability that nobody manages. The deliverables are IS
deliverables: a validated instrument, its psychometrics, and a deployable quality gate —
ρ_F as a monitored quality attribute of an LLM service, like latency or cost."

**"Would this hold on GPT-5-class models?"**
"Unknown and honest limitation: we need hidden states and decoding control, so three open 7–8B
models. The *method* transfers to any model exposing logits/states; the findings are claims
about these three."

**"Your accuracy effect halved when you changed the grading — isn't that suspicious?"**
"That's us catching our own confound, before reviewers did: scoring one pinned reading of an
ambiguous question counts 'guessed the annotator's reading' as ability. AmbigQA's own protocol
scores all valid readings. We report both scorings, quantify the artifact at 47–72 %, and use
the honest number as primary. It's in the paper as a protocol contribution."

---

## Part 4 · Numbers to have cold (all from data/stats_hygiene.md unless noted)

| Quantity | Value |
|---|---|
| Δ accuracy, union gold (Q/L/M) | **+.064 / +.125 / +.119** — BH 3/3, Holm 2/3; pooled p = 3.3×10⁻⁵ |
| Δ accuracy, target gold | +.225 / +.238 / +.247 → lottery share **72/47/52 %** |
| Δ H_sem | −.124 / −.433 / −.139 bits (Holm-robust in Llama only) |
| Δ ρ_F (specificity dial) | −.002 / +.013 / +.013 — n.s., CIs exclude \|Δ\| > .04 |
| ρ_F levels (hier., target gold) | Qwen .49 > Mistral .25 > Llama .13 |
| Width dial ρ_F (MoM, n→p→w) | .356→.463→.519 (p=.004) / .113→.135→.138 (.079) / .211→.230→.276 (.051) |
| Width: accuracy / H_sem | flat, Friedman p ≥ .47; realized width 4.6 < 8.4 < 9.2 tokens; wide-arm NLI rejection 64 % |
| Swap agreement (ρ_F per cell) | +.41 / +.27 / +.51; ranking preserved (Qwen .36 > Mistral .31 > Llama .13) |
| Split-half reliability | .38 / .52 / .57 |
| Payoff prediction (out-of-sample) | +.61 / +.39 / +.70 (survives accuracy partialling) |
| Factor structure | Horn: exactly 3; top-3 = 74.6 % of 14 metrics |
| Identities | S_τ, FI_out(fixed), Var[FI_out] ≤ 5×10⁻¹⁶; AUFI ≡ accuracy (ρ = −.9997) |
| POSIX discrimination | n.s.: p = .42 / .63 / .054 |
| Probe in-domain / zero-shot | .873–.874 vs .756 text; **.670–.678 vs .545–.587 frozen text** |
| Probe operating point | flags 67–73 %, precision ≈ .66, recall .77–.83 (base rate .59) |
| Scale | 150 q × 2 levels × 3 models × 10 × 10 ≈ 90,000 graded answers per gold set |

**Order convention: all triples are Qwen / Llama / Mistral.** (Exception: ρ_F level rankings are
stated as Qwen > Mistral > Llama.)

---

## Part 5 · If you're running over

Cut order: 12 (divider words) → shorten 4 to 30 s ("three families, all response-side") →
shorten 5 to 30 s (definition + "two spaces") → compress 13 to the chevron + Horn box.
Never cut: 11 (the model), 14–15 (the dials), 16 (the artifact).
