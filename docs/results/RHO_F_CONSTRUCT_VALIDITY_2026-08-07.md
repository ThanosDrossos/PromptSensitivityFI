# Is ρ_F measuring what we claim? — construct-validity dossier

Prompted by Thanos's question (2026-08-07): *"turning the dial maintained ρ_F flat. What would be the
dial for ρ_F then? How can we claim it is an important metric?"* All numbers below computed from the
local parquets; commands reproducible from this file's tables.

---

## 1. The framing that resolves the puzzle

ρ_F is a property of **(question × model × generator G × decoding config)** — not of the question alone.
FI_spec is a *question-side* dial, and the framework's own thesis is that specificity moves **ability**,
not formulation sensitivity. Expecting the ability dial to move axis 2 would *contradict* the three-axis
claim, not support it. The flat response (now a properly powered null: n = 150/model, both golds) is the
**discriminant** half of construct validity.

What was genuinely missing is the **positive-control** half: an intervention that *should* move ρ_F and
does. Each axis has its own dial family, and ρ_F's dials act on G, the decoding, and the model — not on
the question text:

| dial for ρ_F | expected effect | status |
|---|---|---|
| **Generator width G** (wider paraphrase distribution: register, syntax, indirectness) | ρ_F ↑ with perturbation strength | **the missing experiment** (was R6, dropped). Untestable in current data: the NLI@0.9 gate compresses universe diversity to ESS_in ∈ [0, 0.23], and within that restricted range ρ_F–ESS_in is +.08/+.14/+.09 — range restriction, not refutation |
| **Decoding temperature** | T→0 shrinks MS_W ⇒ the *share* → its ceiling while σ²_B is stable | mechanical; validates the decomposition, not the construct. Two-point version exists in-data (T=0 vs T=1 passes) |
| **Model-side robustness interventions** (paraphrase-consistency training à la RoParQ) | ρ_F ↓ | future; the claim a model developer would actually test |
| **Evidence fragility** (answer findable only via specific keywords) | ρ_F ↑ | plausible; testable with a lexical-anchor analysis, not yet run |

## 2. The validity evidence we now have (all computed 2026-08-07)

**(a) Cross-decoding-regime convergence.** ρ_F is estimated from the T=1.0 samples. The T=0 pass is a
different decoding regime and a separate scoring pass. Partial correlation of T=0 phrasing-disagreement
(spread > 0) with ρ_F_hier, controlling for accuracy-extremeness (the mechanical channel):

| qwen | llama | mistral |
|---|---|---|
| **+0.514** | **+0.404** | **+0.541** |

**(b) Out-of-sample payoff prediction — the utility criterion.** ρ_F computed from the *first* k=10
samples predicts the rephrasing payoff (F_max − F̄) measured on the **disjoint second half** of the k=20
arm — and survives partialling out accuracy:

| model | n | Spearman(ρ_F^k10, payoff on disjoint half) | partial out accuracy |
|---|---|---|---|
| qwen | 42 | +0.609 (p = 2e-5) | **+0.635** |
| llama | 62 | +0.390 (p = .002) | **+0.379** |
| mistral | 62 | +0.701 (p = 2e-10) | **+0.708** |

This is the decision-relevant claim in operational form: **ρ_F tells you, before you try, whether
rephrasing this question will pay off on this model** — measured on data the estimate never saw.

**(c) Previously established:** split-half reliability 0.35–0.58 (> 0: a stable per-question signal
exists); gold-robustness .53–.69 across scoring conventions with identical model ordering; the islands
statistic tracks ρ_F at +0.83/+0.93 (the "some phrasings work, others don't" phenomenon *is* what ρ_F
quantifies); σ²_B confirms the model ranking on an absolute scale.

## 3. So: is it a good metric, and why is it important?

**As a descriptive statistic: yes, demonstrably.** It estimates a real, replicable quantity — the
noise-corrected phrasing-attributable share of task success — that converges across decoding regimes,
grading conventions, and readouts, and predicts an economically meaningful quantity out of sample.

**As a validated construct: incomplete, and the paper should say so.** The generator-width dial is the
missing positive control. A thermometer is not validated by being easy to move; it is validated by
agreeing with independent measurements and staying put when nothing relevant changes. We have both of
those. What we lack is the demonstration that turning ρ_F's *own* dial moves it.

**Why it is important (three arguments, in order of strength):**
1. **It answers the paper's motivating question.** A practitioner with a wrong answer cannot tell a
   knowledge failure from a phrasing failure. ρ_F is the phrasing-failure share, and (b) shows it
   predicts — out of sample — whether rephrasing will help. "Rephrase or give up" is a real decision.
2. **It is the per-item effect size for multi-prompt evaluation.** High ρ_F ⇒ a single-prompt benchmark
   score is a phrasing lottery for that item; mean ρ_F tells you how many prompts vs samples an
   evaluation budget needs (the G-theory decision-study reading; connects to Mizrahi et al., PromptEval).
3. **It is the quantity the published indices claim to measure but conflate** — with decoding noise
   (no index separates them) and with dispersion (POSIX does not discriminate; our reductions show the
   rest are one object).

## 4. What goes in the paper

- A **Construct validity** paragraph in the Discussion with table (a)+(b) above, the flat-dial result
  framed as discriminant validity, and the generator-width dial named plainly as the missing positive
  control / future work.
- The payoff-prediction check (b) promoted into Results — it is the strongest utility evidence we have
  and costs nothing.
- Never claim more than: *"a distinct, reliable-above-chance, decision-relevant trait of
  (question × model); its experimental manipulation remains future work."*
