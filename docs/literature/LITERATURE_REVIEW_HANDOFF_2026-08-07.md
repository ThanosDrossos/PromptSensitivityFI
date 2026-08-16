# Literature review — hand-off for a dedicated session

> **⚠ SUPERSEDED IN PART.** `LITERATURE_REVIEW_2026-08-07_VERIFIED.md` re-checks the ⚠️-marked items
> below against primary sources and audits the folder's own PDF library. **Use that document for
> citations.** This one remains useful only as (a) the record of what the sweep surfaced and
> (b) the open-questions list in §4 and the negative results in §3.

**Produced:** 2026-08-07, as a by-product of `REVIEW_2026-08-06_Adversarial.md`.
**Purpose:** everything the adversarial literature sweep surfaced, with an honest confidence label on each item, plus the open questions a dedicated literature session should answer.
**Method:** 8 parallel search lenses + 6 deep-read agents, 101 unique works surfaced. The session's web-search budget (200 queries) was exhausted, so **a large fraction could not be re-verified**.

---

## 0. How to read this — verification tiers

| tier | meaning | action |
|---|---|---|
| **✅ VERIFIED** | I read the PDF myself, or the work is in the project's own PDF library, or it is a landmark I can vouch for | cite freely |
| **🟡 LIKELY** | well-known venue/author/arXiv id consistent with my knowledge, but not re-checked this session | verify the claim, then cite |
| **⚠️ UNVERIFIED** | surfaced only by a search agent; several have 2026 arXiv IDs I cannot confirm | **do not cite until confirmed to exist and to say what is claimed** |

> **Warning.** Agent-reported citations are not trustworthy without confirmation. Roughly half the 2026-dated IDs below could be real, mis-transcribed, or confabulated. Treat this document as a **search plan**, not a bibliography.

---

## 1. The three findings that change what we can claim

### 1.1 ✅ Hazen et al. (2007) already applied FI to natural language — VERIFIED, read directly

`Hazen, Griffin, Carothers & Szostak (2007), PNAS 104(suppl 1):8574–8581`, pp. 8575–8576, has a titled section **"The Functional Information of Letter Sequences"**:
- configuration space = sequences of *n* letters (26ⁿ);
- degree of function E_x = "the probability that a local fire department will understand and respond to the message" — i.e. **the probability that a natural-language string evokes the intended response from a receiver**;
- I(E_x) = −log₂[M(E_x)/26ⁿ]; worked example ≈1000 of 26¹⁰ ten-letter sequences ("FIREONMAIN", "MAINSTFIRE") ⇒ **≈36 bits**;
- it even discusses degraded variants ("phonetic misspellings (FYRE or MANE), mistakes in grammar or usage… or typing errors") with lower response probability.

**Consequence.** "First application of the Szostak/Hazen formalism to prompts" and "a literature search for FI applied to prompts returns nothing" are **not defensible**. Text was one of the three founding demonstrations (letter sequences, Avida, RNA aptamers).

**Also from the same source — a constraint we violate.** Hazen: rigorous FI "requires knowledge of two attributes: (i) all possible configurations of the system … and (ii) the degree of function x for every configuration." Our U_q is 10 samples from Phi-4. So FI_in is a **generator-relative** survival rate, not Hazen's I(E_x).

**Recommended reframing** (already in the review as R5): define `FI_in^G(q,k)` with the proposal distribution G explicit; the honest novelty is *"we instantiate Hazen's letter-sequence construction with an LLM as the receiver and a controlled meaning-preserving proposal distribution, and characterise what carries over."*

### 1.2 ✅ AmbigQA's own protocol scores against all interpretations — VERIFIED, read directly

`Min, Michael, Hajishirzi & Zettlemoyer (2020), EMNLP`, §3.1–3.2: the task is to "output a set of semantically distinct and **equally plausible** answers y₁…y_n"; correctness is a **set F1** over predicted (question, answer) pairs against the full gold reference set (`c_i = max_j 1[y_i ∈ Ȳ_j] f(x_i, x̄_j)`, prec = Σc_i/m, rec = Σc_i/n).

**Consequence.** Our target-only L0 scoring is non-standard for this benchmark. This is the literature basis for the R1 union-gold arm.

**Follow-up for the dedicated session:** does any prior work use AmbigQA with a *single pinned* interpretation? If yes, how do they justify it? If no, we are the first and must justify it explicitly.

### 1.3 The FI framing is contested at its source — ⚠️ needs verification

- ⚠️ **Root-Bernstein, M. (2024)**, PNAS 121(34) e2318689121, "Evolution is not driven by and toward increasing information and complexity" — a **critical letter** on Wong/Cleland/Hazen (2023), with a published reply (e2406598121). Wong 2023 is *the* citation that licenses leaving biology. **Verify both letter and reply.**
- ⚠️ **Zenil, H. (2025)** — public critique that FI is observer-relative and conflates rarity with function; analogous to the peer-reviewed Assembly Theory takedowns. Non-peer-reviewed, so cite carefully — but the **observer-relativity objection transfers exactly** to our generator-relativity problem.
- ⚠️ **Dembski & Marks (2009)**, IEEE Trans. SMC-A, "Conservation of Information in Search" — −log₂(p) as "endogenous information". **Reputational hazard:** this is intelligent-design-adjacent literature and Wikipedia's *Functional information* page cross-links *Specified complexity*. Worth knowing before a public talk; probably not worth citing.
- ⚠️ **Cotterell, R. (2026)**, "Surprisal Theory is Tautological (without Rational Grounding)", arXiv:2607.21574 — **this title and ID look plausible but I cannot confirm it.** If real, it is the general form of the "wrapping a success rate in −log₂ adds nothing" objection.

---

## 2. Prior art by contribution

### 2.1 Against ρ_F (our claimed novel axis)

| work | tier | what it does | what survives for us |
|---|---|---|---|
| **Cox et al., AAAI 2025 39(22):23696** "Mapping from Meaning" (arXiv:2510.17028) | ✅ in our PDF library, already implemented as `rho_u` | ρ_u = U_e/U_t, per-question epistemic/total variance share of **response embeddings**, explicitly "to quantify how much LLM uncertainty is attributed to prompt sensitivity" | ρ_F is **gold-referenced** (whether the model was *right*, not merely *different*) and **noise-corrected** (ρ_u's null expectation is (N−1)/(Nk−1), not 0). `main_body.tex:277-292` already argues this well — **keep that paragraph verbatim** |
| **Romanou et al., BrittleBench** (arXiv:2603.13285) | ✅ in our PDF library | law of total variance + random-effects ANOVA; reports ICC-like Π_m = perturbation variance / total variance | Theirs is **model-level** and their inference is **deterministic**, so the sampling-noise term vanishes — which is exactly the term ρ_F needs. Ours is per-question. `main_body.tex:294-307` already handles this |
| **Żatuchin (2026)**, arXiv:2607.13304, "Where Does the Noise Come From? A Variance-Components Decomposition of Non-Determinism in LLM Brand Answers" | ⚠️ **UNVERIFIED — highest-priority check** | claimed: explicit crossed random-effects / generalizability-theory decomposition separating within-prompt resampling (34.8 %) from paraphrase and other facets, reporting ICCs | **If real, this is the most direct scoop of ρ_F.** It would force repositioning from "we introduce a variance decomposition" to "we apply G-theory per question with a gold-referenced outcome". Verify first. |
| **Pecher et al. (2026)**, arXiv:2602.04297, "Revisiting Prompt Sensitivity … The Role of Prompt Underspecification" | ⚠️ UNVERIFIED | claimed: "a significant portion of observed prompt sensitivity is attributable to prompt underspecification"; also linear probes on internal representations | **If real, this is our thesis published first.** Differentiator would have to be the *quantification* in bits from annotated interpretation counts |
| **Hou et al.**, "Decomposing Uncertainty for LLMs through Input Clarification Ensembling" (arXiv:2311.08718) | 🟡 LIKELY | decomposes uncertainty by clarifying the input — conceptually the FI_spec dial paired with the dispersion axis | must cite; check whether they already do the L0→L1 comparison |
| **Ye/Zhou et al.**, "LLM Psychometrics: A Systematic Review" (arXiv:2505.08245) | ⚠️ UNVERIFIED | IRT / psychometrics for LLM evaluation | raises the bar: IRT gives a better-identified difficulty parameter with standard errors |

**Open question for the session:** has anyone published a **per-item, sampling-noise-corrected, gold-referenced** variance share? That exact conjunction is what is left of ρ_F's novelty. Search terms that were *not* exhausted: "generalizability theory LLM", "decision study LLM evaluation", "variance components prompt paraphrase sampling", "beta-binomial overdispersion LLM prompt".

### 2.2 Against the dispersion axis and H_sem

| work | tier | relevance |
|---|---|---|
| **Kuhn, Gal, Farquhar (2023)** semantic uncertainty; **Farquhar et al. (2024)** *Nature* 630:625 | ✅ both in our PDF library | the source of H_sem. Note the correct attribution chain — Kuhn 2023 introduced it |
| **Kossen et al. (2024)**, "Semantic Entropy Probes", arXiv:2406.15927 | 🟡 LIKELY (well-known) | **linear probes on hidden states predicting semantic entropy.** This is our dispersion head, already published. **Must cite and compare** |
| **Han, Kossen, Razzak, Gal**, "Semantic Entropy Neurons" | ⚠️ UNVERIFIED | same family; localises semantic uncertainty in the representation |
| **Nikitin et al.**, "Kernel Language Entropy" (arXiv:2405.20003) | 🟡 LIKELY | fine-grained alternative to hard NLI clustering — challenges H_sem as *the* canonical dispersion measure |
| **McCabe et al.**, "Estimating Semantic Alphabet Size" (arXiv:2509.14478); **SENECA** (arXiv:2605.00668); **SHADE** (arXiv:2604.19162) | ⚠️ UNVERIFIED | small-sample entropy / alphabet-size estimation. **Directly relevant:** our \|A_q\| is observed richness at k=10 and our Chao1 attempt was dropped in one sentence |
| **Nguyen, Payani, Mirzasoleiman**, Findings of ACL 2025 (arXiv:2506.00245) | ⚠️ UNVERIFIED | claims semantic entropy degrades when clusters saturate at small k — exactly our k=10 regime |
| **Tomov et al.**, "The Illusion of Certainty: UQ for LLMs Fails under Ambiguity" (arXiv:2511.04418) | ⚠️ UNVERIFIED | **most on-the-nose title in the sweep.** If real, it directly addresses using H_sem on ambiguous questions — our exact setting |
| **Yu et al.**, "Generation Space Size" (arXiv:2510.12699) | ⚠️ UNVERIFIED | \|A\| as an open-endedness/ambiguity diagnostic |
| **Chen, Da, Liu, Wei**, "Position: UQ in LLMs is Just Unsupervised Clustering" (arXiv:2605.19220) | 🟡 possibly `37_Position_Uncertainty_Quanti.pdf` in our folder — **check** | if it argues the UQ zoo collapses to clustering, it is a **friendly precedent** for our "one construct" claim and must be positioned against |

**This is the strand with the most unverified items and the highest payoff.** Our §2.4 finding (S_τ ≡ H_sem/log₂\|A\|, Var[FI_out] ≡ Var[H_sem]) is an *analytic* result that does not depend on any of these — but the framing "the field measures one thing" needs to know whether someone said it first.

### 2.3 Against the probes

| work | tier | relevance |
|---|---|---|
| **Kossen et al. (2024)** semantic entropy probes | 🟡 LIKELY | scoops the dispersion head (see above) |
| **Azaria & Mitchell (2023)**, "The Internal State of an LLM Knows When It's Lying" (arXiv:2304.13734) | ✅ landmark | the general "linear probe on hidden states predicts correctness" precedent |
| **Kadavath et al. (2022)**, "Language Models (Mostly) Know What They Know" (arXiv:2207.05221) | ✅ landmark | self-knowledge baseline |
| **Zhang, Duan, Kim, Xu**, "Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs" (arXiv:2509.13664) | ⚠️ UNVERIFIED | **claimed to scoop the vagueness head, with cross-dataset generalisation.** High-priority check |
| **Ramesh, Dou, Xu**, "Localizing Prompt Ambiguity … (PRIG)" (arXiv:2606.05486) | ⚠️ UNVERIFIED | claimed probe-targeted attribution for prompt ambiguity |
| **Cencerrado et al.**, "No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes" (arXiv:2509.10625) | ⚠️ UNVERIFIED | claimed to scoop the reliability head |
| **Lavi, Milo, Geva**, "Detecting (Un)answerability … with Linear Directions" (arXiv:2509.22449) | ⚠️ UNVERIFIED | adjacent: answerability rather than ambiguity |
| **Query Performance Prediction (IR)** — Rabinovich et al. (arXiv:2311.01152) and the wider QPP subfield | 🟡 | a whole subfield predicts question difficulty pre-retrieval. **Our .66 OOD AUROC must be positioned against QPP numbers** |

**Open question:** is there any published *zero-shot cross-labelling-mechanism* transfer result for an ambiguity probe? That is the specific thing our OOD holdout shows (0.667 vs 0.544 for a matched-protocol text baseline), and it may be our remaining probe novelty.

### 2.4 Against the empirical headline (specificity → accuracy)

| work | tier | relevance |
|---|---|---|
| **Kim (2025)**, "DETAIL Matters" (arXiv:2512.02246) | ✅ in our PDF library | graded specificity ladder driving accuracy, with a doubling on weaker models. **The specificity→accuracy result is not new** |
| **Zi et al.**, "More Than a Score" (IJCNLP-AACL 2025) | ✅ in our PDF library | prompt specificity for code generation |
| **Keluskar, Bhattacharjee, Liu**, IEEE BigData 2024 (arXiv:2411.12395) | ⚠️ UNVERIFIED | claimed to have an "upper bound" row equal to our L1 condition on AmbigQA |
| **Cole et al.**, "Selectively Answering Ambiguous Questions" (EMNLP 2023) | 🟡 LIKELY | ambiguity → output-dispersion link |
| **Stelmakh et al., ASQA** (EMNLP 2022); **CondAmbigQA** (EMNLP 2025) | 🟡 LIKELY | second and third formalisations of the same evaluation problem; ASQA independently confirms the all-interpretations protocol |
| **Malaviya et al.**, "Contextualized Evaluations" (TACL 2025) | ⚠️ UNVERIFIED | claimed: evaluating an underspecified query without specifying context is itself invalid — **would directly support our R1 change** |
| **Kobalczyk et al.**, "Active Task Disambiguation with LLMs" (ICLR, arXiv:2502.04485) | 🟡 LIKELY | information-theoretic framing of specification — closest thing to FI_spec |

**Consequence for framing (agreed with Thanos):** the specificity→accuracy result is **not a finding**; it goes in Methods as the justification for the R1 union-gold design. That is now the plan.

### 2.5 Prompt-sensitivity index landscape (for the measurement-model section)

All ✅ (in our PDF library) unless noted: **POSIX** (Chatterjee et al. 2024, arXiv:2410.02185) · **ProSA/PSS** (Zhuo et al. 2024) · **PromptRobust** (Zhu et al. 2023) · **FormatSpread** (Sclar et al. 2024) · **Errica et al.** S_τ and 1−TVD (NAACL 2025) · **Mizrahi et al.** multi-prompt (TACL 2024) · **Cao et al.** worst-prompt (NeurIPS 2024) · **Lu et al.** (NAACL 2024) · **Seleznyov et al.** (Findings EMNLP 2025) · **Hua et al.** "Flaw or Artifact?" (EMNLP 2025) · **Polo et al. PromptEval** (NeurIPS 2024) · **Elazar et al. ParaRel** (🟡, arXiv:2102.01017).

**Key sweep conclusion (worth verifying but consistent with our own reading of the code):** *none* of these separates prompt-induced variance from decoding noise. POSIX assumes deterministic generation; Errica fixes temperature 0, seed 42, one generation per variant; BrittleBench states the inference-variance term vanishes by design. **That is the gap ρ_F fills, and it is a narrower and more defensible claim than "a new axis".**

⚠️ **Two possible collision papers to check:** Qin et al., "Evaluating and Explaining Prompt Sensitivity of LLMs" (ICML 2026 poster 65089) and Liu & Chu, "Understanding the Prompt Sensitivity" (arXiv:2604.18389).

### 2.6 On paraphrase sets as an instrument (relevant to R6, deferred)

⚠️/🟡: **Wahle et al.**, "Paraphrase Types Elicit Prompt Engineering Capabilities" (EMNLP 2024) · **Leidinger et al.**, "The language of prompting" (Findings EMNLP 2023) · **PAWS** (Zhang et al. NAACL 2019) · **Ribeiro et al. SEARs** (ACL 2018) · **Cegin et al.** (ACL 2024, LLM-generated paraphrase diversity) · **PARAPHRASUS** (COLING 2025) · **Habba et al.** (Findings ACL 2025).

**Directly relevant caveat already in our own docs:** Romanou et al. report paraphrasing is the **weakest** of their four perturbation families and warn LLM paraphrasing "might make queries easier … by standardizing language and decreasing diversity". This is the strongest argument for R6 (second, rule-based generator) whenever it happens.

**Also relevant:** NLI models are trained on declarative pairs; our bidirectional NLI@0.9 gate is applied to **interrogative** pairs. Look for published evidence on NLI failure modes for questions (Ben Abacha & Demner-Fushman 2019 is a starting point, ⚠️).

---

## 3. Negative results from the sweep (searched hard, found nothing)

These are *supportive* — record them, they are the residual novelty:

1. **No prior work applies Szostak/Hazen FI specifically to LLM prompts.** Queries run: "functional information metric applied to large language models prompts LLM"; "functional information prompt engineering LLM paraphrase surviving fraction bits metric 2025 2026"; "functional information prompt arxiv large language model Szostak Hazen import NLP metric novel"; "information content of a prompt bits measure prompt LLM task difficulty -log probability success rate metric arxiv"; "information-theoretic measure of prompt quality bits log success fraction paraphrases robustness LLM 2025 arxiv in bits".
   *But note §1.1:* the label is unclaimed; the underlying construction is not.
2. **No peer-reviewed philosophy-of-science rebuttal of the Wong/Hazen law** in Synthese, Erkenntnis, Biology & Philosophy, or BioSystems. The critical record is the PNAS letter + reply plus non-peer-reviewed critiques. **So say "disputed", never "refuted".**
3. **No prior naming of −log₂(surviving fraction) within ML/NLP** under a dedicated term. Prior namings are outside ML: Shannon self-information, Dembski–Marks "endogenous information", Hazen's "functional information".

---

## 4. Open questions for the dedicated session

Ordered by how much they change the paper:

1. **Verify Żatuchin arXiv:2607.13304.** Highest scoop risk for ρ_F. If real, read the method and decide whether ρ_F repositions as "per-question G-theory with a gold-referenced outcome" or is subsumed.
2. **Verify Pecher arXiv:2602.04297.** Second-highest scoop risk — for the whole underspecification thesis.
3. **Verify Kossen "Semantic Entropy Probes" and Zhang arXiv:2509.13664.** These bound what we can claim for the probe heads (currently deferred to R7, but the citations are needed regardless).
4. **Search the exhausted-but-unfinished strand: generalizability theory / decision studies / mixed models in LLM evaluation.** Terms not yet tried: "G-theory LLM eval", "decision study prompt budget", "variance components paraphrase sampling seed", "beta-binomial prompt overdispersion", "Bowyer don't use the CLT in LLM evals", "Miller adding error bars to evals".
5. **Has anyone published the metric-collapse result?** i.e. "prompt-sensitivity indices correlate/reduce to one construct". If yes, we cite; if no, our §2.4 *analytic* reduction is a clean contribution. Check the two ⚠️ collision papers in §2.5 first.
6. **Confirm no prior work pins a single AmbigQA interpretation as gold.** Determines whether R1 is a correction or a novel protocol contribution.
7. **Find the QPP (query performance prediction) numbers** to position the .66 OOD AUROC honestly.
8. **Decide the Wong 2023 citation policy** given the Root-Bernstein letter and the Dembski–Marks adjacency.

---

## 5. What the literature *confirms* about our work (do not lose these)

- **Errica et al.**: sensitivity need not track accuracy — supports our axis-1/axis-2 separation in principle.
- **Hua et al. "Flaw or Artifact?" (EMNLP 2025)**: exact match inflates apparent sensitivity — we already follow this (NLI-with-gold, never exact match). ✅ correct call.
- **Romanou et al.**: paraphrasing is the weakest perturbation family — this makes our effects *conservative*, and should be stated as such.
- **The "no index separates phrasing from decoding noise" gap appears real** across POSIX / ProSA / Errica / FormatSpread / BrittleBench. This is the surviving core of ρ_F.
- **About half of Natural Questions are ambiguous** (AmbigQA) — the practical motivation for the whole axis is solid and citable.

---

## 6. Raw material

The full 101-entry threat dump (severity, lens, citation, URL, what-they-did, claim-threatened, how-to-respond) is at:
`<scratchpad>/threats.json` — regenerate from the workflow journal if it has been cleaned up:
`.claude/projects/<project>/<session>/subagents/workflows/wf_25bf7a1b-9c3/journal.jsonl`.
