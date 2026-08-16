# §7 Expanded: Functional Information for Prompts

**Purpose.** This document replaces and extends §7 of the research synthesis. It builds the definition of Functional Information (FI) for prompts from first principles, reconciles it with the two perspectives you already formulated to your supervisor (Input-Raum / Output-Raum), commits to one primary implementation, and specifies estimator, validation, and pitfalls.

**Claim in one sentence.** Adapt Hazen, Griffin, Carothers, and Szostak (2007) `I(E_x) = -log_2[F(E_x)]` from biopolymer space to prompt space. The adaptation yields two mathematically distinct but complementary indices, FI_in (Input-space) and FI_out (Output-space), that map exactly onto the two perspectives you already gave your supervisor. The primary deliverable is FI_in, because it is the direct translation of Szostak/Hazen and the gap in the prompt-sensitivity literature.

---

## 7.1 Ground-truth definition of Functional Information

The original biology formalism is compact. Two papers fix the definition and one 2023 paper extends it beyond biology. All three are necessary to justify the adaptation to prompts.

Szostak (2003, Nature 423, 689) introduces functional information as "−log₂ of the probability that a random sequence will encode a molecule with greater than any given degree of function." The example given is an RNA aptamer that binds ATP with micromolar affinity: the probability that a random 70-mer does this is about 10⁻¹¹, which corresponds to ≈ 37 bits of functional information (Szostak 2003, p. 689). Szostak stresses that FI "is not a property of any one molecule, but of the ensemble of all possible sequences, ranked by activity" (p. 689).

Hazen, Griffin, Carothers, and Szostak (2007, PNAS / NBK254308 book chapter, p. 28–29) formalize it:

> Let E_x denote the "degree of function x" of a configuration. Then functional information is
>
> **I(E_x) = −log₂[F(E_x)]**
>
> where F(E_x) is the fraction of all possible configurations of the system that possess a degree of function ≥ E_x. Equivalently, for a system with N possible configurations and M(E_x) achieving function ≥ E_x,
>
> **I(E_x) = −log₂[M(E_x) / N]**.

Two geometric features of the FI-vs-function curve are reported in Hazen et al. (2007, pp. 35–38): *exponential decay* (high-function configurations are exponentially rare) and *stepped behavior* (function plotted against FI shows discontinuities, interpreted as "islands of solutions" in configuration space). The step structure matters for prompts, because a prompt family is likely to show discrete quality tiers rather than a smooth gradient: keep the stepped-FI-curve expectation as a sanity check on experimental results.

Corona, Di Benedetto, Gabriele, Giancarlo, Utro (Dagstuhl 2010, p. 2) confirm the formula on a second biological application (protein-DNA interaction at genome scale) and show that FI is empirically distinct from Shannon entropy and Kolmogorov complexity, i.e. it carries information that cannot be recovered from sequence-only complexity measures. The pattern transfers: we should expect FI for prompts to be distinct from token-level entropy of the prompt string.

Wong, Cleland, Hazen et al. (PNAS 2023, "On the roles of function and selection in evolving systems," DOI 10.1073/pnas.2310223120) generalize FI away from biology. They argue that any system of many configurations with a selection process acting on function will evolve to high-FI regions. This is the move that licenses our use of the framework outside biology: FI is well-defined for any (U, F) pair where U is an enumerable configuration space and F is a measurable function, per Wong et al. (2023, section "A general law" and "functional information revisited").

**Two structural conditions** for the formalism to apply, both in Hazen et al. (2007, p. 30) and re-stated by Wong et al. (2023):

1. The configuration space U is enumerable or sample-able.
2. The function F is measurable on every configuration, and produces a scalar degree of function.

Both conditions are satisfiable for prompts, though condition 1 requires care (see §7.5.1).

---

## 7.2 Two translation routes, both of which appeared in your supervisor note

Your supervisor note articulated the two perspectives precisely. The German phrasing maps one-to-one onto the two mathematical directions the FI formalism can take.

> "Input-Raum: Welcher Anteil der Prompt-Varianten liefert einen akzeptablen Output?"

This is FI over the prompt space. Configurations = prompt strings; function = accuracy (or any quality measure) of the output. It is the direct translation of Szostak/Hazen: a prompt is to its response what an RNA sequence is to its binding target.

> "Output-Raum: Wie „restriktiv" ist ein Prompt – also wie stark schränkt er den zulässigen Output-Raum ein?"

This is FI over the output space. Configurations = possible response strings; function = acceptability of the response; the prompt x acts as a filter on this space. The quantity of interest is how sharply a prompt x narrows the distribution of responses relative to a reference distribution.

These are duals, not alternatives. A prompt family can be high-FI on one axis and low on the other, and the diagnostic value of the metric comes from reporting both. §7.3 and §7.4 give the formal definitions; §7.6 explains why FI_in is the right primary deliverable.

---

## 7.3 FI_in: Functional Information in prompt space (Input-Raum)

### 7.3.1 Definition

Fix a query q expressed by a canonical reference prompt x_0. Define:

- **U_q**: the set of all prompt strings semantically equivalent to x_0. This is the prompt-space analog of Hazen's "sequence space."
- **F(x)**: the degree of function of prompt x ∈ U_q, measured as task performance (for Q&A: 1 if the generated answer is correct under exact-match or NLI-with-gold; for open-ended generation: FLASK or PEEM score in [0, 1]).
- **k**: a threshold degree of function.
- **N_k(q) = |{x ∈ U_q : F(x) ≥ k}|**: the number of prompts in U_q that achieve function at least k.

Then

**FI_in(q, k) = −log₂( N_k(q) / |U_q| )**.

Reading of the formula:
- FI_in = 0 bits means every paraphrase achieves function k. Low prompt sensitivity.
- FI_in = log₂|U_q| bits means only one paraphrase achieves function k. High prompt sensitivity, only one "magic phrasing" works.
- FI_in undefined (∞) means no paraphrase achieves function k. The query is effectively unanswerable at that threshold by this model.

This is literally the translation of Hazen et al. (2007, p. 29) `I(E_x) = −log₂[M(E_x)/N]` with the substitution (biopolymer sequence → prompt string), (degree of function → task performance), (|sequence space of length n| → |paraphrase universe of query|).

### 7.3.2 Function of k, not a scalar

Following Hazen et al. (2007, Figs. 2.1, 2.2), FI is a function of the threshold, not a number. Report the FI_in curve over k ∈ {0.25, 0.5, 0.75, 1.0} for continuous F, or over the discrete function levels when F is discrete. The expected shape is decreasing-but-stepped: FI_in rises as k rises, and Hazen et al. predict "islands of function" that appear as discontinuities. Seeing steps in the empirical FI_in(q, k) curve would be a direct analog of Figure 2.1 in the Hazen chapter and is worth reporting.

### 7.3.3 Summary-statistic

For scalar reporting in tables, define

**AUFI_in(q) = ∫₀¹ FI_in(q, k) dk**

(Area Under the FI_in curve). This single number integrates over all thresholds and is comparable across queries. Report AUFI_in alongside the curves, not instead of them.

### 7.3.4 Per-model

FI_in depends on the model, because F(x) = F(x; M) is model-dependent. Denote FI_in(q, k; M) when disambiguation is needed. Cross-model reporting is the interesting axis: a more robust model has lower AUFI_in at fixed U_q.

---

## 7.4 FI_out: Functional Information in output space (Output-Raum)

### 7.4.1 Two candidate formalizations

Your "Output-Raum" perspective asks how strongly a single prompt x constrains the output space. There are two coherent ways to turn this into an FI-style quantity, and I will take both seriously before picking.

**Candidate A: output-rarity view.** Define Y as the set of possible responses (bounded in practice by the model's output distribution support). Fix an acceptance predicted A(y) ∈ {0, 1} (the answer is "correct" or not). Then

FI_out^A(x) = −log₂( P_model[A(Y) = 1 | X = x] )

This has the Szostak form applied to outputs: the number of bits of surprise the prompt delivers when it lands an acceptable answer. A prompt with P = 10⁻³ of producing a correct answer has FI_out^A = 10 bits; a prompt with P = 1 has 0 bits. But the reading is inverted: here HIGH FI_out^A means the prompt is bad, not restrictive. This breaks the intuition and should be rejected for our purpose.

**Candidate B: restrictiveness-as-entropy-reduction view.** This is the one that matches your supervisor phrasing. Let H_0(q) be the entropy of the response distribution under an uninformative prior (the model's marginal over "any answer the query class admits"). Let H_sem(Y | X = x) be the semantic entropy (Farquhar, Kossen, Kuhn, Gal, Nature 2024) of responses conditional on prompt x, computed by sampling k completions and clustering them by bidirectional NLI entailment. Define

**FI_out(x) = log₂|𝒜_q| − H_sem(Y | X = x)**

where |𝒜_q| is the effective size of the semantic answer space for query class q, estimated from the union of semantic clusters observed across the whole paraphrase universe U_q. Reading of this formula:
- FI_out(x) = log₂|𝒜_q| means the prompt produces one semantic cluster with probability 1. Maximally restrictive.
- FI_out(x) = 0 means the prompt produces a uniform distribution over all observed semantic clusters. Minimally restrictive.
- Negative FI_out is in principle possible if we under-estimate |𝒜_q|, but is bounded below by −H_sem_max and should be clamped at zero in reporting.

This is the `log N − H` form of the Kullback–Leibler divergence from uniform, which is the cleanest information-theoretic expression of "how much does this prompt narrow the output distribution."

### 7.4.2 Restrictiveness profile

Compute FI_out(x) for every x ∈ U_q and aggregate:

- **Mean restrictiveness**: E_{x ∈ U_q}[FI_out(x)]. Characterizes the family.
- **Variance of restrictiveness**: Var_{x ∈ U_q}[FI_out(x)]. High variance means "how restrictive the prompt is" itself depends strongly on phrasing, which is a second-order sensitivity signal not captured anywhere in the literature reviewed in §2 of the synthesis.

### 7.4.3 Connection to Errica et al. (NAACL 2025) sensitivity S_τ

Errica's sensitivity S_τ(x) = −E[ln p_τ(y|x)] / ln C is normalised token-or-class entropy. For multiple-choice tasks with C classes and no semantic clustering, S_τ(x) = H(Y|X=x) / log₂ C, which means

FI_out(x) = log₂ C − log₂ C · S_τ(x) = log₂ C · (1 − S_τ(x))

i.e. FI_out is a linear rescaling of (1 − Errica's sensitivity) for multiple-choice. They are equivalent up to units. This is a feature: it shows FI_out is not novel math on discrete-output tasks, it is a re-interpretation of an existing sensitivity measure in bits of "functional information in output space." On open-generation tasks, the semantic-entropy version gives FI_out content that Errica's formulation does not, because it uses NLI clustering rather than token distribution.

---

## 7.5 Which perspective is primary

For the seminar deliverable, FI_in is the primary contribution. FI_out is the reported companion. The reasoning is specific:

1. **Literature gap.** The web search for Szostak/Hazen FI applied to prompts returns nothing ([search results](https://www.pnas.org/doi/10.1073/pnas.0701744104), [results listing](https://aclanthology.org/2025.acl-long.1562/); no hit combines FI with LLM prompts). FI_in is genuinely novel here; FI_out is one rescaling away from Errica (NAACL 2025) on multiple-choice and one step beyond Farquhar (Nature 2024) on open generation, i.e. the novelty margin is smaller.

2. **Direct Szostak/Hazen translation.** FI_in = −log₂(N_k/|U_q|) is the exact 2007 formula with one substitution (sequence space → prompt space). FI_out requires a secondary move (fixing the reference |𝒜_q|) and sits adjacent to existing entropy metrics.

3. **Supervisor framing.** Your "Welcher Anteil der Prompt-Varianten liefert einen akzeptablen Output?" is formally FI_in at threshold k ≈ "acceptable." The metric answers exactly the question you already posed.

4. **Diagnostic independence.** FI_in uses only (x, y, ground-truth) and needs no logit access. It works on closed-API models. FI_out needs either logits or a sampling budget of k ≈ 10 per prompt for semantic entropy. This matters for reproducibility and cost.

5. **Actionable deliverable.** FI_in gives a curve and an integrated scalar per (query, model). That is exactly what a seminar paper needs to show cross-model and cross-task results on.

FI_out is reported alongside because (a) it closes the Input-Output dual your supervisor already sees and (b) the variance-of-restrictiveness is a genuinely novel second-order signal.

---

## 7.6 Implementation specification

### 7.6.1 Paraphrase universe U_q: construction

This is the delicate step. U_q is not given, it is constructed, and the estimator for FI_in is sensitive to the construction choice. Three requirements:

**(R1)** Semantic equivalence: every x ∈ U_q must be semantically equivalent to the canonical x_0. Operationalize as bidirectional NLI entailment under a strong NLI model (DeBERTa-v3-large-MNLI, or GPT-4o-as-judge with the Razavi et al. 2025 ECIR "PromptSET" prompt template, arXiv:2502.06065).

**(R2)** Coverage: U_q should sample the practical paraphrase space a user might plausibly produce, not just adversarial rewrites. Generate from at least two generators (LLM-rewrite with GPT-4o, rule-based format/specificity variants a la Sclar et al. 2023 and Seleznyov et al. 2025) so the FI_in estimate is not tied to a single generator.

**(R3)** Independence: the generator should not be the model being evaluated, otherwise FI_in confounds prompt quality with prompt-match-to-model-distribution.

Procedure:

```
Input: canonical prompt x_0, ground-truth answer a_0, target |U_q|
1. Generate K_raw paraphrases (K_raw ≈ 50 to 100) via:
   a. LLM rewrite (GPT-4o, temperature 0.8), prompt template per Razavi et al. 2025
   b. rule-based format mutations (6 format axes per Sclar 2023)
   c. rule-based specificity mutations (remove/add constraint, per DETAIL / Kim 2025)
2. NLI-filter: retain x iff DeBERTa-v3-MNLI(x_0 ⇒ x) ≥ 0.9 AND (x ⇒ x_0) ≥ 0.9
3. Deduplicate by edit distance > 5.
4. Output U_q as the post-filter set.
```

Report |U_q| per query. The denominator of FI_in uses this |U_q|.

### 7.6.2 Function F(x)

- **Factoid Q&A (TriviaQA, NaturalQuestions):** F(x) = 1 iff the model's sampled answer matches the gold by exact-match or by NLI-with-gold. Take the majority of 5 samples to stabilize under sampling noise.
- **Multi-hop (HotpotQA):** same, with NLI-with-gold.
- **Multiple-choice (MMLU subset):** F(x) = 1 iff the selected option matches the gold.
- **Open-ended generation (a FLASK or PEEM scored subset):** F(x) ∈ [0, 1] from the rubric.

### 7.6.3 Estimator and uncertainty

```python
import numpy as np

def fi_in(scores, k):
    """Hazen–Szostak functional information for a threshold k."""
    N = len(scores)
    M_k = sum(1 for s in scores if s >= k)
    if M_k == 0:
        return float('inf')
    return -np.log2(M_k / N)

def fi_in_curve(scores, ks=np.linspace(0.0, 1.0, 21)):
    return {float(k): fi_in(scores, k) for k in ks}

def aufi_in(curve, N):
    """Area under FI_in(k) curve on k in [0, 1]. N = |U_q|."""
    ks = sorted(curve.keys())
    # clamp infinities at log2(N+1) before trapezoidal integration
    cap = np.log2(N + 1)
    vals = [min(curve[k], cap) for k in ks]
    return np.trapezoid(vals, ks)

def fi_in_bootstrap(scores, ks, B=1000):
    """Percentile bootstrap over paraphrase resamples."""
    N = len(scores)
    curves = []
    for _ in range(B):
        s = [scores[i] for i in np.random.choice(N, N, replace=True)]
        curves.append(fi_in_curve(s, ks))
    return curves  # analyze per-k CI afterward
```

Report FI_in(q, k) with 95% percentile-bootstrap CI from B = 1000 resamples of U_q.

### 7.6.4 FI_out implementation

```python
def fi_out(samples_by_prompt, cluster_fn, A_q_size):
    """
    samples_by_prompt: dict x -> list of sampled responses
    cluster_fn: list[str] -> list[int]  (Farquhar semantic clustering via NLI)
    A_q_size: estimated |A_q|, size of semantic answer space for this query
    """
    fi = {}
    for x, ys in samples_by_prompt.items():
        clusters = cluster_fn(ys)
        counts = np.bincount(clusters)
        probs = counts / counts.sum()
        H_sem = -sum(p * np.log2(p) for p in probs if p > 0)
        fi[x] = max(0.0, np.log2(A_q_size) - H_sem)
    return fi
```

|𝒜_q| is estimated by running the cluster function on the pooled samples from all prompts in U_q and counting unique cluster IDs.

### 7.6.5 Sample sizes

- Paraphrases per query: |U_q| ≥ 30 post-filter (Errica et al. 2025 uses Q = 30, we follow).
- Samples per prompt for FI_out: k = 10 (Farquhar et al. 2024 Nature uses 10).
- Queries per task: 100 for TriviaQA, 100 for HotpotQA, 100 for MMLU (subset).
- Models: at least three open-weight (Llama-3-8B, Llama-3-70B-Instruct, Mistral-7B-Instruct), one closed (GPT-4o).

Total model calls, per (task, model) cell: 100 queries × 30 paraphrases × (1 call for F + 10 calls for FI_out) = 33 000. Across four models and three tasks, ≈ 400 000 calls. This is a real budget and is the reason Polo/Nitsure's (NeurIPS 2024) IRT extrapolation should be pre-wired into the experimental harness from day one, even if not used in the pilot.

---

## 7.7 Connection to your supervisor's "embedding distance to most-specific input"

The supervisor brief flagged a separate but related intuition: there exists a "most specific" prompt x*(q), and Euclidean distance in embedding space ||e(x) − e(x*)|| should predict degradation in model performance. FI_in gives a sharp, testable formalization of this.

**Definition.** For a query q and model M, define the maximum-functional-information prompt as

x*(q; M) = arg max_{x ∈ U_q} F(x; M).

Ties broken by minimum length, then alphabetically. This is operational: every paraphrase universe has such an x* (by finite arg max). In Hazen et al. (2007, p. 29) this is `E_max`, the highest-function configuration.

**Hypothesis H1.** F(x; M) decreases in ||e_M(x) − e_M(x*)||, where e_M is the model's own last-layer hidden state at the final prompt token.

**Test.** For each (query, model) pair, plot (F(x; M), ||e_M(x) − e_M(x*)||) over all x ∈ U_q. Fit a monotone regression (isotonic or rank-linear). Report Spearman ρ. If H1 holds, ρ ≪ 0 with small CI across queries.

**Why this connects to FI_in.** If H1 holds, FI_in at threshold k corresponds to the fraction of U_q outside a ball of radius r(k) around x*, where r(k) is the critical distance above which F drops below k. The FI_in curve is then geometrically the distribution-function of embedding distances to x*. A low-FI_in model has a flat F-vs-distance curve: the embedding ball around x* is effectively the whole paraphrase universe. A high-FI_in model has a sharp drop: only prompts very close to x* succeed.

This is the novel geometric reading of prompt sensitivity. It ties the information-theoretic (Hazen/Szostak) and geometric (supervisor's intuition) views into one picture.

A related open question: is x*(q; M) close in embedding space to the x*(q; M') of a different model M'? If yes, there is a model-independent "most specific" prompt, and FI_in measures how well each model approximates that idealized target. If no, each model has its own x* and FI_in is irreducibly model-dependent. Either result is publishable.

---

## 7.8 Pitfalls and criticisms to address in the write-up

Three serious criticisms will hit the paper, and each has a specific mitigation.

**P1. U_q is generator-dependent.** FI_in = −log₂(N_k/|U_q|) depends on the paraphrase generator and the NLI filter. A critic will say: "you did not measure FI, you measured FI under your particular paraphrase distribution." Mitigation: report FI_in under two independent generators (GPT-4o rewrite and rule-based) and check that the ordering of prompts by F(x) is stable (Kendall τ between generators). If orderings are stable, the generator-dependence is a uniform shift, not a structural distortion.

**P2. FI_in conflates model-specific and query-specific fragility.** A query that is intrinsically hard has high FI_in regardless of phrasing. Mitigation: report FI_in relative to a baseline model (e.g., FI_in(q, k; M) − FI_in(q, k; M_base)), not absolute. The relative quantity isolates how much the tested model is more or less sensitive than the reference for this query.

**P3. |𝒜_q| estimation for FI_out is noisy.** Cluster-count estimators of semantic-answer-space size are notoriously under-biased at small sample sizes. Mitigation: use a rarefaction-style estimator (Chao 1987, not in folder, add as reference) that corrects for unseen clusters, and report FI_out with this correction.

---

## 7.9 Why this lands as a contribution

Three claims to land in the seminar paper, each with an experimental pin.

**C1. FI_in is a well-defined, computable, cross-model prompt sensitivity metric, distinct from existing indices.** Pin: empirical Spearman ρ between FI_in and each of {POSIX (Chatterjee et al. 2024), Errica S_τ (NAACL 2025), variation ratio (Lu 2023), performance spread (Sclar 2023)} on the pilot 100-query × 4-model grid. If FI_in has ρ < 0.8 with all of these, it is a distinct axis. If ρ > 0.9 with any, FI_in is a rescaling of that one and the claim weakens.

**C2. FI_in exhibits the stepped-function behavior predicted by Hazen et al. (2007) for prompt space.** Pin: visual inspection of FI_in(k) curves for at least 20 queries. Steps are the Hazen prediction; smooth decay is a null; mixed is informative.

**C3. Model sensitivity, as measured by AUFI_in, is predicted by embedding distance to the model's own x*(q; M).** Pin: cross-query Spearman ρ between F(x) and ||e_M(x) − e_M(x*)||, averaged over queries, ≤ −0.4.

None of these three pins require novel experimentation beyond the stack already specified in §6 of the synthesis. They are re-reads of the same data through the FI lens.

---

## 7.10 Concrete next steps, ordered

1. **Pin the paraphrase generator and NLI filter.** Implement the two-generator pipeline from §7.6.1. Validate on 10 queries by manual inspection: does every post-filter paraphrase actually preserve meaning? If not, tighten the NLI thresholds before scaling.

2. **Implement FI_in and FI_out in a single Python module.** Functions `fi_in`, `fi_in_curve`, `aufi_in`, `fi_in_bootstrap`, `fi_out` as sketched in §7.6.3 and §7.6.4. Unit tests against two hand-worked cases: a perfectly-uniform paraphrase family (FI_in = 0) and a single-magic-phrasing family (FI_in = log₂|U_q|).

3. **Pilot.** 20 TriviaQA queries × 30 paraphrases × 2 models (Llama-3-8B, GPT-4o). Compute FI_in(k) curves and AUFI_in. Compare against the POSIX and Errica S implementations.

4. **Validation experiments C1, C2, C3.** In order. C1 and C2 are computed from the pilot data with no new calls. C3 requires extracting embeddings, which is an additional pass on the open model only (GPT-4o embeddings via text-embedding-3-large for the closed condition).

5. **IRT extrapolation.** Fit the Polo/Nitsure (NeurIPS 2024) IRT model on the measured grid; report how far the FI_in estimates can be extended with partial coverage. This is the compute-budget-buster and should be prototyped in parallel with step 3.

6. **Write-up.** Structure: (i) FI review with Szostak/Hazen/Wong, (ii) FI_in definition, (iii) FI_out companion, (iv) empirical curves, (v) validation via C1/C2/C3, (vi) embedding-geometry reading, (vii) pitfalls P1/P2/P3, (viii) limitations and future work. Ordered as in §6 of your research synthesis.

---

## 7.11 References

Hazen, R. M., Griffin, P. L., Carothers, J. M., Szostak, J. W. 2007. "Functional Information and the Emergence of Biocomplexity." Chapter 2 in Avise & Ayala (eds.), *In the Light of Evolution, Vol. 1: Adaptation and Complex Design*, National Academies Press, pp. 25–43. Definition on pp. 28–30. Read from `Bookshelf_NBK254308.pdf` in the project folder. Also published in PNAS 104, 8574–8581.

Szostak, J. W. 2003. "Functional information." *Nature* 423, 689. Read from `Szostak_2003.pdf`. ATP-aptamer example on p. 689.

Corona, D., Di Benedetto, V., Gabriele, A., Giancarlo, R., Utro, F. 2010. "Functional Information, Biomolecular Messages and Complexity of BioSequences and Structures." Dagstuhl Seminar Proceedings 10231. Read from `Corona et al. ...pdf`. Protein-DNA result and distinctness-from-Kolmogorov on p. 2–6.

Wong, M. L., Cleland, C. E., Arend, D., Bartlett, S., Cleaves, H. J., Demarest, H., Prabhu, A., Lunine, J. I., Hazen, R. M. 2023. "On the roles of function and selection in evolving systems." *PNAS* 120(43), e2310223120. DOI 10.1073/pnas.2310223120. Generalization of FI beyond biology. Not in folder; confirmed via web search.

Chatterjee, A. et al. 2024. "POSIX: A Prompt Sensitivity Index For Large Language Models." arXiv:2410.02185. Formula verified against pp. 3–5.

Errica, F. et al. 2025. "What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering." NAACL 2025. Read pp. 1–6 directly.

Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. 2024. "Detecting hallucinations in large language models using semantic entropy." *Nature* 630, 625–630.

Razavi, A., Soltangheis, M., Arabzadeh, N., Salamat, S., Zihayat, M., Bagheri, E. 2025. "PromptSET: Answerability Prediction for Search Engine Tasks via Retrieval-Augmented Prompting." ECIR 2025. arXiv:2502.06065. Paraphrase-generation pipeline used in §7.6.1.

Sclar, M. et al. 2023. "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design." Format-mutation axes.

Polo, F. M., Nitsure, M. et al. 2024. "Efficient Multi-Prompt Evaluation of LLMs." NeurIPS 2024. IRT extrapolation pipeline.

Sun, Y., Wang, X., Cao, G., Mao, S. 2025. "Functional Data Analysis-Guided Prompt Design for RFID Sensing and Localization Using LLMs." IEEE GLOBECOM 2025. **This paper uses "Functional Data Analysis" (FDA), a statistics technique for smoothing time-series, and is NOT related to Szostak/Hazen functional information. Acronym collision only.** Flagged here so it is not accidentally cited as precedent.

---

## 7.12 What you can tell your supervisor in one paragraph

"Ich schlage vor, beide Perspektiven mathematisch in einem FI-Rahmen zu fassen. Die Input-Raum-Perspektive wird zu FI_in(q, k) = −log₂(N_k/|U_q|), der direkten Szostak-Hazen-Formel angewandt auf eine Paraphrase-Universe U_q. Die Output-Raum-Perspektive wird zu FI_out(x) = log₂|𝒜_q| − H_sem(Y|X=x), also KL-Divergenz von Uniform im semantischen Antwortraum. FI_in ist der primäre Beitrag (Literatur-Lücke, direkte Szostak-Übersetzung, braucht keine Logits), FI_out die Begleit-Metrik. Die „Distanz zum spezifischsten Input" aus deinem Brief wird bei uns zu: x*(q) = arg max F(x), und die empirische Frage ist, ob ||e(x) − e(x*)|| linear in F(x) fällt, wodurch FI_in eine geometrische Lesart in Embedding-Raum bekommt."
