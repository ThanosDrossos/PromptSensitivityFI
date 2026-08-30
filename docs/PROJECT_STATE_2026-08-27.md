# Project state — 2026-08-27 (seminar submitted, ICLR reframe next)

One-page orientation for anyone joining now. Conventions and the fixed
vocabulary: `CLAUDE.md`. Earlier history: `docs/PROJECT_STATE_2026-08-16.md`
(superseded by this file for anything about the current state).

## What the project claims

Prompt sensitivity is not one number. Placed on a single functional-information
ruler (bits = −log₂ surviving fraction), fourteen published candidate metrics
collapse onto **three measured axes**, and two **independently manipulated
variables** move them apart:

| | measured by | responds to |
|---|---|---|
| Competence | graded accuracy, FI_in(q,θ) curve | specificity (up 6–13 accuracy points) |
| Formulation sensitivity | **ρ_F** (noise-corrected ICC), σ²_B | width (up; significant in Qwen, directional in all three) |
| Output dispersion | **H_sem** | specificity (down) |
| *Specificity* — content variable | FI_spec = log₂(m₀/m_valid), annotations only | — manipulated |
| *Width* — form variable | generator configuration (narrow/production/wide, + swap) | — manipulated |

Plus: most published dispersion indices are summaries of **one clustering** of
the sampled answers, so their agreement is arithmetic rather than corroborating
evidence; and a linear probe on a single forward pass ranks underspecified
questions before any answer is generated.

## Timeline

- **Mar–Jul 2026** — metric-design eras, then the AmbigQA specificity pivot.
  See `docs/PROJECT_STATE_2026-08-16.md`.
- **2026-08-02/03** — final cluster run complete. No further runs planned.
- **2026-08-06 → 08-11** — adversarial review, R1–R9 work programme, KSRI talk,
  supervisor feedback and the follow-up audits.
- **2026-08-14** — second adversarial review (`docs/reviews/REVIEW_2026-08-14_Paper_ICLR.md`).
- **2026-08-16** — Windows→Mac migration finished; the whole 08-14 fix list
  implemented in both repos (new analyses, decontaminated probes, artifact-fed
  figures) and pushed.
- **2026-08-23/24** — editorial revision: abstract rewritten, figures
  redesigned (level panels, chance-anchored probe bars), fixed vocabulary
  introduced, citation audit (10 real mismatches corrected), reviewer panel
  incorporated.
- **2026-08-24** — the design is reframed as **two independently manipulated
  variables**, closing the old Methods/Results asymmetry where two
  interventions were reported but only one variable declared. Self-adjudicating
  prose removed.
- **2026-08-25/26** — clarity pass (jargon unpacked, section roadmaps, detail
  trimmed), FI_spec restated as the quantification of a manipulation rather
  than a fourth measurement. **Seminar paper submitted 2026-08-26.**
- **2026-08-27** — full numbers audit against the artifacts; 13 corrections;
  docs and branches consolidated onto `main`.

## Status by area

| area | status |
|---|---|
| Data collection | **Complete.** 3 models × 150 questions × 2 levels × up to 10 paraphrases × 10 samples = 900 cells, 89,730 scored responses; plus width/swap/POSIX/k=20/holdout arms. Full LLM cache is cluster-only. |
| Analysis code | On `main`. 399 tests, CPU-only except one gated NLI test. All figures and tables re-derive from committed artifacts. |
| Committed artifacts | 50 parquets + 18 `data/*.md` write-ups + `data/run_manifest.json` (seeds, thresholds, sample, hashes). Hidden states and holdout dumps (~450 MB) stay out of git. |
| Paper | `../SensitivityFunctionalInformationPaper`, branch `main`, submitted seminar build: 24 pages, content pp. 1–13, refs 15–19, appendix 20–24. Every number traced to an artifact. |
| Numbers audit | 2026-08-27: ~360 claims traced, each flag adversarially re-verified; 13 corrections landed (listed in `CLAUDE.md`). |
| Literature | Verified through 2026-08-23; `references.bib` carries per-entry verification comments. |
| Next | **ICLR 2027 reframe with additional contributors.** |

## Headline results (union gold is the primary endpoint)

- **Specificity → competence**: +0.064 / +0.125 / +0.119 (Qwen / Llama /
  Mistral), BH-significant 3/3, Holm-robust 2/3, pooled p = 3.3e-5. The naive
  target-gold numbers (+0.22…+0.25) are 47–72 % **grading lottery** and are
  reported only as the protocol comparison. Moderator: disambiguating toward a
  question's *first-listed* reading buys ~18 points in every model; later-listed
  readings buy nothing in Qwen and about half that in Llama and Mistral.
- **Specificity → dispersion**: H_sem falls −0.124 / −0.433 / −0.139
  (Holm-robust in Llama, pooled p = 6.1e-5).
- **Specificity → formulation sensitivity**: no response (p ≥ .41 union gold),
  with the estimator's resolution stated: shrinkage attenuates true changes
  3–5×, and the Rubin-pooled bracket is ±0.08.
- **Width → formulation sensitivity**: σ²_B and MoM ρ_F rise narrow→wide in all
  three models; individually significant only in Qwen. Competence does not
  respond (CIs within 4 accuracy points).
- **Generator swap**: per-cell structure survives (ρ_F Spearman +0.27…+0.51,
  σ²_B +0.64…+0.80); the model ordering Qwen > Mistral > Llama survives **on the
  share** (under the swap generator, σ²_B reorders to Mistral > Qwen > Llama).
- **Probes**: underspecification 0.85–0.87 AUROC in-distribution and
  **0.655–0.670 zero-shot** on 1,852 annotator-labeled holdout questions vs
  frozen text baselines 0.543–0.571; dispersion 0.69–0.77; fragility is a clean
  negative in all three models; correctness rank correlations 0.34–0.51.

## Reading order for someone new

1. `CLAUDE.md` — the frame, the fixed vocabulary, where truth lives, and the
   corrections that older documents still get wrong.
2. The paper in `../SensitivityFunctionalInformationPaper` (`main.tex` →
   `main_body.tex` → `appendix.tex`).
3. `data/stats_hygiene.md` — the declared test family and the primary numbers.
4. `FORKING_PATHS.md` — every analysis decision that moved a number.
5. `PIPELINE_WALKTHROUGH.md` — how the data was produced.
