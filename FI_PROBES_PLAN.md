# FI Probes — predicting prompt robustness from a single forward pass

**Status:** design note, 2026-07-18. Final-goal capstone of the seminar: transfer the
Semantic Entropy Probes methodology (Kossen et al. 2024, arXiv:2406.15927,
github.com/OATML/semantic-entropy-probes) from semantic entropy to Functional
Information. End-user framing: *"your prompt has low FI_in → the answer is
robust to rephrasing → likely correct."*

## 1. What SEP does (verified against paper + repo)

- **Target:** binarized semantic entropy H_SE of a query's sampled generations
  (N=10 samples, T=1.0, NLI-clustered, discrete SE). Threshold γ* chosen to
  minimize within-class variance (regression-tree splitting objective); the
  probe's predicted probability retains a fine-grained score.
- **Probe:** L2-regularized **logistic regression** (LBFGS) on frozen LLM
  hidden states. Two token positions: **TBG** (last token of the prompt,
  *before* generation) and SLT (second-last token of the response). Layers
  swept; best = late layers for short-form, mid layers for long-form;
  adjacent high-performing layers concatenated.
- **Training data:** 1k–2k queries; labels need NO ground truth (SE is
  sampling-derived) — the probe is "unsupervised" w.r.t. correctness.
- **Evaluation:** AUROC for hallucination detection vs. accuracy probes,
  log-likelihood, p(True). In-distribution ≈ accuracy probes; **out-of-
  distribution SEPs generalize better** (+7.7 AUROC on Llama-2-7B) because SE
  is model-internal. ~10× cheaper than sampling at inference (1 forward pass).
- **Pipeline (repo):** `generate_answers.py` (samples + hidden states) →
  `compute_uncertainties.py` → probe notebook (sklearn-style, wandb artifacts).

## 2. The transfer: FI probes

Same recipe, three targets of increasing novelty. All features come from ONE
forward pass of the prompt (TBG hidden state h^l(x)) — exactly the user-facing
setting (warn before generating).

| Probe | Target (label) | Novelty |
|---|---|---|
| **P1 (headline)** | binarized **AUFI_in of the prompt's cell** — does x sit in a formulation-robust equivalence class? Each of the N paraphrases inherits its cell's label → N training pairs per cell. | **New.** SEP predicts output dispersion of *one* prompt; P1 predicts sensitivity of the whole *paraphrase universe* from a single member. This is "prompt robustness from the inside." |
| P2 | **F_graded(x)** = P(correct \| x) from the k temperature samples (the user's "likelihood of a functional answer"). | Correctness/calibration probes exist; ours is the graded-F variant aligned with the FI stack. Strong baseline for P1. |
| P3 | binarized **H_sem(x)** per prompt. | SEP replicated verbatim in our setting — sanity check connecting to the literature + the FI_out side (FI_out_fixed = log2(m0) − H_sem is affine in it). |

Design decisions (mirroring SEP where possible):
- **Features:** last-token (TBG) hidden states at ~4 layers (25/50/75/100% of
  depth), float16, per eval model. Optional: mean-pooled variant (our existing
  `embed_hidden`) as ablation. SLT deferred (requires capturing generation
  states; TBG is the pre-generation use case).
- **Probe:** sklearn LogisticRegression (L2, lbfgs) per (layer, target);
  optionally concatenate best adjacent layers as in SEP. No deep nets.
- **Binarization:** SEP's γ* splitting objective on the label distribution;
  report AUROC + Spearman of the continuous probe score against the raw label.
- **Splits:** **by question_id** (never split paraphrases of one question
  across train/test — cell-level labels leak). Secondary: split by question
  *and* level; cross-model transfer as an extra table.
- **Baselines:** token log-likelihood of the greedy answer, prompt length,
  P2-as-baseline-for-P1, and (if available) POSIX.
- **Data volume:** full run = 150 q × 2 levels × N=10 paraphrases = 3,000
  labeled prompt states per model (SEP used 1k–2k) — sufficient for linear probes.

## 3. Sequencing (decided)

1. **Now, before the full run:** implement hidden-state capture so the run's
   forward passes are not wasted — extend `embed_hidden` with
   `pooling="last_token"` + `layers=[...]`, dump per-cell states to
   `data/hidden_states_<model>.parquet` behind a `--dump-hidden` flag.
   Smoke on `gpu_a100_short` (1 window).
2. **Prototype probes in parallel on existing data:** one short window dumps
   TBG states for the finished v2 dataset (1,000 prompts × 1 forward ≈
   minutes) → build the probe module + notebook locally against real labels
   while the full run queues. Probe training itself is laptop-scale (sklearn).
3. **Full run v3** (see §4) with `--dump-hidden` on → probe training set.
4. Train/evaluate probes; report P1 vs P2 vs baselines.

## 4. Full run v3 spec

- 150 questions (of 609 coverage-filtered), 2 levels, **all 3 eval models**,
  k=10, N=10, uniform evidence, graded-F track (commit 3147bb1), fresh
  `--out data/specificity_v3_metrics.parquet`, `--inspect-n 5`, `--dump-hidden`.
- POSIX: separate arm (qwen × 50 q) — N² echo passes are too heavy for the
  full grid and POSIX is a triangulation metric, not the headline.
- **Queue: `gpu_a100_short` singleton chains** (proven machinery, schedules in
  minutes, per-cell checkpointing makes 30-min windows safe). One prep-heavy
  qwen chain first (Phi-4 universes are model-independent, ~300 universes ≈
  8–12 h of windows), then llama + mistral chains in parallel off the shared
  cache. MI300 excluded (ROCm ≠ our CUDA stack); H100 queues offer no benefit
  we need (peak VRAM = Phi-4 28 GB < A100-40) at the cost of env risk;
  `*_il` 48-h queues only if a single uninterrupted job becomes necessary.
- Probe training: **no cluster needed** (laptop/CPU, sklearn).

## 5. Risks / scope guards

- Binary-F collapse made AUFI ≡ accuracy in v2 (ρ = −1.000): P1 on v2-style
  labels would reduce to an accuracy probe. The **graded track is therefore a
  prerequisite** for P1 being distinct — validate `aufi_in_graded` variance in
  the v3 smoke before the full run.
- Keep the probe stack minimal (sklearn, one notebook/script) — the seminar's
  novelty is the FI construct + P1, not probe engineering.
- OOD generalization (SEP's strongest claim) needs a second dataset — out of
  scope for the seminar run; note as future work (TriviaQA/NQ via the same
  specificity-free pipeline).
