# Runbook — R6 generator-width dial on bwUniCluster

The ρ_F **positive control** (its own dial) + the **paraphraser-swap ablation** (first in this
literature, per `LITERATURE_INTEGRATION_2026-08-07.md` §4.4). Decisions locked 2026-08-07:
**50 q × both levels** (the k20/POSIX subset) · swap generator **OLMo-2-13B-Instruct** · wide arm
**fluent rewrites only** (no typos — surface noise is a different perturbation family).

---

## 0. Preregistered predictions — fixed BEFORE any arm data exists

*(Also in `scripts/width_dial_analysis.py`'s docstring, which prints PASS/FAIL against exactly these.)*

| id | prediction | why |
|---|---|---|
| **P0** | Manipulation check gates everything: realized width of the *accepted* universes is ordered narrow < medium < wide (mean pairwise token edit distance). If the identical gates censored the ordering away, the dial is **inconclusive** — report that, do not interpret P1–P4 | the NLI@0.9 gate may compress the wide arm |
| **P1** | σ²_B (absolute phrasing variance) **increases** with width (one-sided narrow < wide, per model) | primary: more diverse phrasings ⇒ more phrasing-attributable variance |
| **P2** | ρ_F (hierarchical) **increases** with width | the share rises since decoding config (MS_W) is unchanged |
| **P3** | accuracy responds **less** than σ²_B/ρ_F (no sign prediction; report Δ + CI) | width is not an ability manipulation |
| **P4** | H_sem responds **less** than σ²_B/ρ_F | per-prompt dispersion shouldn't care how *other* prompts vary |
| **P5** | swap arm: per-cell ρ_F under OLMo-medium correlates positively with Phi-4-medium; model ordering (qwen > mistral > llama) preserved | generator-identity robustness of the construct |

**P3 + P4 are the payoff:** the specificity dial moves competence and not ρ_F; the width dial should
move ρ_F and not competence. Together: a **double dissociation** — the strongest construct-separation
evidence a measurement paper can present.

## The arms (gates identical everywhere; only G moves)

| arm | personas | temp | generator | judge | cache |
|---|---|---|---|---|---|
| narrow | 3 × minimal-edit | 0.5 | Phi-4 | Phi-4 | `paraphrases_width_narrow.parquet` |
| medium | the production 8 | 0.8 | Phi-4 | Phi-4 | `paraphrases_ambigqa.parquet` — **the existing v3 data, zero new compute** |
| wide | 8 × register/syntax/indirect (fluent only) | 1.0 | Phi-4 | Phi-4 | `paraphrases_width_wide.parquet` |
| swap | the production 8 | 0.8 | **OLMo-2-13B** | **OLMo-2-13B** | `paraphrases_width_swap.parquet` |

> **Design note (flagged for the write-up):** the swap arm swaps generator **and** gold judge
> together. This preserves the production rule *judge ≡ generator* and is forced by VRAM: OLMo (26 GB)
> + Phi-4 (28 GB) exceed the 40 GB A100 (the job-5762430 OOM pattern). The **primary semantic gate —
> DeBERTa NLI @0.9 — is identical across all four arms**, and per-arm gate-censoring stats are
> persisted (`*_reject_stats.parquet`) so any judge-strictness difference is visible, not silent.

---

## 1. Push

```bash
export BWUC_USER=<bwuc-username>
export BWUC_SSH_KEY=<path-to-ssh-key>
bash cluster/run.sh push
```

## 2. PREP — build the universes (run first, must finish before eval)

```bash
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" \
  "cd PromptSensitivityFI && bash cluster/submit_width_dial.sh prep"
```

Three singleton chains (`psf-wd-prep-{narrow,wide,swap}`), 4 × 30-min windows each, per-universe
resume. **First swap window downloads OLMo-2-13B (~28 GB) from HF** — expect that window to be mostly
download; it is cached afterwards (ungated repo, no token needed).

Done when each `cluster_logs/width_<arm>_prep.log` prints `PREP DONE universes=100 missing=0`.
Progress any time:

```bash
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" \
  "cd PromptSensitivityFI && bash cluster/submit_width_dial.sh check"
```

**Early-warning signs in the prep logs (act, don't wait):**
- narrow arm: many `edit_distance_close` rejections → minimal edits colliding under the ≥6 dedup; if
  universes stall below 10, the narrow instructions may need a 4th persona (tell me).
- wide arm: heavy `nli_low` rejections → the NLI gate censoring width (this is exactly what P0
  measures; high censoring + failed P0 = the gate, not the dial, is the finding).
- swap arm: `judge JSON parse failed` in bulk → OLMo not following the judge format; the judge prompt
  may need an OLMo-specific tweak (tell me before burning windows).

## 3. EVAL — 9 chains (3 arms × 3 models)

```bash
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" \
  "cd PromptSensitivityFI && bash cluster/submit_width_dial.sh eval"
```

3 windows per chain by default; per-cell resume; surplus windows no-op. 100 cells per (arm × model) at
~1.5–2.5 min/cell ≈ 2.5–4 h per chain → expect two evenings. Add windows by re-running the command.

## 4. Pull + analyse (laptop)

```bash
bash cluster/run.sh pull
```

```bash
uv run python -m prompt_sensitivity.scripts.width_dial_analysis
```

Output: `data/width_dial_analysis.md` — P0 manipulation check + gate-censoring table first, then
P1–P5 per model with the preregistered tests (one-sided Wilcoxon narrow<wide, Friedman,
per-cell monotonicity fraction, swap-arm agreement).

## 5. Reading the outcome (decided in advance)

| result | meaning for the paper |
|---|---|
| P0 ✓, P1–P2 ✓, P3–P4 ✓ | **ρ_F validated by its own dial** — the double dissociation goes in Results as the centerpiece of the construct-validity section; C2 gets its positive control |
| P0 ✓, P1–P2 ✗ | a real and important negative: phrasing width (within meaning-preserving G) does **not** drive phrasing-attributable variance — ρ_F reflects question×model structure, not universe construction. Honest finding either way; reframe C2's validity section around the convergence evidence instead |
| P0 ✗ | the NLI gate, not the generator, controls realized width — itself a publishable methodological point about *every* NLI-filtered paraphrase evaluation (report censoring table; dial inconclusive) |
| P5 ✓ | first paraphraser-swap ablation in the literature (per lit-integration) — a contribution bullet |
| P5 ✗ | ρ_F is generator-relative in *identity*, not just width — FI^G framing becomes load-bearing; must be disclosed prominently |

## Prep diagnostics observed 2026-08-07 (mid-run; record for the write-up)

At 58/11/38 universes (narrow/wide/swap), the gate-censoring sidecars showed:

| arm | fallbacks | dropped (<10 accepted) | NLI reject | dedup reject | relaxed NLI@0.85 used |
|---|---|---|---|---|---|
| narrow | 0 % | **72 %** | 6 % | **89 %** | **72 %** |
| wide | 0 % | 0 % | **64 %** | 24 % | 0 % |
| swap | 0 % | 0 % | 49 % | 35 % | 0 % |

Readings, all anticipated by the early-warning list:
1. **Narrow universes are undersized** — the ≥6-char dedup gate collides with minimal edits by
   construction, so most narrow universes fill below 10. Not fatal (0 fallbacks); handled in the
   analysis by the **N-matched robustness pass** (per-cell |U| equalised across arms by seeded
   subsampling) added to `width_dial_analysis.py` 2026-08-07. Disclose also that 72 % of narrow
   universes were built at the relaxed NLI@0.85 (identical *policy* across arms; the realized
   threshold adapts — conservative, since narrow got the looser gate and still came out narrower).
2. **Wide is healthy but expensive** (64 % NLI censoring ≈ 8 min/universe) — that censoring is the
   P0 quantity, not a defect. Cost fix: extend the prep chains on `gpu_a100_il` with 4-h windows
   (CLI overrides, no sbatch edit): `sbatch --partition=gpu_a100_il --time=04:00:00
   --job-name=psf-wd-prep-wide --export=ALL,ARM=wide,PHASE=prep cluster/width_dial.sbatch`.
3. **Swap arm fully healthy** — OLMo generates and judges without drama (judge rejects 2 %).
4. Sidecar caveat: `raw`/reject counts **double-count the relaxed-retry pass** (it re-evaluates all
   candidates); interpret rates within an arm, and prefer comparing arms that did not relax.
5. **Narrow |U| distribution measured mid-prep (58 universes, 2026-08-07): mean 6.2, median 6,
   p25 = 4, min 2, p75 = max = 10.** Pre-stated decision rule applied — median ≥ 5 ⇒ **ρ_F remains
   the primary outcome for the narrow arm** (σ²_B reported alongside as everywhere); the N-matched
   subsampling pass covers the unequal-|U| objection. Decision taken BEFORE any outcome data existed.

## Eval incident 2026-08-08 — SQLite lock storm (RESOLVED; record for provenance)

**Symptom:** eval windows dying early — `FAILED 2:0` with `sqlite3.OperationalError: database is
locked` on up to 93/100 cells per window, plus `FAILED 135:0` (SIGBUS, "Bus error (core dumped)")
concentrated on H100 nodes. 248/900 cells completed before diagnosis; none lost (parquet checkpoints).

**Root cause:** all nine eval chains shared `data/cache/llm_cache.sqlite` from up to nine nodes
concurrently over Lustre. SQLite WAL coordinates writers via an mmapped `-shm` file, which is
**unsafe across nodes on a network filesystem** — producing both the lock storm and the SIGBUS
(broken cross-node shm mapping). The v3 run survived with only 3 concurrent chains by luck/stagger.
**The H100 crashes were NOT an sm90/FlashAttention problem** — H100 nodes simply hosted the most
concurrent jobs when the storm peaked. Decision rule going forward: H100 windows are re-enabled; if
a 135 recurs *with per-chain DBs*, only then treat it as architecture-specific and drop H100.

**Fix (2026-08-08):** one cache DB per serial chain. `registry._get_cache` honours `PSF_CACHE_DB`;
`width_dial.sbatch` exports `data/cache/llm_cache_width_${ARM}_${MODEL}.sqlite` per eval chain
(per-arm for prep). Each singleton chain is serial ⇒ one writer per DB file, ever. Plus
`busy_timeout=60000` in `cache.py` as same-node mitigation. Cost: interrupted cells regenerate
mid-flight samples instead of hitting the shared cache — minutes, not hours.

**Operational trap:** Slurm snapshots the sbatch script at submission — jobs submitted before the
fix carry the OLD script and must be cancelled + resubmitted after the push.

## Cost summary

| phase | compute |
|---|---|
| prep narrow+wide | Phi-4 generation+judging, ~200 universes — a few windows |
| prep swap | OLMo download + generation+judging, 100 universes |
| eval | 9 chains × 100 cells (medium arm free — reuses v3) |
| analysis | laptop, minutes |
