# PromptSensitivityFI — Codebase Walkthrough

A step-by-step guide to how this repository works, written for someone who has
never seen it. Everything below is derived from the code itself, not the design
documents. Where a code comment contradicts the live config (it happens, the
project pivoted several times), the walkthrough follows what the code actually
executes and flags the divergence.

Package root: `prompt_sensitivity/`. Entry points: `prompt_sensitivity/scripts/`.
Config: `config.yaml` (loaded into a frozen Pydantic model by `config.py`).

---

## 1. What the code is trying to measure

The headline object the code computes is **FI_in**, "Functional Information in
prompt space." `metrics/fi_in.py` defines it as

```
FI_in(scores, k) = -log2( N_k / N )
```

where `N` is the number of paraphrases of one question and `N_k` is how many of
those paraphrases produce an answer scoring at least `k`. It is a
Szostak/Hazen functional-information quantity carried over from biopolymer
sequence space to the space of prompt phrasings. Intuition: if only a few
special phrasings of a question get the model to the right answer, the question
is "prompt sensitive" and FI_in is high (many bits); if almost any phrasing
works, FI_in is near zero.

The research question the code is built to answer (read off the experiment
drivers, `scripts/e2e_smoke.py` and the `cluster/*.sbatch` jobs):

> For multi-hop QA, how does a model's prompt sensitivity (FI_in over a
> paraphrase universe) change as you add information along two different axes,
> the amount of **retrieved context** and the amount of **reasoning scaffold**,
> and do the two axes behave differently?

This is the "dual ladder." One ladder feeds progressively more context
paragraphs; the other feeds progressively more steps of the gold reasoning
chain. The same questions and the same paraphrase universe run through both, so
their FI_in curves are directly comparable.

Supporting metrics sit alongside FI_in so the novel number can be cross-checked
against published prompt-sensitivity and uncertainty measures (Errica, Cox's
rho_u, POSIX, Farquhar's semantic entropy, performance spread, variation ratio,
ESS_in, FI_out). The full list and formulas are in Section 8.

---

## 2. The shape of one experiment

The unit of computation is a **cell**: one `(question, ladder_family,
ladder_type, level, model)` combination. For each cell the code:

1. Takes the question's paraphrase universe (N paraphrases, semantically
   equivalent rewrites of the question).
2. Builds N prompts by splicing the ladder content (context paragraphs or
   reasoning hops) for that `(ladder_type, level)` in front of each paraphrase.
3. Generates one answer per prompt at temperature 0, scores each answer to get
   `F(x) in [0,1]`. The vector of N scores is the input to FI_in.
4. Generates `k` extra answers per prompt at temperature 1.0, clusters them by
   meaning, and feeds the clusters to the output-space metrics (H_sem, FI_out,
   Errica, variation ratio, rho_u).
5. Optionally collects an N×N log-probability matrix for POSIX and embeds
   prompts/responses for ESS_in and rho_u.
6. Packs everything into one `MetricTuple` row and appends it to a parquet file.

A run sweeps many cells (questions × ladder types × levels × models) and writes
one parquet of `MetricTuple` rows. Analysis scripts then aggregate that parquet
into tables and plots.

---

## 3. Configuration and infrastructure

### 3.1 `config.py` and `config.yaml`

All hyperparameters live in `config.yaml` and load through `config.py` into a
frozen Pydantic `Config` (`model_config = frozen=True, extra="forbid"`), so no
call site can mutate them mid-run and an unknown key fails loudly.
`load_config()` is `lru_cache`d, so it reads the file once per process. You can
point it at another file with the `PROMPT_SENSITIVITY_CONFIG` env var.

Notable config groups: `sampling` (per-dataset question counts and HF dataset
ids), `ladders` (levels, k_gold, families), `paraphrases` (generator, NLI,
constraint, dedup), `models` (per-model capability flags), `scoring` (NLI
thresholds), `h_sem` (clustering knobs), `bootstrap`, `embedding`, `api`,
`cache`. `config_version` is bumped whenever a value changes so experiments can
be keyed on it.

### 3.2 Logging

`logging_setup.configure_logging(name)` wires loguru to write both to stderr and
to `logs/<name>.log`. Every script calls it once at startup with its own name.

### 3.3 The model cache (`models/cache.py`)

`LLMCache` is a thread-safe SQLite store (WAL mode) keyed by the SHA-256 of the
canonical JSON of a request. The rule it enforces: never call the same
`(prompt, model, sampling params)` tuple twice. Any change to the messages,
temperature, model id, seed, logprobs flag, echo flag, or the free-form
`purpose` string changes the hash and forces a fresh call. This is what makes
re-running an experiment cheap: only genuinely new calls cost compute. Both
chat requests (`LLMRequest`) and echo requests (`CompletionRequest`) share the
table via a `Cacheable` protocol.

### 3.4 Rate limiting and retries

`models/rate_limiter.py` is a classic token bucket, one per provider, used to
respect a QPS budget. `models/registry.py` wraps every provider call in a
`tenacity` exponential-backoff retry whose predicate (`_is_retryable`) treats
network errors, the OpenAI SDK's transient classes, and a list of gateway
hiccup message markers (502/503/504, "connection error", "overloaded", etc.) as
retryable, while letting genuine 4xx errors fail fast.

---

## 4. The models layer (`models/`)

### 4.1 The client interface

`registry.BaseLLMClient` is the abstract interface. Its `complete(LLMRequest)`
does cache-lookup, then rate-bucket, then retried call, then cache-store. Its
`score_continuation(CompletionRequest)` is the echo path used by POSIX (exact
teacher-forced token logprobs). Subclasses implement `_raw_call` (chat) and
`_raw_completion` (echo).

`get_client(model_key, config)` is the factory. It reads
`config.models[model_key]`, looks at the `provider` field, and returns a
singleton client. Two providers exist:

- `litellm` -> `LiteLLMClient`: the OpenAI Python SDK pointed at a LiteLLM
  gateway (`base_url` + `api_key` from env). Chat via `/v1/chat/completions`,
  echo via `/v1/completions`, plus `list_gateway_models()` for `GET /v1/models`.
- `local` -> `LocalHFClient`: Hugging Face `transformers` weights loaded
  in-process on the GPU. Registered lazily on first use so a gateway-only run
  never imports torch/CUDA.

### 4.2 The local backend (`models/local_hf.py`)

This is the backend the current experiments use. `_load_model` loads
`(tokenizer, model)` once per model id (bf16 on CUDA, eval mode, FlashAttention-2
with an SDPA fallback), cached so re-`get_client` is free. It exposes, all from
one process:

- `_raw_call`: applies the model's own chat template (`apply_chat_template`,
  with a fold-system-into-user fallback for templates that reject a standalone
  system role), generates, decodes, and optionally returns per-token logprobs
  with top-k from `output_scores`. Sampling is reproducible without touching the
  global RNG (`torch.random.fork_rng` + `manual_seed` only when sampling).
  `_apply_stop` truncates at stop strings.
- `_raw_completion`: one forward pass over a raw prompt returning
  `log P(t_i | t_<i)` for each token, the exact teacher-forced scoring POSIX
  needs (no "echo" hack).
- `embed_hidden`: mask-mean-pools the last hidden layer to a `(N, D)` matrix,
  then L2-normalizes each vector. This is the model's "own encoder," used by the
  own-encoder variants of ESS_in and rho_u and by the x* analysis. L2
  normalization matters: raw hidden-state norms differ 5-10x across
  architectures, so unit vectors put every model on one scale.

### 4.3 The configured models (`config.yaml` `models:`)

| key | model_id | provider | chat_logprobs | echo_completions | has_hidden | role |
|-----|----------|----------|:-:|:-:|:-:|------|
| `llama_3_1_8b` | meta-llama/Llama-3.1-8B-Instruct | local | yes | yes | yes | eval |
| `mistral_7b_v03` | mistralai/Mistral-7B-Instruct-v0.3 | local | yes | yes | yes | eval |
| `qwen_2_5_7b` | Qwen/Qwen2.5-7B-Instruct | local | yes | yes | yes | eval |
| `phi_4_14b` | microsoft/phi-4 | local | no | no | no | paraphrase generator + constraint judge |
| `gpt_4o` | kit.gpt-4.1 | litellm | yes | no | no | legacy gateway entry, unused on cluster |

The three capability flags are the contract the metric stack reads:
`chat_logprobs` (per-token logprobs available), `echo_completions` (POSIX
prerequisite), `has_hidden` (own last-layer hidden states reachable). The flags
are intentionally off for `phi_4_14b` because it is only a generator/judge, so
`scripts/local_check.py` will not fail-gate it on capabilities it does not need.

Worth knowing: several module docstrings (`registry.py`, `paraphrases/generate.py`,
`paraphrases/constraint_filter.py`, `paraphrases/__init__.py`) still describe the
generator/judge as "GPT-4.1" and the eval models as gateway-hosted. Those are
stale from the earlier gateway design. The code is config-driven: the generator
is `config.paraphrases.generator_model` and the judge is
`config.paraphrases.constraint_filter.judge_model`, both currently `phi_4_14b`,
and the eval models are `provider: local`. Trust the config, not those comments.

### 4.4 The embedding encoder (`models/embedding.py`)

`encode_texts` wraps a `sentence-transformers` model (default
`all-mpnet-base-v2`), lazy-loaded once per process. This is the "external
encoder" used for ESS_in and rho_u when the own-encoder path is not requested.

---

## 5. The data layer (`data/`)

### 5.1 The unified record (`data/schemas.py`)

Every dataset is parsed into one Pydantic model, `MultiHopQuestion`, with:

- `id`, `dataset` ("hotpotqa" | "2wikimultihopqa" | "musique"), `question`,
  `answer`, `question_type`, `level` (HotpotQA only).
- `paragraphs`: a list of `HotpotParagraph(title, sentences, is_gold)`.
  `joined()` concatenates the sentences back into the original text.
- `supporting_facts`: `(title, sent_id)` entries (HotpotQA / 2Wiki only).
- `question_decomposition`: a list of `DecompositionHop(hop_idx, sub_question,
  sub_answer, supporting_paragraph_idx)`. MuSiQue only. Its presence is what
  enables graded chain scoring.
- `n_hops`: MuSiQue hop count.

Helpers: `gold_paragraphs()`, `distractor_paragraphs()`, and
`has_decomposition()` (true iff chain scoring is available). A model validator
keeps the per-paragraph `is_gold` flags in sync with `supporting_facts` for
HotpotQA/2Wiki, while leaving MuSiQue's loader-set flags untouched.

### 5.2 The loaders

- `load_hotpotqa.py`: parses the HF `hotpotqa/hotpot_qa` distractor config. Each
  record ships 10 paragraphs; `is_gold` is derived from `supporting_facts.title`.
- `load_2wiki.py`: a near-clone for `framolfese/2WikiMultihopQA`, which uses the
  exact HotpotQA field layout. The difference is four `type` values and no
  `level`.
- `load_musique.py`: loads MuSiQue-Answerable. Tries a local jsonl first
  (`data/raw/musique/musique_ans_v1.0_dev.jsonl` and variants), then a HF
  mirror. Maps `paragraph_text` (tolerating `text`) to paragraphs with
  `is_gold = is_supporting`, and `question_decomposition` to `DecompositionHop`s.
  Paragraph counts vary (~20), nothing hardcodes 10. `question_type` is the
  clamped hop-count label `2hop`/`3hop`/`4hop`.

### 5.3 Stratified sampling (`data/sample_questions.py`)

`stratified_sample` drops questions with fewer than `k_gold` gold paragraphs,
groups the rest by the stratify key (`level` for HotpotQA, `type` for 2Wiki),
allocates an equal share per stratum with the remainder going to the largest
strata, samples without replacement under `config.random_seed`, tops up any
shortfall from the leftover pool, and shuffles. `scripts/sample_questions.py`
runs this and writes `data/sample_v1.json` (100 HotpotQA + 50 2Wiki ids).
MuSiQue questions are sampled directly in `e2e_smoke` instead (Section 7.2),
not through `sample_v1.json`.

---

## 6. The paraphrase pipeline (`paraphrases/`)

The goal: for each question, produce a set of N (target 30) rewrites that mean
the same thing and have the same correct answer, but vary in surface form. This
set is the universe over which FI_in is measured, so its quality is critical: if
a "paraphrase" secretly changes the answer, FI_in is contaminated.

### 6.1 Generation (`prompts.py`, `generate.py`)

`prompts.py` defines a persona-conditioned rewrite, following the PromptSET
pattern. One fixed system prompt instructs the model to rewrite a single
question preserving the answer set bit-for-bit, output one line, no preamble. A
per-persona instruction varies the register. Eight personas exist: `neutral`,
`journalist`, `casual_user`, `domain_expert`, `student`, `terse_keyword`,
`formal_academic`, `second_language`. The active set is
`config.paraphrases.templates`.

`generate.generate_raw_paraphrases` builds the `(system, user)` messages per
persona and fires `n=1` requests at `generator_temperature` (0.8) with a
deterministic per-`(question, role, sample_idx)` seed, so every candidate is
independently cacheable. `_clean_one_line` strips quotes and keeps the first
line. Each surviving candidate becomes a `RawParaphrase`.

### 6.2 The four filters

A candidate must pass all four to be accepted. Schemas
(`paraphrases/schemas.py`): `RawParaphrase` -> filters -> `AcceptedParaphrase`
or `RejectedParaphrase` (the rejection reason is recorded for audit). The
`RejectionReason` literal enumerates `nli_low`, `nli_one_direction`,
`constraint_mismatch`, `edit_distance_close`, `exact_duplicate`, `empty`.

1. **Identity drop**: a candidate equal to the original (case-folded) is rejected
   as `exact_duplicate`.

2. **NLI bidirectional entailment** (`nli_filter.py`): uses
   `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. For each candidate
   it runs the model in both directions, premise=original/hypothesis=candidate
   and the reverse, takes the softmax probability of the `entailment` class each
   way, and accepts iff both directions are `>= threshold`. Default threshold is
   `0.9` (`bidirectional_threshold`); a relaxed `0.85` fallback exists. A
   one-direction pass is labelled `nli_one_direction`, otherwise `nli_low`. The
   DeBERTa weights are loaded once per process via `lru_cache` and reused by the
   scorer and the H_sem clusterer.

3. **Constraint (answer-set) filter** (`constraint_filter.py`), two modes:
   - **Gold-based** (preferred, used whenever a gold answer is available): one
     yes/no judge call per candidate, "given the original question, its known
     correct answer, and this rewrite, is the known answer still a valid answer
     to the rewrite?" The judge is told to be generous about wording shifts and
     strict only about a genuinely different fact. Parse failures conservatively
     count as reject.
   - **Jaccard fallback** (no gold answer): the judge lists the answer set for
     the original and for the candidate separately; accept iff Jaccard `>= 0.9`.
     This path is fragile (two judge calls drift in surface form) and is only a
     fallback. `Jaccard(empty, empty) = 1.0` by convention.

4. **Deduplication** (`deduplicate.py`): a greedy Levenshtein dedup. Walk the
   accepted candidates in priority order, keep one, reject any later candidate
   whose distance to an already-kept candidate is below `min_edit_distance`.
   Two metrics: `char` (character Levenshtein, default, min distance 6, matching
   the brief's "edit distance > 5") and `token` (word-level, recommended min 3
   for short questions). The implementation is an in-house two-row DP. Drops are
   sub-classified into `exact_duplicate` (distance 0, a generator-diversity
   problem) vs `edit_distance_close` (small but nonzero, a threshold problem).

### 6.3 The orchestrator loop (`pipeline.build_paraphrase_set`)

Per question, it loops: generate the next batch of `samples_per_template`
candidates per persona, drop identicals, NLI filter, constraint filter, then
dedup the accumulated accepted pool (sorted by priority: higher of the two NLI
scores first, then higher Jaccard). If the pool reaches the target (30), stop
and materialise the first 30 as indexed `AcceptedParaphrase`s. If not, generate
another batch. Generation is capped by `max_regeneration_attempts`, interpreted
as the maximum raw candidates per template (`next_sample_start`). When that cap
is hit and the pool is still short, it relaxes the NLI threshold to the fallback
(0.85) and re-evaluates everything already generated once (cheap, all cache
hits) via `_retry_with_relaxed`. If still short, the question is marked
`dropped=True` and the caller is expected to replace it.

### 6.4 The Sprint-2 driver and gate scripts

- `scripts/generate_paraphrases.py`: walks `sample_v1.json`, runs the pipeline
  per question passing the dataset's gold answer, and writes a flat parquet
  (`data/paraphrases_v1.parquet`) with one row per accepted/rejected paraphrase.
  `--resume` skips question ids already present.
- `scripts/export_annotation_sample.py`: exports 20 questions × paraphrases to a
  CSV with blank `thanos`/`diener` columns for human review.
- `scripts/compute_kappa.py`: computes Cohen's kappa between the two annotator
  columns by hand; the gate passes iff kappa `>= 0.8`, else it tells you to
  tighten the NLI threshold and regenerate.
- `scripts/diagnose_paraphrases.py`: reports the rejection breakdown, NLI score
  distributions, and a threshold-counterfactual table to help pick a threshold.

---

## 7. The dual ladder (`ladders/`)

A "ladder" is a sequence of rungs (levels) that feed progressively more
information into the prompt. Two families exist, separated by the
`ladder_family` field on `LadderRow`.

### 7.1 The row schema (`ladders/schemas.py`)

`LadderRow` records `question_id`, `ladder_type`
(`random`/`gold_first`/`distractor_first`/`reasoning`), `level_idx`, `level`,
`paragraph_indices` + `paragraph_titles` (the subset spliced in), `gold_count`,
`permutation` (random only), `ladder_family` (`context`/`reasoning`), and
`hops_provided` (reasoning only).

### 7.2 The context family (three ladders)

All three order the question's paragraph pool and take prefixes at each level
(`config.ladders.levels = [0, 2, 4, 6, 8, 10]`). The difference is the ordering,
which controls when the gold paragraphs enter:

- **random** (`random_ladder.py`): one deterministic shuffle per question, seed
  `42 + sha256(question_id)` (a stable hash, because Python's built-in `hash` is
  salted per process). This is the headline ladder; gold enters at a random
  rung.
- **gold_first** (`gold_first_ladder.py`): gold paragraphs first, then
  distractors in dataset order. Best case: at level 2 you already have both gold
  paragraphs.
- **distractor_first** (`distractor_first_ladder.py`): distractors first, then
  gold. Worst case: gold only enters at the top level.

These three bound the effect of context composition: at any given level,
`gold_first` should score highest, `distractor_first` lowest, `random` in
between. At the top level all three contain the identical paragraph set (only
the order differs), which `build_ladders.py` asserts as a sanity invariant.

### 7.3 The reasoning family (`reasoning_ladder.py`)

A different manipulation, MuSiQue only. Instead of context paragraphs it feeds
the gold reasoning chain progressively. At rung `k` the model receives the first
`k` decomposition hops, each rendered as `sub-question -> sub-answer`, as
scaffolding. The final hop is always withheld (max scaffold is `n_hops - 1`), so
the model is never handed the answer. A 4-hop question yields rungs with
`{0, 1, 2, 3}` hops provided. `render_reasoning_scaffold` produces the
`- sub_question -> sub_answer` lines that get injected. The builder sets
`paragraph_indices = []` (it feeds hops, not paragraphs) and records `gold_count`
as the number of gold paragraphs associated with the provided hops, for
analytics only.

### 7.4 Bit-cost estimators (`bit_cost.py`)

Theoretical companions to the empirical curves:

- `b_theo(N, K, l) = -log2( 1 - C(N-K, l) / C(N, l) )`: the hypergeometric
  bit-cost of "a random size-`l` subset of `N` paragraphs contains at least one
  of `K` gold." Validated against `scipy.stats.hypergeom`. Used for the binary
  random-ladder model.
- `expected_hop_coverage(N, n_hops, l) = l / N`: the v6 graded analogue. With
  one supporting paragraph per hop, the expected fraction of hops whose
  paragraph lands in a random size-`l` subset is `l/N`, a straight line from 0
  to 1. This replaces `b_theo` on the graded-scoring path.
- `b_emp(u_above, u_below) = log2( |U_above| / |U_below| )`: an empirical
  bit-cost diagnostic from observed success-set sizes at consecutive rungs.

### 7.5 The standalone driver (`scripts/build_ladders.py`)

For the HotpotQA/2Wiki sample it builds all three context ladders × six levels
per question (~2700 rows), drops questions with fewer than `k_gold` gold or with
a paragraph count other than 10 (so `b_theo` stays valid), writes
`data/ladders.parquet`, prints the `b_theo` table, and asserts the sanity
invariants (level 0 empty, top level full, same multiset across ladders, gold
counts ordered gold_first >= random >= distractor_first). Note: the end-to-end
driver does not read this parquet, it rebuilds ladders on demand per question
(Section 9). `build_ladders.py` is the Sprint-3 gate artifact.

---

## 8. Scoring F(x) (`scoring/`)

`F(x)` is the per-prompt quality score that feeds FI_in. There are two scoring
paths; `e2e_smoke` picks one per question based on `question.has_decomposition()`.

### 8.1 Binary NLI-with-gold (`nli_with_gold.py`)

Used for HotpotQA / 2Wiki, and kept as a secondary score for MuSiQue. Never
exact match (the project cites Hua 2025 that exact match inflates measured
sensitivity). The method:

- Run DeBERTa NLI asymmetrically with premise = gold answer, hypothesis = model
  answer.
- `F = 1` iff `entail_prob >= entail_threshold` (0.7) AND
  `contradict_prob < contradict_threshold` (0.5), else 0.

`f_score_batch` scores a whole cell at once. A `permissive=True` flag
re-thresholds the same NLI pass at `entail_threshold_permissive` (0.5) for the
secondary final-answer column, so a too-strict NLI cannot masquerade as model
failure. `exact_match_score` exists but is appendix-only.

### 8.2 Graded chain-completion (`chain_score.py`)

The primary MuSiQue path, and the reason MuSiQue was adopted. The binary
final-answer score on HotpotQA collapses FI_in to a step function (the answer is
one short span, F is 0 or 1 with nothing between). Chain scoring instead asks
what fraction of the gold reasoning hops the model recovered:

```
F = recovered_hops / H        in {0, 1/H, 2/H, ..., 1}
```

This gives a graded F so FI_in integrates a smooth curve. Mechanics:

- `resolve_placeholders` substitutes MuSiQue's `#k` references in a sub-question
  with the earlier hop's gold sub-answer, making each hop self-contained.
- `build_fact_statements` renders each hop as `"{resolved_sub_question}
  {sub_answer}"` (not the bare answer, which would be too weak an NLI
  hypothesis).
- A hop is "recovered" by bidirectional NLI: score response-entails-fact and
  fact-entails-response, take the higher-entailment direction, and require it to
  pass the same entail/contradict thresholds as the binary path. The same
  DeBERTa model is reused (no second model loaded).
- `_scaffold_recovered`: on the reasoning ladder, hops the scaffold itself
  already states are OR-credited, so a terse model that omits already-given hops
  is not under-counted.
- `chain_completion_score_batch` amortises the NLI forward passes across a
  cell's responses.

The metric stack downstream consumes this float exactly as it consumed the
binary 0/1, so nothing under `metrics/` changes between the two paths.

### 8.3 Prompt templates (`prompts/templates/qa_prompt.py`)

One template per mode, model-agnostic (the same text for all models, the
tokenizer-specific chat template is applied by the backend). Variables are only
the question and the context block.

- **Baseline** (`assemble_qa_messages`): "use only the context, answer in a
  brief phrase, no reasoning, say `unknown` if not answerable." Used for the
  binary path.
- **Chain-of-thought** (`assemble_qa_cot_messages`): "reason step by step, state
  each intermediate conclusion, end with a line starting `Answer:`." Used for
  MuSiQue so the chain scorer can see intermediate answers.

At level 0 (no paragraphs) the context block is omitted, a true closed-book
condition. `parse_answer_line` extracts the final answer from a CoT response
tolerantly: it accepts `Answer:`, `**Answer:**`, `Final Answer:`, `### Answer`,
reads the value off the next line if the label line is bare, falls back to a
"the answer is X" prose match, then to the last non-empty line, and never
returns the whole essay (which scored 0 under the asymmetric NLI). This parsed
line feeds the secondary binary score and (by config) the H_sem clustering.

---

## 9. The metrics (`metrics/`)

Every metric module is a pure function over precomputed inputs. The
`orchestrator.build_metric_tuple` glues them into one `MetricTuple`. The
collection of responses, embeddings, and logprobs happens in `e2e_smoke`
(Section 10), not here.

### 9.1 FI_in (`fi_in.py`) — the primary contribution

- `fi_in(scores, k) = -log2(N_k / N)`, `N_k = #{ s in scores : s >= k }`.
  Returns 0 bits when every paraphrase reaches `k`, `log2(N)` when only one
  does, and `+inf` when none do.
- `fi_in_curve(scores, ks)`: FI_in over a grid of thresholds. Default `ks` is a
  21-point linspace on `[0, 1]`.
- `aufi_in(curve, n)`: area under the FI_in(k) curve by the trapezoid rule, with
  `+inf` clamped to `log2(n + 1)` so an unreachable threshold does not blow up
  the integral. This single scalar `AUFI_in` is the headline summary, in bits.
- `fi_in_bootstrap`: a percentile bootstrap CI per `k`, resampling paraphrases
  `n_iterations` times (1000 by config), clamping infinities to `log2(n+1)`.

Reading: higher FI_in / AUFI_in means more prompt sensitivity (only special
phrasings recover the answer). It should fall as you climb a ladder, the mirror
image of rising accuracy.

### 9.2 H_sem (`h_sem.py`) — semantic entropy (Farquhar 2024)

Clusters a prompt's `k` sampled responses by meaning, then takes the Shannon
entropy of the cluster proportions:

- `cluster_responses`: builds all unordered pairs over the unique response
  strings, runs DeBERTa NLI both ways per pair, merges two responses (union-find)
  iff both directions count as entailment. The merge rule is
  `config.h_sem.cluster_criterion`: `label` (argmax of the 3 NLI classes is
  entailment, strict, the default and Farquhar's rule) or `prob` (entail prob
  `>= cluster_threshold`, lenient, legacy). Exact-duplicate strings are collapsed
  before the O(u^2) NLI pass.
- `entropy_from_assignment`: `H = -sum_c p_c log2 p_c` over cluster proportions.
- `cluster_responses_pooled`: the API the driver must use. It pools all
  responses across paraphrases in a cell, clusters once, and slices back, so
  cluster id `c` means the same cluster across every paraphrase. This is a hard
  contract: FI_out, Errica, variation ratio, and `|A_q|` all require comparable
  ids, and independent per-paraphrase clusterings would corrupt them.

By config the clusterer runs on the parsed answer line of each sample
(`cluster_on = "answer"`), not the full generation, so style and verbosity do
not over-merge distinct answers.

### 9.3 FI_out (`fi_out.py`) — output-space restrictiveness

`FI_out(x) = max(0, log2|A_q| - H_sem(Y|X=x))`. It measures how much one prompt
narrows the answer distribution relative to the whole answer space `A_q`.
`estimate_a_q` sets `|A_q|` to the number of distinct pooled clusters observed
across paraphrases (floor 1). Note: the schema comment and an old code comment
mention a Chao-1987 unseen-species correction, but `estimate_a_q` explicitly
dropped it (a 2026-06-29 change) because with many singleton clusters it
over-estimated wildly; the live code uses the directly observed count. The
identity check `FI_out + H_sem = log2|A_q|` holds before the floor-at-0.

### 9.4 Errica two-number axis (`errica.py`)

- `s_tau_freeform(assignment, a_q) = H_sem(Y|X=x) / log2|A_q|` in `[0, 1]`,
  per-prompt sensitivity normalized by answer-space size (0 when `|A_q| <= 1`).
  A multiple-choice variant `s_tau_multiple_choice` normalizes by `log2(C)`.
- `tvd_consistency`: mean of `1 - TVD` over all pairs of paraphrases, where TVD
  is the total variation distance between two paraphrases' cluster-proportion
  distributions on the shared support. 1.0 means perfectly consistent across
  rewrites. Both require pooled cluster ids.

### 9.5 POSIX (`posix.py`) — Chatterjee 2024 prompt sensitivity index

```
psi = (1 / (N(N-1))) * sum_i sum_{j!=i} |log P(y_j|x_i) - log P(y_j|x_j)| / L_{y_j}
```

Given an N×N matrix `log_p[i,j] = log P(y_j | x_i)` and the token lengths
`L_{y_j}`, it measures how much the log-probability of paraphrase `j`'s natural
response shifts when scored under a different paraphrase's prompt, length
normalized and averaged over off-diagonal pairs. `log_p_from_token_logprobs`
sums the last `L_{y_j}` echo logprobs to recover `log P(y|x)`. POSIX needs
`echo_completions`, so it is only computed for the echo-capable models.

### 9.6 ESS_in (`ess_in.py`) — input-embedding dispersion

`ESS_in = trace(Cov)` of the `(N, D)` prompt-embedding matrix, computed as the
sum of per-feature sample variances (ddof=1). It is the geometric spread of the
paraphrase prompts in embedding space. Known limitation documented in the code:
with the external mpnet encoder (trained to map paraphrases close together) and
a shared long context block, the variance collapses toward 0, so external ESS_in
mostly tracks context length, not paraphrase diversity. The own-encoder variant
does not have this property.

### 9.7 rho_u (`rho_u.py`) — Cox 2025 epistemic-variance ratio

A law-of-total-variance decomposition of the response embeddings (N paraphrases
× k samples each):

- `U_t = tr(Cov_total)` total uncertainty,
- `U_a = tr(Cov_within)` aleatoric (within-paraphrase sampling noise),
- `U_e = tr(Cov_across)` epistemic (across-paraphrase mean variability),
- `rho_u = U_e / (U_t + epsilon)`, clamped to `[0, 1]`.

`rho_u = 0` means the prompt phrasing does not move the response distribution
(robust); `rho_u = 1` means phrasing fully determines the output (sensitive).
Variances are population (ddof=0) and sample-count weighted so within + across
recover the total.

### 9.8 Spread and variation ratio

- `spread(scores) = max(F) - min(F)` over paraphrases (`spread.py`).
- `variation_ratio(items) = 1 - mode_count / N` (`variation_ratio.py`). In the
  orchestrator it runs over the modal cluster of each paraphrase, so it measures
  disagreement of the dominant answer across paraphrases.

### 9.9 Hazen step test and x* analysis

- `hazen_test.py`: classifies FI_in(k) curve transitions into flat vs step using
  the bootstrap CIs (a step needs non-overlapping CIs and a jump above
  `jump_bits`), counts plateaus, and decides whether a cell fits Hazen's
  "islands of function" stepped pattern. It reads curve columns off the
  `MetricTuple`, it does not recompute FI_in.
- `analysis/x_star.py`: tests the supervisor's geometric conjecture. Per cell it
  picks `x*` (the max-F paraphrase, deterministic tie-break), measures each other
  paraphrase's L2 distance to `x*` on the own-encoder embedding, and computes the
  Spearman correlation between F and distance. The C3 target is mean rho `<= -0.4`
  (farther means less functional). Its `main()` reconstructs F and embeddings
  from the cache and so must run where the cache and weights live (the cluster).

### 9.10 The output record (`metrics/schemas.py`)

`MetricTuple` is a frozen Pydantic model holding the per-cell results:
`f_mean`, `aufi_in`, the full FI_in curve (`fi_in_curve_ks`/`_vals`) and its CI
(`fi_in_ci_lower`/`_upper`), `fi_out_mean`/`_var`, `s_tau_mean`,
`consistency_mean`, `spread`, `variation_ratio`, `posix_psi`, `ess_in`, `rho_u`,
`h_sem_mean`/`_var`, `a_q`, plus diagnostics (`n_paraphrases`,
`n_samples_per_prompt`, `encoder_label`). Any metric whose inputs are missing
(for example POSIX on a non-echo model) is `None`; the convention is to record
the limitation, never impute.

One subtlety to know: `MetricTuple.ladder_type` is a `Literal` of only
`random`/`gold_first`/`distractor_first`. Reasoning-ladder cells are stored with
`ladder_type="random"` in the tuple, and the true family/type live in the extra
columns `ladder_family` and `ladder_type_raw` that `e2e_smoke` attaches after
`model_dump()` (Section 10.4).

### 9.11 The orchestrator (`metrics/orchestrator.py`)

`build_metric_tuple` takes the cell's `scores`, pooled `cluster_assignments`,
`prompt_embeddings`, per-paraphrase `response_embeddings`, optional POSIX matrix,
and computes every metric it can. It derives `|A_q|`, H_sem mean/var, FI_out
mean/var, S_tau mean, TVD consistency, spread, variation ratio, the FI_in curve
(clamped) + AUFI_in + bootstrap CI, mean F, ESS_in, rho_u, and POSIX (if a matrix
was supplied). It returns the assembled `MetricTuple`.

`scripts/smoke_metrics.py` is the no-network gate for this layer: it builds a
hand-made 5-paraphrase cell and asserts every scalar lands in a plausible range.

---

## 10. The end-to-end pipeline (`scripts/e2e_smoke.py`)

This is the central driver that wires everything together. Despite the name it
is the real experiment runner; the `pilot*` and `cluster` jobs are just
parameterized invocations of it.

### 10.1 Inputs and question selection

Two ways to get questions and their paraphrases:

- **From a paraphrase parquet** (default): read accepted paraphrases per
  question from `data/paraphrases_smoke.parquet` or `paraphrases_v1.parquet`,
  take the first `--n-questions`, and look each id up in HotpotQA/2Wiki/MuSiQue
  via `_index_questions`.
- **MuSiQue-direct** (`--musique-direct N` or stratified `--musique-strata N`):
  sample N MuSiQue questions straight from the loader (highest hop count first,
  or N per hop stratum with a seeded shuffle so every model run shares the same
  set), and generate their paraphrase universe live via the Sprint-2 pipeline.
  `_generate_musique_paraphrases` persists accepted paraphrases to
  `data/paraphrases_musique.parquet` and reuses them on later runs (this is the
  expensive generation step, so it is cached and can be run alone with
  `--paraphrase-only`). `--singleton` skips generation and uses the original
  question as a one-element universe for a fast first look.

### 10.2 Building the cells

`_build_cell_rows(question, families, ladders, levels)` produces the ladder rows
to evaluate: for the context family it builds each requested context ladder and
keeps the rows whose level is in `--levels`; for the reasoning family (only if
the question has a decomposition) it builds the reasoning ladder. The driver
rebuilds ladders on demand here rather than reading `data/ladders.parquet`.

The plan (cell count and an LLM-call estimate) is logged, and `--dry-run` prints
it and exits before any model call.

### 10.3 Running one cell (`_run_cell`)

For each `(question, row, model)`:

1. `use_cot = question.has_decomposition()` decides chain (MuSiQue) vs binary
   (HotpotQA/2Wiki) scoring, and which prompt template is used. `_assemble_messages`
   builds the chat messages: context family splices the row's paragraphs;
   reasoning family injects the hop scaffold under a "Known intermediate steps so
   far" header.
2. **F(x)**: one response per paraphrase at temperature 0 (seed 42), cached under
   a per-cell `purpose` string. Chain path -> `chain_completion_score_batch`
   (with scaffold OR-credit on the reasoning ladder) plus a secondary
   `final_answer_f_mean` (strict and permissive) from the parsed `Answer:` line.
   Binary path -> `f_score_batch`.
3. **H_sem samples** (skipped when `--fast`): `k` samples per paraphrase at
   `h_sem.sampling_temperature` (1.0), seeded distinctly. The samples (or their
   parsed answer lines, per `cluster_on`) are pooled-clustered with
   `cluster_responses_pooled`.
4. **Embeddings**: prompts and responses are embedded, either with the external
   mpnet encoder or, under `--own-encoder` for `has_hidden` models, with the
   model's own `embed_hidden`. `encoder_label` records which.
5. **POSIX** (only with `--include-posix`, echo-capable model, context family):
   `_posix_matrix` builds the N×N log-prob matrix via `score_continuation`;
   any NaN voids the matrix and POSIX stays `None`.
6. `build_metric_tuple` produces the `MetricTuple`.

The `k < 5` case logs a resolution warning (semantic entropy can barely
distinguish 0 from 1 bit at tiny k). Default `k` is
`config.h_sem.n_samples_per_prompt` (10).

### 10.4 Output, resume, and audit sidecars

The tuple is dumped to a dict, then v6 columns are appended after the dump (so
`metrics/` stays untouched): `dataset`, `ladder_family`, `ladder_type_raw`,
`scoring_mode`, `final_answer_f_mean`, `final_answer_f_mean_permissive`,
`n_hops`. Rows accumulate into the `--out` parquet, checkpointed after every cell
via an atomic temp-file replace, so a crash never loses progress. On restart,
already-computed cells (keyed by `(qid, family, ladder_type, level, model)`) are
skipped. Cell failures are isolated, logged, and the run continues.

Two optional audit artifacts: a per-sample sidecar
`hsem_samples_<out>.parquet` (raw text, extracted answer, cluster id per sample,
so a clustering collapse is inspectable) and a start-to-finish markdown
inspection bundle for the first `--inspect-n` questions
(`inspect_<out>.md`: question, gold chain, paraphrase universe with roles/NLI,
prompt, F-responses, scores, the H_sem answer-to-cluster map, and metrics).

### 10.5 The built-in result summary (`_print_summary`)

At the end it prints the per-cell table and two regression guards that are
specific to this project's history: a **graded-F check** (is some `f_mean`
strictly between 0 and 1? if not, the step-function bug is back) and an
**output-cluster degeneracy check** (fraction of cells with `a_q == 1`, which
floors H_sem/FI_out/variation ratio).

---

## 11. Obtaining and evaluating results

### 11.1 Local sequence (Makefile / tasks.ps1)

The Makefile maps each step to `uv run python -m prompt_sensitivity.scripts.<x>`.
Sprint by sprint: `list-models` and `api-check` (gateway connectivity, only for
the legacy gateway path), `data-download`, `sample` (-> `sample_v1.json`),
`paraphrases` / `paraphrases-smoke` (-> paraphrase parquet) gated by
`compute-kappa`, `build-ladders` (-> `ladders.parquet`), `smoke-metrics` (the
metric-layer gate), `e2e-smoke` and the `pilot*` targets (-> metric parquets),
then `show-results` / `plot-pilot`.

### 11.2 Cluster sequence (the real full run)

`cluster/submit_full_pilot.sh` submits the two-stage SLURM pipeline:

1. `full_paraphrase_prep.sbatch`: `e2e_smoke --musique-strata 17
   --paraphrase-only --max-paraphrases 30`, building the shared paraphrase
   universe once (17 questions per 2/3/4-hop stratum = 51 questions) with the
   configured generator, persisted to `data/paraphrases_musique.parquet`.
2. Three per-model eval jobs (each `--dependency=afterok` on the prep), e.g.
   `e2e_smoke --musique-strata 17 --families context,reasoning --ladders random
   --levels 0,2,4,6,8,10 --k-samples 10 --max-paraphrases 30 --models <M>
   --own-encoder --include-posix --out data/full_<M>.parquet`. One parquet per
   model avoids a write race.

Each eval is preceded by `local_check` (Section 4.3) which loads each model and
verifies generate + logprobs + echo-score + hidden states before spending
walltime. `cluster/e2e_local.sbatch` is the single-node sequential variant of
the same flow.

Then, on the laptop: `merge_results` concatenates `data/full_*.parquet` into
`data/full_run.parquet`, de-duplicating on cell identity; `show_results` and
`plot_pilot` analyse it.

### 11.3 Reading the parquet (`show_results.py`)

Splits columns into a headline set (`f_mean`, `final_answer_f_mean`, `aufi_in`,
`posix_psi`, `fi_out_mean`, `h_sem_mean`) and de-emphasised diagnostics. Prints
two pivots: mean chain-F by `(ladder_family, level)` and mean AUFI_in by
`(ladder_family, level)`, the latter being the dual-ladder functional-information
headline (it should fall as you climb either ladder). It runs the graded-F
sanity check and a **collinearity guard**: if `Spearman(f_mean, aufi_in)`
exceeds 0.95, AUFI_in is just a transform of mean F at this sample, so the
headline should be the full FI_in(k) curve, not the scalar. With `--plot` it
writes the accuracy dual-ladder figure (chain-F solid, final-answer F dashed)
and the AUFI_in dual-ladder figure.

### 11.4 The presentation plots (`plot_pilot.py`)

Reads a metric parquet and writes a numbered set of PNGs plus a `REPORT.md`
under `data/plots/`: F accuracy vs level, AUFI_in vs level, H_sem vs level, the
three-ladder envelope (gold_first/random/distractor_first bars, the V3 bound
check), a Spearman correlation heatmap across the metric scalars (the
validation-lite cross-check), model comparisons, quality vs context bits
(empirical curve against `b_theo`), and a Hazen-step plot. `_v3_check` verifies
the gold_first >= random >= distractor_first ordering. The report embeds the
plots with a small-N caveat.

### 11.5 What "good" looks like, in the code's own terms

The validation targets are encoded in the analysis scripts: the three-ladder
ordering must hold (`_v3_check`), the FI_in/AUFI_in curve should fall as
context or reasoning is added while accuracy rises, AUFI_in must not be perfectly
collinear with mean F (else report the curve), and the x* distance correlation
should be `<= -0.4`. The dual-ladder comparison (does feeding reasoning lift the
chain-completion fraction faster per item than feeding the paragraphs that
contain it?) is the headline output, read off the `(ladder_family, level)`
pivots and figures.

---

## 12. Data flow in one line each

```
sample_questions      HF datasets ----------------> data/sample_v1.json
generate_paraphrases  sample_v1.json + datasets ---> data/paraphrases_v1.parquet   (gen=phi_4_14b, NLI+gold-judge+dedup)
build_ladders         sample_v1.json + datasets ---> data/ladders.parquet          (3 context ladders x 6 levels)
e2e_smoke             paraphrases + datasets ------> data/<run>.parquet             (one MetricTuple row per cell)
                        per cell: prompts -> F(x) -> H_sem samples -> pooled clusters
                                  -> embeddings (+POSIX) -> build_metric_tuple
merge_results         data/full_*.parquet ---------> data/full_run.parquet
show_results          metric parquet --------------> stdout tables + dual-ladder PNGs
plot_pilot            metric parquet --------------> data/plots/*.png + REPORT.md
x_star (cluster)      full_run.parquet + cache ----> data/x_star_analysis.parquet
```

Throughout, the SQLite cache (`data/cache/llm_cache.sqlite`) sits under every
model call, so re-runs only pay for genuinely new `(prompt, model, params)`
tuples.

---

## 13. Tests

`tests/` mirrors the package one file per module (`test_fi_in.py`,
`test_h_sem_clustering.py`, `test_errica.py`, `test_posix.py`,
`test_chain_score.py`, `test_ladders.py`, `test_paraphrase_pipeline.py`,
`test_nli_filter.py`, `test_local_hf.py`, `test_orchestrator.py`, and more). The
heavy seams (DeBERTa NLI, model loading) are patchable, so the pure math and
control flow are unit-tested without a GPU or network. `tests/conftest.py` holds
fixtures; `pyproject.toml` marks `needs_api` and `needs_gpu` so those are skipped
by default. Run with `make test` (`uv run pytest -q`).

---

## 14. Quick-start mental model for a new developer

1. A **question** has a gold answer and (for MuSiQue) a gold reasoning chain.
2. The **paraphrase pipeline** turns it into ~30 meaning-preserving rewrites,
   guarded by bidirectional NLI, a gold answer-set judge, and dedup.
3. The **dual ladder** builds prompts that add either context paragraphs
   (random / gold_first / distractor_first) or reasoning hops, level by level.
4. For each `(question, ladder, level, model)` **cell**, the model answers every
   paraphrase; answers are scored (binary NLI-with-gold, or graded
   chain-completion for MuSiQue) and sampled-again-and-clustered for the
   output-space metrics.
5. **FI_in** = `-log2(fraction of paraphrases that succeed)`, summarised as
   **AUFI_in** (bits). High means prompt sensitive. The supporting metrics
   (H_sem, FI_out, Errica, POSIX, ESS_in, rho_u, spread, variation ratio)
   triangulate the same phenomenon.
6. Everything is a pure function over precomputed inputs, assembled into a
   `MetricTuple` per cell by `e2e_smoke`, written to parquet, and analysed by
   `show_results` / `plot_pilot`.
7. The headline result is the FI_in/AUFI_in curve falling as you climb each
   ladder, and the comparison between the context and reasoning ladders.
```
