"""In-process Hugging Face `transformers` backend — provider ``"local"``.

Sprint 6 (KIT cluster). The LiteLLM gateway (`registry.LiteLLMClient`) could only
return generated text + top-20 chat logprobs and had no echo / no hidden states
(`kit.gpt-4.1`). Loading the weights **directly in this process** exposes the
full surface the metric stack wants, all from one forward/generate pass:

  * generated text                          -> `complete()`            (chat)
  * per-token logprobs (+ top-k, any k)     -> `complete(logprobs=True)`
  * exact teacher-forced `log P(y|x)`       -> `score_continuation()`  (POSIX)
  * last-layer hidden states (mask-mean)    -> `embed_hidden()`        (ESS_in^own / rho_u^own)

No HTTP, no server, no vLLM, no LiteLLM proxy. One model is resident per process
(the cluster driver runs one eval model per invocation; see cluster/e2e_local.sbatch).

The chat template is applied with the model's own tokenizer
(`apply_chat_template`), which the gateway docstrings explicitly could not do.

See `registry.py` (factory registers this under provider "local") and
Section_7 §7.6 for the metric contracts.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from loguru import logger

from ..config import Config, ModelEntry
from .cache import LLMCache
from .registry import BaseLLMClient
from .schemas import CompletionRequest, LLMRequest, LLMResponse, TokenLogprob


# --------------------------------------------------------------------------- #
# Model loading (one weights copy per model_id per process)                   #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=2)
def _load_model(model_id: str):  # type: ignore[no-untyped-def]
    """Load `(tokenizer, model)` once per `model_id`, eval mode.

    Default (cluster): bf16 on CUDA, exactly as before. Two env overrides for
    the laptop/live-demo path (app/streamlit_app.py sets them automatically):

      PSF_DEVICE=cpu   bf16 on CPU (no GPU / no CUDA torch build);
      PSF_4BIT=1       NF4 4-bit via bitsandbytes with device_map="auto" —
                       fits a 7B on a 6 GB RTX 2060. Hidden states under NF4
                       differ slightly from the bf16 training features; the
                       app runs and reports a fidelity check (2026-07-27).

    Tries FlashAttention-2 first, falls back to SDPA. lru_cache: re-get_client
    is free; cache_clear() + empty_cache() frees VRAM between models.
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = os.environ.get("PSF_DEVICE", "cuda")
    use_4bit = os.environ.get("PSF_4BIT", "0") == "1"
    logger.info("loading local CausalLM {} ({}) ...", model_id,
                "4-bit NF4" if use_4bit else f"bf16, {device}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        # Decoder-only models often ship without a pad token; reuse EOS. We
        # always mask, so this never contaminates pooled embeddings.
        tok.pad_token = tok.eos_token

    common: dict[str, Any] = dict(torch_dtype=torch.bfloat16, device_map=device)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        common = dict(
            # {"": 0} pins the whole model to GPU 0, bypassing accelerate's
            # auto planner — whose conservative headroom reserve refuses to
            # pack ~4.8 GB into a 6 GB card and then demands CPU offload
            # (which segfaults bnb 0.50 on Windows).
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                # Quantize lm_head too (bnb skips it by default, keeping it
                # bf16 ~1.1 GB for Qwen's 152k vocab). We only pool hidden
                # states — logits are never read — and without that block the
                # 7B doesn't fit a 6 GB GPU and accelerate wants CPU offload,
                # which segfaults bnb 0.50 on Windows (tested 2026-07-27).
                llm_int8_skip_modules=[],
            ),
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", **common
        )
    except Exception as exc:  # noqa: BLE001 — flash-attn absent or unsupported
        logger.warning("flash_attention_2 unavailable ({}); using sdpa", str(exc)[:120])
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="sdpa", **common
        )
    model.eval()
    return tok, model


# --------------------------------------------------------------------------- #
# Pure-ish helpers (unit-tested without a GPU)                                 #
# --------------------------------------------------------------------------- #


def _format_chat(tokenizer, messages: list[dict]) -> Any:
    """Apply the model's chat template -> BatchEncoding (input_ids + attention_mask).

    `return_dict=True` so we get a `BatchEncoding` regardless of the tokenizer
    (some return a bare tensor, some a dict) and can pass `attention_mask` to
    `generate`. Some instruct templates (older Mistral) reject a standalone
    ``system`` role; fall back to folding the system text into the first user
    turn so the three eval models + the Qwen generator all work uniformly.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception as exc:  # noqa: BLE001 — template rejected the role layout
        logger.warning("chat template rejected roles ({}); folding system->user", str(exc)[:120])
        folded: list[dict] = []
        sys_parts: list[str] = []  # accumulate ALL system turns until a user consumes them
        for m in messages:
            if m["role"] == "system":
                sys_parts.append(m["content"])
                continue
            if sys_parts and m["role"] == "user":
                folded.append({"role": "user", "content": "\n\n".join([*sys_parts, m["content"]])})
                sys_parts = []
            else:
                folded.append(m)
        if sys_parts:  # system text never reached a user turn -> emit it as one
            folded.append({"role": "user", "content": "\n\n".join(sys_parts)})
        return tokenizer.apply_chat_template(
            folded, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )


def _apply_stop(text: str, stop: list[str] | None) -> tuple[str, bool]:
    """Truncate `text` at the earliest stop string. Returns (text, stopped?)."""
    if not stop:
        return text, False
    cut = len(text)
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    if cut < len(text):
        return text[:cut], True
    return text, False


def _build_token_logprobs(
    tokenizer, scores, new_token_ids, top_logprobs: int | None
) -> list[TokenLogprob] | None:
    """Per-generated-token logprob (+ top-k) from `generate(output_scores=True)`.

    `scores` is a tuple (len = #generated tokens) of (1, vocab) logits. We
    log-softmax each step, read off the chosen token's logprob, and attach the
    top-k alternatives. Full vocab is available locally (no gateway top-20 cap):
    `top_logprobs` is honoured as-is (default 5 when None), clamped only to the
    vocabulary size.
    """
    import torch

    k = 5 if top_logprobs is None else int(top_logprobs)
    out: list[TokenLogprob] = []
    n = min(len(scores), int(new_token_ids.shape[0]))
    for t in range(n):
        logp = torch.log_softmax(scores[t][0].float(), dim=-1)
        chosen_id = int(new_token_ids[t].item())
        top: dict[str, float] = {}
        if k > 0:
            vals, idxs = torch.topk(logp, min(k, int(logp.shape[0])))
            for v, j in zip(vals.tolist(), idxs.tolist()):
                top[tokenizer.decode([j])] = float(v)
        out.append(
            TokenLogprob(
                token=tokenizer.decode([chosen_id]),
                logprob=float(logp[chosen_id].item()),
                top_logprobs=top,
            )
        )
    return out or None


def _masked_mean_pool(last_hidden, attention_mask):  # type: ignore[no-untyped-def]
    """Mean-pool a (B, T, D) hidden state over real tokens -> (B, D) float32 numpy."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)                    # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)                     # (B, 1)
    return (summed / counts).float().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# The client                                                                  #
# --------------------------------------------------------------------------- #


class LocalHFClient(BaseLLMClient):
    """Provider ``"local"`` — weights in-process via `transformers`.

    Reuses `BaseLLMClient.complete` / `score_continuation` for the SQLite cache,
    timing, and the (no-op for local) rate bucket. Only the provider-specific
    `_raw_call` / `_raw_completion` and the extra `embed_hidden` are new.
    """

    def __init__(self, model_entry: ModelEntry, config: Config, cache: LLMCache) -> None:
        super().__init__(model_entry, config, cache)

    # ---- chat generation ----------------------------------------------------

    def _raw_call(self, request: LLMRequest) -> LLMResponse:
        import torch

        tok, model = _load_model(self.entry.model_id)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        enc = _format_chat(tok, messages).to(model.device)
        input_ids = enc["input_ids"]
        prompt_len = int(input_ids.shape[1])

        do_sample = bool(request.temperature and request.temperature > 0.0)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=request.max_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
            output_scores=bool(request.logprobs),
        )
        if do_sample:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        # Seed reproducibly WITHOUT touching global RNG. `transformers.set_seed`
        # would reset python/numpy/torch globals on every call and perturb e.g.
        # the bootstrap CIs computed later in the same process. fork_rng saves +
        # restores the RNG around just this generate; we seed inside it, and only
        # when sampling (greedy is already deterministic).
        fork_devices = [input_ids.device.index] if input_ids.device.type == "cuda" else []
        with torch.random.fork_rng(
            devices=fork_devices, enabled=do_sample and request.seed is not None
        ):
            if do_sample and request.seed is not None:
                torch.manual_seed(int(request.seed))
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)

        seq = out.sequences[0]
        new_token_ids = seq[prompt_len:]
        text = tok.decode(new_token_ids, skip_special_tokens=True)

        eos_ids = _eos_id_set(tok)
        hit_eos = new_token_ids.shape[0] > 0 and int(new_token_ids[-1].item()) in eos_ids
        finish_reason = "stop" if hit_eos else "length"

        token_logprobs = None
        if request.logprobs and getattr(out, "scores", None) is not None:
            token_logprobs = _build_token_logprobs(
                tok, out.scores, new_token_ids, request.top_logprobs
            )

        text, stopped = _apply_stop(text, request.stop)
        completion_tokens = int(new_token_ids.shape[0])
        if stopped:
            finish_reason = "stop"
            # Keep completion_tokens / token_logprobs aligned with the returned
            # (truncated) text. Best-effort: a prefix's token count can differ
            # slightly from the original generation's boundaries.
            trunc_n = len(tok(text, add_special_tokens=False)["input_ids"])
            completion_tokens = min(completion_tokens, trunc_n)
            if token_logprobs is not None:
                token_logprobs = token_logprobs[:completion_tokens] or None

        return LLMResponse(
            request_hash=request.cache_key(),
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            token_logprobs=token_logprobs,
            raw_provider_response=None,
        )

    # ---- exact teacher-forced scoring (POSIX) -------------------------------

    def _raw_completion(self, request: CompletionRequest) -> LLMResponse:
        """One forward pass over `request.prompt`; per-token logprob of each
        actual next token (log P(t_i | t_<i)). Exact — replaces the gateway's
        `echo=true` approximation. `_posix_matrix` slices the continuation tail.
        """
        import torch

        tok, model = _load_model(self.entry.model_id)
        enc = tok(request.prompt, return_tensors="pt").to(model.device)
        ids = enc["input_ids"][0]
        with torch.no_grad():
            logits = model(**enc).logits[0].float()  # (T, V); **enc keeps attention_mask
        logp = torch.log_softmax(logits, dim=-1)

        token_logprobs: list[TokenLogprob] = []
        for t in range(1, int(ids.shape[0])):
            tok_id = int(ids[t].item())
            token_logprobs.append(
                TokenLogprob(token=tok.decode([tok_id]), logprob=float(logp[t - 1, tok_id].item()))
            )

        return LLMResponse(
            request_hash=request.cache_key(),
            text=request.prompt if request.echo else "",
            finish_reason="stop",
            prompt_tokens=int(ids.shape[0]),
            completion_tokens=0,
            token_logprobs=token_logprobs or None,
            raw_provider_response=None,
        )

    # ---- last-layer hidden states (own-encoder ESS_in / rho_u) --------------

    def embed_hidden(
        self, texts: list[str], *, batch_size: int = 16, max_length: int | None = None
    ) -> np.ndarray:
        """Mask-mean-pooled last hidden layer -> (N, D) float32. The model's OWN
        representation, used by ESS_in^own / rho_u^own instead of the mpnet proxy.

        `max_length` defaults to the model's context window capped at 4096 (to
        bound activation memory, since `output_hidden_states` materialises every
        layer). Inputs longer than the cap are truncated AND a warning is logged
        — silent context loss would corrupt ESS_in / rho_u for high-context
        ladder rungs (10 MuSiQue paragraphs can exceed 2048 tokens).
        """
        import torch

        tok, model = _load_model(self.entry.model_id)
        if not texts:
            return np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)

        model_max = int(getattr(model.config, "max_position_embeddings", 4096) or 4096)
        cap = int(max_length) if max_length is not None else min(model_max, 4096)

        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[start : start + batch_size]]
            raw_lens = [len(ids) for ids in tok(batch, add_special_tokens=True)["input_ids"]]
            if any(rl > cap for rl in raw_lens):
                logger.warning(
                    "embed_hidden: {} input(s) exceed max_length={} (longest={}) — truncating; "
                    "ESS_in/rho_u for these use a truncated context",
                    sum(rl > cap for rl in raw_lens), cap, max(raw_lens),
                )
            enc = tok(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=cap
            ).to(model.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            chunks.append(_masked_mean_pool(out.hidden_states[-1], enc["attention_mask"]))
        pooled = np.concatenate(chunks, axis=0).astype(np.float32)
        # P1-1: L2-normalize so ESS_in / rho_u measure prompt-space DISPERSION,
        # not raw hidden-state norm (which differs 5-10x across architectures —
        # Llama median 22 vs Mistral 209 on the smoke, Smoke_Run §5.5). Unit
        # vectors put every model on the same scale.
        norms = np.linalg.norm(pooled, axis=-1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    # ---- TBG hidden states for FI probes (FI_PROBES_PLAN.md §2) --------------

    def chat_hidden_states(
        self,
        messages_batch: list[list[dict]],
        *,
        layer_fracs: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        max_length: int | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """TBG (token-before-generation) hidden states at selected layers.

        For each conversation: apply the model's chat template with the
        generation prompt appended (`_format_chat`, same path `complete()`
        uses), run ONE forward pass, and take the LAST prompt token's hidden
        state — the exact state generation would start from (SEP's TBG
        position, Kossen et al. 2024) — at each fractional-depth layer.

        Returns `((N, L, D) float16 array, resolved layer indices)`. RAW states
        (no pooling, no L2 norm — unlike `embed_hidden`): probes should see the
        model's native geometry; any standardisation is the probe's choice.
        Conversations are processed one at a time: with per-sample encoding
        there is no padding, so "last token" needs no mask arithmetic; the
        forward is prompt-length only and cheap relative to generation.
        """
        import torch

        tok, model = _load_model(self.entry.model_id)
        num_layers = int(model.config.num_hidden_layers)
        layer_idxs = resolve_layer_fracs(num_layers, layer_fracs)
        hidden = int(model.config.hidden_size)
        if not messages_batch:
            return np.zeros((0, len(layer_idxs), hidden), dtype=np.float16), layer_idxs

        model_max = int(getattr(model.config, "max_position_embeddings", 4096) or 4096)
        cap = int(max_length) if max_length is not None else min(model_max, 4096)

        rows: list[np.ndarray] = []
        for messages in messages_batch:
            enc = _format_chat(tok, messages)
            if enc["input_ids"].shape[1] > cap:
                logger.warning(
                    "chat_hidden_states: prompt of {} tokens exceeds cap {} — truncating "
                    "(TBG state then sits on a truncated prompt)",
                    int(enc["input_ids"].shape[1]), cap,
                )
                enc = {k: v[:, -cap:] for k, v in enc.items()}
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            # (L, D): last token of the templated prompt at each selected layer.
            rows.append(
                np.stack(
                    [
                        out.hidden_states[i][0, -1].to(torch.float32).cpu().numpy()
                        for i in layer_idxs
                    ]
                ).astype(np.float16)
            )
        return np.stack(rows), layer_idxs


def _eos_id_set(tokenizer) -> set[int]:  # type: ignore[no-untyped-def]
    """Tokenizers may carry a list of EOS ids (e.g. Llama-3's <|eot_id|>)."""
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple)):
        return {int(x) for x in eos}
    return {int(eos)}


def resolve_layer_fracs(num_layers: int, fracs: "Sequence[float]") -> list[int]:
    """Map fractional depths to indices into HF's `hidden_states` tuple.

    `hidden_states` has num_layers+1 entries: index 0 is the embedding output,
    index num_layers the final layer. frac f -> round(f * num_layers), clamped
    to [1, num_layers] (the embedding layer carries no computation and is never
    selected). Deduplicated preserving order, so e.g. fracs (0.25, 0.5, 0.75, 1.0)
    on a 28-layer model give [7, 14, 21, 28]. Fractions make the layer choice
    model-agnostic across our eval models (28 vs 32 transformer layers).
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}")
    out: list[int] = []
    for f in fracs:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"layer fracs must be in (0, 1]; got {f}")
        idx = min(max(1, round(f * num_layers)), num_layers)
        if idx not in out:
            out.append(idx)
    return out
