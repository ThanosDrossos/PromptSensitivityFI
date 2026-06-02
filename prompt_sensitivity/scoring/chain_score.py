"""Graded chain-completion scorer. Dataset_Evaluation_v6_Dual_Ladder.md §2, §5.

The v3 binary F (final-answer match) collapsed the FI_in curve to a step
function: on HotpotQA the answer is one short span fixed by two gold
paragraphs, so F was 0 or 1 with nothing in between (see data/plots/REPORT.md).

The fix (v6): score the *reasoning chain*, not the final answer string. For a
MuSiQue question with H gold hops, F is the fraction of those hops the model
recovers in its response:

    F = recovered_hops / H        in {0, 1/H, 2/H, ..., 1}

With 4-hop questions F lands in {0, 0.25, 0.5, 0.75, 1.0} — graded, so AUFI_in
integrates a genuinely smooth curve. The metric stack consumes this float
exactly as it consumed the binary 0/1 (the orchestrator already takes
`scores: Sequence[float]`); nothing under metrics/ changes.

Hop recovery is judged with the SAME bidirectional DeBERTa NLI as the binary
path (no second model loaded). For each gold hop we build a fact statement and
check whether the model's response entails it.

This module does NOT modify `nli_with_gold.py`; it reuses
`score_batch_nli_with_gold` as its NLI primitive.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..config import Config, load_config
from ..data.schemas import DecompositionHop
from .nli_with_gold import score_batch_nli_with_gold


_PLACEHOLDER_RE = re.compile(r"#(\d+)")


def resolve_placeholders(decomposition: Sequence[DecompositionHop]) -> list[str]:
    """Resolve `#k` placeholders in each sub-question to earlier gold sub-answers.

    MuSiQue sub-questions reference earlier hops as "#1", "#2" (1-based). We
    substitute each "#k" with hop k-1's gold sub_answer so the rendered
    sub-question reads as a self-contained natural-language question.
    Unresolvable references (out of range) are left as-is.
    """
    answers_by_one_based: dict[int, str] = {
        hop.hop_idx + 1: hop.sub_answer for hop in decomposition
    }

    def _sub(text: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            k = int(m.group(1))
            return answers_by_one_based.get(k, m.group(0))

        return _PLACEHOLDER_RE.sub(_repl, text)

    return [_sub(hop.sub_question) for hop in decomposition]


def build_fact_statements(decomposition: Sequence[DecompositionHop]) -> list[str]:
    """One gold fact statement per hop, used as the NLI target.

    Design choice (documented per the brief): we render
    `"{resolved_sub_question} {sub_answer}"` rather than the bare `sub_answer`.
    Rationale: a bare entity ("John Kelly Sr.") is a weak NLI hypothesis — a
    model response can entail the entity by coincidence. Pairing the resolved
    sub-question with its answer ("Who is the maternal grandfather of Albert
    II of Monaco? John Kelly Sr.") makes the hypothesis specific to the hop's
    actual claim, so entailment requires the response to support that
    particular fact. If the sub-question is empty we fall back to the bare
    sub_answer.
    """
    resolved_questions = resolve_placeholders(decomposition)
    facts: list[str] = []
    for hop, rq in zip(decomposition, resolved_questions, strict=True):
        rq = rq.strip()
        ans = hop.sub_answer.strip()
        if rq:
            facts.append(f"{rq} {ans}".strip())
        else:
            facts.append(ans)
    return facts


def chain_fraction(recovered: Sequence[bool]) -> float:
    """Pure fraction helper: recovered_hops / total_hops. Empty -> raises."""
    if not recovered:
        raise ValueError("recovered must be non-empty (empty decomposition)")
    return sum(1 for r in recovered if r) / len(recovered)


def _hop_recovered(
    model_response: str,
    fact: str,
    *,
    config: Config,
) -> bool:
    """Bidirectional NLI: does `model_response` support `fact`?

    Direction A: premise = model_response, hypothesis = fact (does the
                 response entail the gold fact?).
    Direction B: premise = fact, hypothesis = model_response.

    Per the brief we take the direction with the HIGHER entailment
    probability, then apply the SAME entail/contradict thresholds the binary
    path uses (config.scoring.entail_threshold / contradict_threshold). The
    chosen direction's contradiction probability gates the pass.
    """
    a = score_batch_nli_with_gold(model_response, [fact], config=config)[0]
    b = score_batch_nli_with_gold(fact, [model_response], config=config)[0]
    chosen = a if a.entail_prob >= b.entail_prob else b
    return chosen.passes_entail and chosen.passes_contradict


def chain_completion_score(
    decomposition: Sequence[DecompositionHop],
    model_response: str,
    *,
    config: Config | None = None,
) -> float:
    """Fraction of gold reasoning hops recovered in `model_response`, in [0, 1].

    Raises ValueError on an empty decomposition — the caller MUST route
    HotpotQA / 2Wiki (no decomposition) to the binary `f_score` path instead.
    """
    if not decomposition:
        raise ValueError(
            "empty decomposition; chain scoring is MuSiQue-only. Route "
            "non-decomposed questions to the binary f_score path."
        )
    if config is None:
        config = load_config()
    facts = build_fact_statements(decomposition)
    recovered = [_hop_recovered(model_response, f, config=config) for f in facts]
    return chain_fraction(recovered)


def chain_completion_score_batch(
    decomposition: Sequence[DecompositionHop],
    responses: Sequence[str],
    *,
    config: Config | None = None,
) -> list[float]:
    """Chain-completion fraction for many responses (one cell's paraphrases).

    Amortises the DeBERTa forward passes: for N responses and H hops we issue
    N + H batched NLI calls (one per response in direction A, one per fact in
    direction B) rather than 2*N*H individual calls. Each call goes through
    the shared lru-cached loader, so no second model is loaded.
    """
    if not decomposition:
        raise ValueError(
            "empty decomposition; chain scoring is MuSiQue-only. Route "
            "non-decomposed questions to the binary f_score path."
        )
    if config is None:
        config = load_config()
    responses = list(responses)
    if not responses:
        return []

    facts = build_fact_statements(decomposition)
    n, h = len(responses), len(facts)

    # Direction A: premise = response_r, hypotheses = all facts.
    #   entail_A[r][hop] = P(response_r entails fact_hop)
    entail_a: list[list[float]] = []
    contra_a: list[list[float]] = []
    for r in responses:
        res = score_batch_nli_with_gold(r, facts, config=config)
        entail_a.append([x.entail_prob for x in res])
        contra_a.append([x.contradict_prob for x in res])

    # Direction B: premise = fact_hop, hypotheses = all responses.
    #   entail_B[hop][r] = P(fact_hop entails response_r)
    entail_b: list[list[float]] = []
    contra_b: list[list[float]] = []
    for f in facts:
        res = score_batch_nli_with_gold(f, responses, config=config)
        entail_b.append([x.entail_prob for x in res])
        contra_b.append([x.contradict_prob for x in res])

    entail_thr = config.scoring.entail_threshold
    contra_thr = config.scoring.contradict_threshold

    scores: list[float] = []
    for r in range(n):
        recovered: list[bool] = []
        for hop in range(h):
            ea, ca = entail_a[r][hop], contra_a[r][hop]
            eb, cb = entail_b[hop][r], contra_b[hop][r]
            # Pick the higher-entailment direction; gate on its contradiction.
            if ea >= eb:
                passed = ea >= entail_thr and ca < contra_thr
            else:
                passed = eb >= entail_thr and cb < contra_thr
            recovered.append(passed)
        scores.append(chain_fraction(recovered))
    return scores
