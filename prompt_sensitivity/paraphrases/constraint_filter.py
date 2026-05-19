"""GPT-4.1-as-judge constraint-set filter.

Section_7 §7.6.1 R1 second check: a paraphrase must preserve the *answer set*.
We elicit the candidate-answer set for both the original and the paraphrase
from the judge model and require Jaccard >= 0.9 between the two sets.

The judge prompt is single-pass deterministic (T=0) and asks the judge to
return JSON `{"answers": ["...", "..."]}`. We tolerate minor format drift
(stripping ```json fences, lowercasing keys).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from loguru import logger

from ..config import Config, load_config
from ..models import LLMRequest
from ..models.registry import get_client
from ..models.schemas import ChatMessage


_SYSTEM = (
    "You are a precise question-analysis assistant. Given a question, list "
    "ALL short factual answers that would be considered correct, in JSON. "
    "Include common surface variants (full name and short name, with and "
    "without honorifics, etc.) but do not invent extra alternatives. "
    "Output ONLY a JSON object of the form {\"answers\": [\"...\", \"...\"]}. "
    "No explanation, no markdown fences."
)


_USER_TEMPLATE = "Question:\n{question}\n\nAnswers JSON:"


@dataclass(frozen=True)
class JaccardResult:
    jaccard: float
    a_set: frozenset[str]
    b_set: frozenset[str]


def _norm(s: str) -> str:
    """Lower-case, collapse whitespace, strip surrounding punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,;:!?\"'()[]{}")
    return s


def _parse_answers_json(text: str) -> list[str]:
    """Extract the `answers` list from the judge response.

    Robust to ```json``` fences, leading/trailing prose, and the empty case.
    """
    if not text:
        return []
    # Strip ```json fences if present.
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # If the model returned prose before the JSON, find the first {...} block.
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("judge JSON parse failed: {} (text={!r})", exc, text[:200])
        return []
    if not isinstance(obj, dict):
        return []
    answers = obj.get("answers") or obj.get("Answers") or []
    if not isinstance(answers, list):
        return []
    return [str(a) for a in answers if isinstance(a, (str, int, float))]


def _judge_request(judge_model_key: str, question: str, config: Config) -> LLMRequest:
    entry = config.models[judge_model_key]
    return LLMRequest(
        provider=entry.provider,  # type: ignore[arg-type]
        model_id=entry.model_id,
        messages=[
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user", content=_USER_TEMPLATE.format(question=question.strip())),
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=config.paraphrases.constraint_filter.judge_max_tokens,
        seed=42,
        purpose="constraint_judge",
    )


def list_answers(question: str, *, config: Config | None = None) -> list[str]:
    """Call the judge on one question; return the parsed answers list."""
    if config is None:
        config = load_config()
    judge_key = config.paraphrases.constraint_filter.judge_model
    client = get_client(judge_key, config)
    resp = client.complete(_judge_request(judge_key, question, config))
    return _parse_answers_json(resp.text)


def compare_answer_sets(original: str, paraphrase: str, *, config: Config | None = None) -> JaccardResult:
    """Run the judge twice and compute Jaccard between the two normalised sets."""
    a = frozenset(_norm(s) for s in list_answers(original, config=config) if s)
    b = frozenset(_norm(s) for s in list_answers(paraphrase, config=config) if s)
    return JaccardResult(jaccard=jaccard(a, b), a_set=a, b_set=b)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard coefficient with the convention Jaccard(empty, empty) = 1.0.

    The empty case is treated as 1.0 because if the judge returns no
    candidates for either side, there is no constraint disagreement either —
    the filter should not flag this as a constraint violation. The NLI filter
    catches actual semantic drift; this filter only catches set drift.
    """
    a_set = set(a)
    b_set = set(b)
    if not a_set and not b_set:
        return 1.0
    union = a_set | b_set
    if not union:
        return 1.0
    return len(a_set & b_set) / len(union)


def filter_by_constraint(
    original: str,
    candidates: Iterable[str],
    *,
    config: Config | None = None,
    threshold: float | None = None,
) -> list[tuple[bool, JaccardResult]]:
    """Apply the constraint-set filter to each candidate against the original.

    Note: the original's answer set is computed ONCE (cache hit on second
    invocation via the gateway cache) and reused across all candidates.
    """
    if config is None:
        config = load_config()
    if threshold is None:
        threshold = config.paraphrases.constraint_filter.jaccard_threshold
    a_set = frozenset(_norm(s) for s in list_answers(original, config=config) if s)
    out: list[tuple[bool, JaccardResult]] = []
    for cand in candidates:
        b_set = frozenset(_norm(s) for s in list_answers(cand, config=config) if s)
        j = jaccard(a_set, b_set)
        out.append((j >= threshold, JaccardResult(jaccard=j, a_set=a_set, b_set=b_set)))
    return out
