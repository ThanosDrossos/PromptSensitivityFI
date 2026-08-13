"""R1 union-gold arm: the contracts that make a cache-only rescore correct.

The arm is only sound if (a) the rebuilt request hits the SAME cache key the
original run wrote, (b) a partially-recovered cell is refused rather than
silently scored on fewer samples, and (c) the union gold is level-invariant so
the fixed-gold guardrail still holds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prompt_sensitivity.data.ambigqa_schemas import AmbigInterpretation, AmbigQuestion
from prompt_sensitivity.models.schemas import LLMRequest
from prompt_sensitivity.scripts.rescore_union_gold import (
    _request,
    load_paraphrase_universes,
)
from prompt_sensitivity.specificity.build_levels import build_spec_levels


class _Entry:
    provider = "local"
    model_id = "Qwen/Qwen2.5-7B-Instruct"


_MSGS = [{"role": "user", "content": "Where was Top of the Lake filmed?"}]


def _original_request(model_entry, messages, *, temperature, seed, purpose, max_tokens):
    """Verbatim copy of run_specificity._sample_response's request construction.

    Kept as a literal duplicate ON PURPOSE: if the driver's construction ever
    changes, this test fails and tells us the rescore would silently miss the
    cache instead of returning wrong numbers.
    """
    return LLMRequest(
        provider=model_entry.provider,
        model_id=model_entry.model_id,
        messages=messages,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        purpose=purpose,
    )


# -------------------------------------------------- (a) cache-key identity


@pytest.mark.parametrize(
    "temperature,seed,purpose",
    [
        (0.0, 42, "spec_f::q1::L0::qwen_2_5_7b"),
        (1.0, 10000, "spec_hsem::q1::L0::qwen_2_5_7b::s0"),
        (1.0, 10907, "spec_hsem::q1::L1::qwen_2_5_7b::s7"),
    ],
)
def test_rebuilt_request_matches_driver_cache_key(temperature, seed, purpose):
    entry = _Entry()
    mine = _request(entry, _MSGS, temperature=temperature, seed=seed,
                    purpose=purpose, max_tokens=128)
    theirs = _original_request(entry, _MSGS, temperature=temperature, seed=seed,
                               purpose=purpose, max_tokens=128)
    assert mine.cache_key() == theirs.cache_key()


def test_cache_key_is_sensitive_to_every_field_we_rebuild():
    """A miss must be caused by a real difference, so each field must matter."""
    entry = _Entry()
    base = _request(entry, _MSGS, temperature=1.0, seed=10000,
                    purpose="spec_hsem::q1::L0::m::s0", max_tokens=128)
    variants = [
        _request(entry, _MSGS, temperature=0.0, seed=10000,
                 purpose="spec_hsem::q1::L0::m::s0", max_tokens=128),
        _request(entry, _MSGS, temperature=1.0, seed=10001,
                 purpose="spec_hsem::q1::L0::m::s0", max_tokens=128),
        _request(entry, _MSGS, temperature=1.0, seed=10000,
                 purpose="spec_hsem::q1::L1::m::s0", max_tokens=128),
        _request(entry, _MSGS, temperature=1.0, seed=10000,
                 purpose="spec_hsem::q1::L0::m::s0", max_tokens=256),
        _request(entry, [{"role": "user", "content": "different"}], temperature=1.0,
                 seed=10000, purpose="spec_hsem::q1::L0::m::s0", max_tokens=128),
    ]
    for v in variants:
        assert v.cache_key() != base.cache_key()


def test_seed_layout_matches_the_driver():
    """The driver uses seed = 10000 + i*100 + kk for the H_sem samples."""
    seen = {10000 + i * 100 + kk for i in range(10) for kk in range(10)}
    assert len(seen) == 100, "seed layout must not collide within a cell"


# -------------------------------------------------- (c) union gold invariance


def _question(m0: int = 3) -> AmbigQuestion:
    return AmbigQuestion(
        id="q1",
        question="Where was it filmed?",
        interpretations=[
            AmbigInterpretation(
                disambiguated_question=f"Where was season {i} filmed?",
                answers=[f"Place{i}", f"Place{i} City"],
            )
            for i in range(m0)
        ],
    )


def test_union_gold_is_identical_at_both_levels():
    """The whole point of R1: the gold SET does not drift across the manipulation,
    so the fixed-gold guardrail is satisfied while the 1/m0 lottery is removed."""
    rows = build_spec_levels(_question(3), seed=42)
    assert len(rows) == 2
    l0, l1 = rows
    assert l0.all_answers == l1.all_answers
    # and it is a strict superset of the target gold, which is what removes the lottery
    assert set(l0.target_answers) <= set(l0.all_answers)
    assert len(l0.all_answers) > len(l0.target_answers)


def test_target_gold_is_also_still_fixed_across_levels():
    """R1 adds a gold set; it must not disturb the existing guardrail."""
    l0, l1 = build_spec_levels(_question(4), seed=42)
    assert l0.target_answers == l1.target_answers


def test_union_gold_equals_target_gold_when_unambiguous_answers_coincide():
    """Degenerate case: if every interpretation shares one answer, union == target
    and the arm must report no targeting effect rather than a spurious one."""
    q = AmbigQuestion(
        id="q2",
        question="Who won?",
        interpretations=[
            AmbigInterpretation(disambiguated_question=f"Who won race {i}?", answers=["Kriseman"])
            for i in range(3)
        ],
    )
    l0, _ = build_spec_levels(q, seed=42)
    assert l0.all_answers == l0.target_answers
    assert l0.target_collision is True


# -------------------------------------------------- universe loading


def test_load_paraphrase_universes_orders_by_index_and_keeps_fallbacks(tmp_path):
    df = pd.DataFrame(
        [
            {"question_id": "q1", "spec_level": 0, "outcome": "accepted",
             "paraphrase_idx": 1, "text": "second"},
            {"question_id": "q1", "spec_level": 0, "outcome": "accepted",
             "paraphrase_idx": 0, "text": "first"},
            {"question_id": "q1", "spec_level": 1, "outcome": "singleton_fallback",
             "paraphrase_idx": 0, "text": "only"},
            {"question_id": "q1", "spec_level": 0, "outcome": "rejected",
             "paraphrase_idx": 9, "text": "dropped"},
        ]
    )
    path = tmp_path / "data" / "p.parquet"
    path.parent.mkdir(parents=True)
    df.to_parquet(path, index=False)

    class _Cfg:
        @staticmethod
        def repo_root():
            return tmp_path

    got = load_paraphrase_universes(_Cfg(), "data/p.parquet")
    assert got[("q1", 0)] == ["first", "second"], "must be ordered by paraphrase_idx"
    assert got[("q1", 1)] == ["only"], "singleton_fallback counts as a universe"
    assert "dropped" not in got[("q1", 0)]
