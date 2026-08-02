"""Multi-level ladder (FINAL_PHASE_PLAN C1): subset chooser, gate, builder."""

from __future__ import annotations

import math

import pytest

from prompt_sensitivity.data.ambigqa_schemas import AmbigInterpretation, AmbigQuestion
from prompt_sensitivity.specificity.build_levels import (
    build_spec_levels,
    build_spec_levels_multilevel,
    choose_target_idx,
)
from prompt_sensitivity.specificity.midlevel import (
    MidQuestion,
    _leak_guard,
    _triviality_guard,
    append_mid_cache,
    choose_mid_subset,
    load_mid_cache,
)


def _q(qid: str = "mlq1", n: int = 4) -> AmbigQuestion:
    return AmbigQuestion(
        id=qid,
        question="Who won the championship?",
        interpretations=[
            AmbigInterpretation(
                disambiguated_question=f"Who won the championship in 200{i}?",
                answers=[f"team-{i}", f"club-{i}"],
            )
            for i in range(n)
        ],
    )


# --------------------------------------------------------------------------- #
# choose_mid_subset                                                            #
# --------------------------------------------------------------------------- #


def test_subset_deterministic_contains_target_correct_size():
    for m0 in (3, 4, 5):
        for target in range(m0):
            s1 = choose_mid_subset("qX", m0, target, seed=42)
            s2 = choose_mid_subset("qX", m0, target, seed=42)
            assert s1 == s2                                   # process-stable
            assert target in s1
            assert len(s1) == math.ceil(m0 / 2)
            assert 1 < len(s1) < m0                           # strictly between
            assert all(0 <= i < m0 for i in s1)
            assert list(s1) == sorted(s1)


def test_subset_varies_with_seed_and_question():
    base = choose_mid_subset("qX", 5, 0, seed=42)
    assert (
        choose_mid_subset("qX", 5, 0, seed=43) != base
        or choose_mid_subset("qY", 5, 0, seed=42) != base
    )


def test_subset_rejects_m0_below_3():
    with pytest.raises(ValueError):
        choose_mid_subset("qX", 2, 0, seed=42)


# --------------------------------------------------------------------------- #
# Guards                                                                       #
# --------------------------------------------------------------------------- #


def test_leak_guard_blocks_answer_strings():
    q = _q()
    assert _leak_guard("Which team took the title back then?", q)
    assert not _leak_guard("Did team-2 win the championship?", q)   # verbatim answer
    assert not _leak_guard("Was it TEAM-3 who won?", q)             # casefold


def test_triviality_guard_blocks_original_and_interpretations():
    q = _q()
    assert not _triviality_guard("Who won the championship?", q)
    assert not _triviality_guard("who won the championship in 2001", q)
    assert _triviality_guard("Who won the championship in the early 2000s?", q)
    assert not _triviality_guard("   ", q)


# --------------------------------------------------------------------------- #
# build_spec_levels_multilevel                                                 #
# --------------------------------------------------------------------------- #


def _mid_for(q: AmbigQuestion, seed: int = 42) -> tuple[tuple[int, ...], str]:
    target = choose_target_idx(q.id, q.m0(), seed=seed)
    subset = choose_mid_subset(q.id, q.m0(), target, seed=seed)
    return subset, "Who won the championship in the early years?"


def test_multilevel_three_rows_guardrails_and_fi_spec_monotone():
    q = _q(n=4)
    subset, text = _mid_for(q)
    rows = build_spec_levels_multilevel(q, seed=42, mid_text=text, mid_subset=subset)
    assert [r.spec_level for r in rows] == [0, 1, 2]
    # guardrail 1: fixed scoring gold across ALL levels
    assert rows[0].target_answers == rows[1].target_answers == rows[2].target_answers
    # guardrail 2: identical evidence across ALL levels
    assert rows[0].evidence == rows[1].evidence == rows[2].evidence
    # m_valid strictly decreasing -> FI_spec strictly increasing
    assert rows[0].m_valid == 4
    assert rows[1].m_valid == len(subset) == 2
    assert rows[2].m_valid == 1
    assert rows[1].question_text == text
    # target consistency with the two-level builder (same seed -> same target)
    two = build_spec_levels(q, seed=42)
    assert rows[0].target_idx == two[0].target_idx
    assert rows[2].question_text == two[1].question_text


def test_multilevel_mid_constraint_is_subset_union_target_first():
    q = _q(n=4)
    subset, text = _mid_for(q)
    rows = build_spec_levels_multilevel(q, seed=42, mid_text=text, mid_subset=subset)
    t = rows[0].target_idx
    mid = rows[1]
    assert mid.constraint_answers is not None
    assert mid.constraint_answers[:2] == [f"team-{t}", f"club-{t}"]
    expected = {a for i in subset for a in (f"team-{i}", f"club-{i}")}
    assert set(mid.constraint_answers) == expected
    # L0 and L_top rows keep the default (None -> driver rule)
    assert rows[0].constraint_answers is None
    assert rows[2].constraint_answers is None


def test_multilevel_rejects_bad_subset_or_empty_text():
    q = _q(n=4)
    subset, text = _mid_for(q)
    bad = tuple(i for i in range(4) if i not in subset)[:2]
    if choose_target_idx(q.id, 4, seed=42) not in bad:
        with pytest.raises(ValueError):
            build_spec_levels_multilevel(q, seed=42, mid_text=text, mid_subset=bad)
    with pytest.raises(ValueError):
        build_spec_levels_multilevel(q, seed=42, mid_text="  ", mid_subset=subset)
    with pytest.raises(ValueError):
        build_spec_levels_multilevel(
            q, seed=42, mid_text=text, mid_subset=tuple(range(4)))  # not strict


# --------------------------------------------------------------------------- #
# Persistence round-trip                                                       #
# --------------------------------------------------------------------------- #


def test_mid_cache_roundtrip(tmp_path):
    p = tmp_path / "mid.parquet"
    m1 = MidQuestion(question_id="a", seed=42, subset=(0, 2), text="Q mid?",
                     outcome="accepted", attempts=1,
                     judge_admits=(True, False, True))
    m2 = MidQuestion(question_id="b", seed=42, subset=(1, 2), text="",
                     outcome="failed", attempts=4)
    append_mid_cache(p, m1)
    append_mid_cache(p, m2)
    back = load_mid_cache(p)
    assert back["a"] == m1
    assert back["b"] == m2
    assert load_mid_cache(tmp_path / "absent.parquet") == {}


# --------------------------------------------------------------------------- #
# generate_mid_question gate (fake generator + judge)                          #
# --------------------------------------------------------------------------- #


class _ScriptedClient:
    """Yields scripted gen candidates; judges 'yes' iff the reading's year
    digit is in `admit_years` (drives the admits-exactly-subset gate)."""

    def __init__(self, gen_texts: list[str], admit_years: set[str]):
        self.gen_texts = list(gen_texts)
        self.admit_years = admit_years
        self.n_gen_calls = 0

    def complete(self, req):
        from prompt_sensitivity.models.schemas import LLMResponse

        if req.purpose == "midlevel::gen":
            text = self.gen_texts[min(self.n_gen_calls, len(self.gen_texts) - 1)]
            self.n_gen_calls += 1
            return LLMResponse(request_hash="h", text=text)
        # judge: reading is on the "Reading B: ..." line
        reading = next(
            line for line in req.messages[-1].content.splitlines()
            if line.startswith("Reading B:"))
        year_digit = reading.rstrip("?").strip()[-1]
        return LLMResponse(
            request_hash="h",
            text="yes" if year_digit in self.admit_years else "no")


def _gen_setup(monkeypatch, q, seed, gen_texts, admit_exactly_subset):
    from prompt_sensitivity.config import load_config
    from prompt_sensitivity.models import registry
    from prompt_sensitivity.specificity import midlevel

    config = load_config()
    target = choose_target_idx(q.id, q.m0(), seed=seed)
    subset = choose_mid_subset(q.id, q.m0(), target, seed=seed)
    admit_years = ({str(i) for i in subset} if admit_exactly_subset
                   else {str(i) for i in range(q.m0())})
    client = _ScriptedClient(gen_texts, admit_years)
    monkeypatch.setattr(registry, "get_client", lambda *a, **k: client)
    return config, subset, client, midlevel


def test_generate_mid_accepts_gated_candidate(monkeypatch):
    q = _q(n=4)
    config, subset, client, midlevel = _gen_setup(
        monkeypatch, q, 42,
        ["Who won the championship in the early years?"], True)
    mid = midlevel.generate_mid_question(q, seed=42, config=config)
    assert mid.outcome == "accepted"
    assert mid.subset == subset
    assert mid.attempts == 1
    assert mid.judge_admits == tuple(i in set(subset) for i in range(4))


def test_generate_mid_retries_past_leaky_candidate(monkeypatch):
    q = _q(n=4)
    t = choose_target_idx(q.id, 4, seed=42)
    config, subset, client, midlevel = _gen_setup(
        monkeypatch, q, 42,
        [f"Did team-{t} win the championship?",           # leaks an answer
         "Who won the championship in the early years?"], True)
    mid = midlevel.generate_mid_question(q, seed=42, config=config)
    assert mid.outcome == "accepted"
    assert mid.attempts == 2


def test_generate_mid_fails_when_gate_never_passes(monkeypatch):
    q = _q(n=4)
    # judge admits EVERY reading -> candidate admits the excluded ones -> fail
    config, subset, client, midlevel = _gen_setup(
        monkeypatch, q, 42,
        ["Who won the championship in the early years?"], False)
    mid = midlevel.generate_mid_question(q, seed=42, config=config)
    assert mid.outcome == "failed"
    assert mid.text == ""
    assert mid.attempts == 4
