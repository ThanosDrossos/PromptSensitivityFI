"""R6 generator-width dial: the contracts that make the positive control valid.

What matters: (a) arm specs are internally consistent and width-ordered by
design, (b) the generation overrides actually reach the LLM request (a silent
fallback to production settings would make all arms identical and the dial
vacuous), (c) arm caches are separated, (d) the manipulation-check statistics
behave, (e) cache keys differ across arms by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prompt_sensitivity.paraphrases.prompts import (
    ROLE_NAMES,
    WIDTH_ARMS,
    _PERSONA,
    build_paraphrase_messages,
    resolve_width_arm,
)
from prompt_sensitivity.scripts.width_dial_analysis import (
    paired_arm_test,
    universe_width,
)


# ---------------------------------------------------------------- arm registry


def test_all_arms_resolve_and_reference_real_personas():
    for arm in WIDTH_ARMS:
        spec = resolve_width_arm(arm)
        assert spec["roles"], arm
        for r in spec["roles"]:
            assert r in _PERSONA
            msgs = build_paraphrase_messages("Where was it filmed?", r)
            assert len(msgs) == 2 and msgs[0].role == "system"


def test_unknown_arm_and_broken_arm_raise():
    with pytest.raises(ValueError):
        resolve_width_arm("gigantic")


def test_medium_arm_is_exactly_the_production_configuration():
    """The middle dial point must reuse the v3 universes bit-for-bit: same
    roles, same temperature, same generator, same cache path."""
    spec = resolve_width_arm("medium")
    assert spec["roles"] == list(ROLE_NAMES)
    assert spec["temperature"] == pytest.approx(0.8)
    assert spec["generator_model"] is None and spec["judge_model"] is None
    assert spec["cache"] == "data/paraphrases_ambigqa.parquet"


def test_width_is_ordered_by_design():
    """Temperature ordering + persona-count sanity; realized width is checked
    empirically by the P0 manipulation check, but the DESIGN must already
    order narrow < medium <= wide."""
    n, m, w = (resolve_width_arm(a) for a in ("narrow", "medium", "wide"))
    assert n["temperature"] < m["temperature"] <= w["temperature"]
    assert set(n["roles"]).isdisjoint(w["roles"])
    assert all(r.startswith("minimal_") for r in n["roles"])


def test_swap_arm_swaps_generator_and_judge_together():
    """judge == generator is the production rule; splitting them would OOM the
    40 GB A100 (OLMo + Phi-4). If someone edits one field without the other,
    this test fails before the cluster does."""
    spec = resolve_width_arm("swap")
    assert spec["generator_model"] == spec["judge_model"] == "olmo_2_13b"
    assert spec["roles"] == list(ROLE_NAMES)
    assert spec["temperature"] == pytest.approx(0.8)


def test_arm_caches_are_pairwise_distinct():
    caches = [resolve_width_arm(a)["cache"] for a in WIDTH_ARMS]
    assert len(set(caches)) == len(caches)


# ------------------------------------------------------- override plumbing


def test_generation_overrides_reach_the_llm_request(monkeypatch):
    """The dial IS the override: if temperature/model silently fall back to
    config, every arm generates identical universes (cache keys collide) and
    the experiment is vacuous. Capture the actual LLMRequests."""
    from prompt_sensitivity.paraphrases import generate as gen

    captured = []

    class _FakeResp:
        text = "Where was the show filmed?"
        request_hash = "x"

    class _FakeClient:
        def complete(self, req):
            captured.append(req)
            return _FakeResp()

    class _Entry:
        provider = "local"
        model_id = "fake/olmo"

    class _Models(dict):
        pass

    class _PCfg:
        generator_model = "phi_4_14b"
        generator_temperature = 0.8
        templates = list(ROLE_NAMES)
        samples_per_template = 2

    class _Cfg:
        paraphrases = _PCfg()
        models = {"phi_4_14b": _Entry(), "olmo_2_13b": _Entry()}

    monkeypatch.setattr(gen, "get_client", lambda key, cfg: _FakeClient())
    out = gen.generate_raw_paraphrases(
        "q1", "Where was it filmed?", config=_Cfg(),
        sample_idxs=[0], roles=["minimal_synonym"],
        temperature=0.5, generator_model="olmo_2_13b",
    )
    assert len(out) == 1
    req = captured[0]
    assert req.temperature == pytest.approx(0.5), "temperature override must reach the request"
    assert out[0].generator_model_key == "olmo_2_13b"
    # and the default path stays byte-identical to production
    captured.clear()
    gen.generate_raw_paraphrases("q1", "Where was it filmed?", config=_Cfg(),
                                 sample_idxs=[0], roles=["neutral"])
    assert captured[0].temperature == pytest.approx(0.8)


def test_arm_requests_can_never_hit_each_others_cache(monkeypatch):
    """Cache-key separation by construction: same question, different arm
    (temperature or persona) => different request hash."""
    from prompt_sensitivity.models.schemas import LLMRequest

    def req(role_text, temp):
        return LLMRequest(provider="local", model_id="microsoft/phi-4",
                          messages=[{"role": "user", "content": role_text}],
                          temperature=temp, top_p=1.0, max_tokens=128,
                          seed=1, purpose="paraphrase_gen")

    base = req("persona A rewrite", 0.8)
    assert req("persona A rewrite", 0.5).cache_key() != base.cache_key()
    assert req("persona B rewrite", 0.8).cache_key() != base.cache_key()


# ------------------------------------------------------- analysis helpers


def test_universe_width_orders_narrow_vs_wide_texts():
    narrow = ["Where was Top of the Lake filmed?",
              "Where was Top of the Lake shot?",
              "Where did they film Top of the Lake?"]
    wide = ["Where was Top of the Lake filmed?",
            "I was wondering if you could tell me the place where the series "
            "Top of the Lake was actually filmed.",
            "Top of the Lake — filming location?"]
    wn = universe_width(narrow)["pairwise_token_dist"]
    ww = universe_width(wide)["pairwise_token_dist"]
    assert ww > wn
    assert np.isnan(universe_width(["single"])["pairwise_token_dist"])


def test_paired_arm_test_detects_a_monotone_dial():
    rng = np.random.default_rng(0)
    rows = []
    for q in range(40):
        base = rng.uniform(0.01, 0.05)
        for lvl in (0, 1):
            for arm, bump in [("narrow", 0.0), ("medium", 0.02), ("wide", 0.05)]:
                rows.append({"question_id": f"q{q}", "spec_level": lvl, "arm": arm,
                             "sigma2_B": base + bump + rng.normal(scale=0.004)})
    t = paired_arm_test(pd.DataFrame(rows), "sigma2_B")
    assert t["n_paired"] == 80
    assert t["wilcoxon_nw_less_p"] < 1e-6
    assert t["means"][0] < t["means"][1] < t["means"][2]


def test_paired_arm_test_reports_null_when_dial_does_nothing():
    rng = np.random.default_rng(1)
    rows = [{"question_id": f"q{q}", "spec_level": lvl, "arm": arm,
             "sigma2_B": rng.uniform(0.01, 0.05)}
            for q in range(40) for lvl in (0, 1)
            for arm in ("narrow", "medium", "wide")]
    t = paired_arm_test(pd.DataFrame(rows), "sigma2_B")
    assert t["wilcoxon_nw_less_p"] > 0.05
