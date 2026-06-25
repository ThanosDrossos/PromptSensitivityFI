"""P0-2b: f_score_batch permissive mode re-thresholds the same NLI pass."""

from __future__ import annotations

from types import SimpleNamespace

from prompt_sensitivity.config import load_config
from prompt_sensitivity.scoring import nli_with_gold


def test_permissive_catches_midrange_entailment(monkeypatch):
    # entail probs: 0.55 (mid), 0.80 (high), 0.30 (low); contradiction all low.
    fake = [
        SimpleNamespace(entail_prob=e, contradict_prob=0.1, f=1 if e >= 0.7 else 0)
        for e in (0.55, 0.80, 0.30)
    ]
    monkeypatch.setattr(
        nli_with_gold, "score_batch_nli_with_gold", lambda g, a, *, config=None: fake
    )
    cfg = load_config()
    strict = nli_with_gold.f_score_batch("gold", ["a", "b", "c"], config=cfg)
    perm = nli_with_gold.f_score_batch("gold", ["a", "b", "c"], config=cfg, permissive=True)

    assert strict == [0, 1, 0]   # only 0.80 clears the strict 0.7 bar
    assert perm == [1, 1, 0]     # 0.55 and 0.80 clear the permissive 0.5 bar


def test_permissive_still_gated_by_contradiction(monkeypatch):
    # High entailment but also high contradiction -> must fail even permissively.
    fake = [SimpleNamespace(entail_prob=0.9, contradict_prob=0.8, f=0)]
    monkeypatch.setattr(
        nli_with_gold, "score_batch_nli_with_gold", lambda g, a, *, config=None: fake
    )
    cfg = load_config()
    assert nli_with_gold.f_score_batch("g", ["x"], config=cfg, permissive=True) == [0]
