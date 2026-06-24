"""Tests for the stratified MuSiQue sampler used by the full cluster run.

The sampler MUST be deterministic for a fixed seed: the paraphrase-prep job and
all 3 per-model array tasks call it independently and have to select the exact
same question set (shared paraphrase universe).
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import prompt_sensitivity.scripts.e2e_smoke as e2e


def _fake_questions():
    qs = []
    for hops in (2, 3, 4):
        for i in range(20):
            qs.append(SimpleNamespace(id=f"{hops}hop_{i:02d}", n_hops=hops))
    return qs


def test_stratified_balanced_and_deterministic(monkeypatch):
    monkeypatch.setattr(e2e, "_load_musique", lambda cfg: _fake_questions())
    a = e2e._pick_musique_questions_stratified(None, 5, seed=42)
    b = e2e._pick_musique_questions_stratified(None, 5, seed=42)

    assert len(a) == 15  # 5 per stratum x 3 strata
    assert Counter(q.n_hops for q in a) == {2: 5, 3: 5, 4: 5}
    # identical selection AND order across calls with the same seed
    assert [q.id for q in a] == [q.id for q in b]


def test_stratified_seed_changes_selection(monkeypatch):
    monkeypatch.setattr(e2e, "_load_musique", lambda cfg: _fake_questions())
    a = e2e._pick_musique_questions_stratified(None, 5, seed=1)
    b = e2e._pick_musique_questions_stratified(None, 5, seed=2)
    assert [q.id for q in a] != [q.id for q in b]


def test_stratified_caps_at_available(monkeypatch):
    # Asking for more than a stratum holds returns all of it, no error.
    monkeypatch.setattr(e2e, "_load_musique", lambda cfg: _fake_questions())
    out = e2e._pick_musique_questions_stratified(None, 999, seed=42)
    assert Counter(q.n_hops for q in out) == {2: 20, 3: 20, 4: 20}
