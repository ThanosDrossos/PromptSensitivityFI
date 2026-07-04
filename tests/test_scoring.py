"""NLI-with-gold F(x) — pure-logic tests (no DeBERTa load)."""

from __future__ import annotations

from prompt_sensitivity.scoring import NLIScoreResult, exact_match_score


def test_f_is_one_when_both_checks_pass():
    r = NLIScoreResult(
        entail_prob=0.95, contradict_prob=0.02, neutral_prob=0.03,
        passes_entail=True, passes_contradict=True,
    )
    assert r.f == 1


def test_f_is_zero_when_either_check_fails():
    fail_entail = NLIScoreResult(
        entail_prob=0.4, contradict_prob=0.02, neutral_prob=0.58,
        passes_entail=False, passes_contradict=True,
    )
    fail_contradict = NLIScoreResult(
        entail_prob=0.95, contradict_prob=0.7, neutral_prob=0.0,
        passes_entail=True, passes_contradict=False,
    )
    assert fail_entail.f == 0
    assert fail_contradict.f == 0


def test_exact_match_normalisation():
    """Lowercased + punctuation-stripped + whitespace-collapsed match."""
    assert exact_match_score("Paris", "paris") == 1
    assert exact_match_score("Paris.", "  PARIS") == 1
    assert exact_match_score("Paris", "Paris, France") == 0
    assert exact_match_score("Paris", "London") == 0


def test_exact_match_used_only_as_appendix(monkeypatch):
    """Sanity: exact_match should NEVER be importable as the default F(x).

    `scoring.__init__` exposes both for the appendix sanity check; the
    primary `f_score` uses NLI-with-gold. This test fences the convention.
    """
    from prompt_sensitivity import scoring

    # The primary scorer is NLI-based, not exact-match.
    assert scoring.f_score.__module__.endswith("nli_with_gold")
    # Exact-match is still available for the appendix.
    assert callable(scoring.exact_match_score)


def test_f_score_batch_multi_gold_is_or_over_variants(monkeypatch):
    """AmbigQA pivot §6: F=1 iff the answer passes against ANY gold variant.

    The heavy per-gold scorer is patched with a deterministic fake (exact match
    against that gold), so the OR/elementwise-max logic runs without DeBERTa.
    """
    from prompt_sensitivity.scoring import nli_with_gold as m

    def fake_f_score_batch(gold, answers, *, config=None, permissive=False):
        return [1 if a == gold else 0 for a in answers]

    monkeypatch.setattr(m, "f_score_batch", fake_f_score_batch)

    answers = ["Tony Goldwyn", "Goldwyn", "Michael C. Hall"]
    got = m.f_score_batch_multi_gold(["Tony Goldwyn", "Goldwyn"], answers)
    assert got == [1, 1, 0]           # either variant counts, wrong answer does not

    # empty inputs are safe
    assert m.f_score_batch_multi_gold([], answers) == [0, 0, 0]
    assert m.f_score_batch_multi_gold(["x"], []) == []
    # blank golds are ignored rather than scored
    assert m.f_score_batch_multi_gold(["", "  "], answers) == [0, 0, 0]
