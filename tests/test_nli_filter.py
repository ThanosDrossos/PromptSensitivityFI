"""DeBERTa NLI filter — pure-logic tests + an opt-in live test.

Loading the real DeBERTa weights (~1.6 GB) is too heavy for CI; the live
test is marked `needs_gpu` and skipped unless `RUN_NLI_LIVE=1` is set in
the environment.
"""

from __future__ import annotations

import os

import pytest

from prompt_sensitivity.paraphrases.nli_filter import NLIScores


def test_nli_scores_passes_threshold():
    s = NLIScores(entail_fwd=0.91, entail_bwd=0.93)
    assert s.passes(0.9)
    assert not s.passes(0.92)


def test_nli_scores_one_direction_fails():
    s = NLIScores(entail_fwd=0.95, entail_bwd=0.4)
    assert not s.passes(0.9)


@pytest.mark.needs_gpu
@pytest.mark.skipif(
    os.environ.get("RUN_NLI_LIVE") != "1",
    reason="set RUN_NLI_LIVE=1 to download MoritzLaurer DeBERTa-MNLI and run",
)
def test_live_nli_round_trip():
    """Sanity check the real loader on one trivially-entailing pair."""
    from prompt_sensitivity.paraphrases.nli_filter import score_pair

    s = score_pair("Paris is the capital of France.", "France's capital is Paris.")
    assert s.entail_fwd > 0.7
    assert s.entail_bwd > 0.7
