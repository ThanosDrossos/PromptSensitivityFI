"""P3-3b: local_check gates capability checks by per-model flags."""

from __future__ import annotations

from prompt_sensitivity.scripts.local_check import LocalCheckRow, _okstr


def test_generator_passes_on_generate_alone():
    # Phi-4-like generator: capability flags off -> checks are n/a (None), not failures.
    r = LocalCheckRow("phi_4_14b", "microsoft/phi-4", generate_ok=True,
                      logprobs_ok=None, score_ok=None, hidden_ok=None,
                      hidden_dim=None, text="ok")
    assert r.passed is True


def test_required_capability_false_fails():
    r = LocalCheckRow("llama_3_1_8b", "x", generate_ok=True, logprobs_ok=True,
                      score_ok=False, hidden_ok=True, hidden_dim=4096, text="ok")
    assert r.passed is False


def test_generate_failure_always_fails():
    r = LocalCheckRow("m", "x", generate_ok=False, logprobs_ok=None,
                      score_ok=None, hidden_ok=None, hidden_dim=None, text="")
    assert r.passed is False


def test_okstr():
    assert _okstr(None) == "n/a"
    assert _okstr(True) == "yes"
    assert _okstr(False) == "NO"
