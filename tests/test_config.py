"""Verify config.yaml loads cleanly and the expected fields are present."""

from prompt_sensitivity.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.config_version >= 1
    assert cfg.random_seed == 42


def test_models_registered():
    cfg = load_config()
    expected = {"llama_3_1_8b", "mistral_7b_v03", "qwen_2_5_7b", "gpt_4o"}
    assert expected <= set(cfg.models.keys())
    # Pre-cluster, every model routes through the LiteLLM gateway.
    for k in expected:
        assert cfg.models[k].provider == "litellm"


def test_capability_flags_match_gateway_matrix():
    """Encodes the capability matrix from registry.py.

    Open-weight Together models support echo (POSIX); GPT-4o does not.
    No model exposes its own hidden state through the gateway (cluster-only).
    """
    cfg = load_config()
    for k in ("llama_3_1_8b", "mistral_7b_v03", "qwen_2_5_7b"):
        assert cfg.models[k].echo_completions is True, f"{k} should support echo via gateway"
    assert cfg.models["gpt_4o"].echo_completions is False, "GPT-4o has no echo on chat models"
    for k in cfg.models:
        assert cfg.models[k].has_hidden is False, f"{k} hidden states are cluster-only"


def test_api_routes_through_litellm():
    cfg = load_config()
    assert cfg.api.api_key_env == "LITELLM_API_KEY"
    assert cfg.api.base_url_env == "LITELLM_BASE_URL"
    assert cfg.api.default_base_url.startswith("https://"), "gateway URL must be https"


def test_ladder_levels_match_design():
    cfg = load_config()
    # Research_Design_v3 §4.2: levels are paragraph counts {0, 2, 4, 6, 8, 10}.
    assert cfg.ladders.levels == [0, 2, 4, 6, 8, 10]
    assert cfg.ladders.k_gold == 2
    assert cfg.ladders.n_total_paragraphs == 10
    assert set(cfg.ladders.variants) == {"random", "gold_first", "distractor_first"}


def test_scoring_uses_nli_not_exact_match():
    """Anti-pattern: F(x) MUST be NLI-with-gold (Hua 2025 EMNLP)."""
    cfg = load_config()
    assert cfg.scoring.method == "nli_with_gold"
    assert cfg.scoring.exact_match_appendix_only is True
