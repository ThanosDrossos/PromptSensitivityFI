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
    # Sprint 6: the three EVAL models run locally (in-process transformers);
    # gpt_4o stays a (now-unused-on-cluster) gateway entry.
    for k in ("llama_3_1_8b", "mistral_7b_v03", "qwen_2_5_7b"):
        assert cfg.models[k].provider == "local"
    assert cfg.models["gpt_4o"].provider == "litellm"


def test_capability_flags_match_local_backend():
    """Sprint 6 capability matrix (models/local_hf.py).

    In-process transformers exposes echo-style teacher-forced scoring (POSIX)
    AND last-layer hidden states (ESS_in^own) for the three local eval models.
    The legacy gateway gpt_4o entry has neither.
    """
    cfg = load_config()
    for k in ("llama_3_1_8b", "mistral_7b_v03", "qwen_2_5_7b"):
        assert cfg.models[k].echo_completions is True, f"{k} supports exact teacher forcing"
        assert cfg.models[k].has_hidden is True, f"{k} exposes its own hidden states locally"
    assert cfg.models["gpt_4o"].echo_completions is False, "gateway gpt_4o has no echo"
    assert cfg.models["gpt_4o"].has_hidden is False, "gateway gpt_4o has no hidden states"


def test_only_expected_models_are_local():
    """Guard against a future model being added with provider=local / has_hidden
    but without review — both imply the in-process transformers backend."""
    cfg = load_config()
    # 3 eval models + the Phi-4 generator (P3-3) are the expected local models.
    local_expected = {"llama_3_1_8b", "mistral_7b_v03", "qwen_2_5_7b", "phi_4_14b"}
    for k, m in cfg.models.items():
        if m.provider == "local" or m.has_hidden:
            assert k in local_expected, f"{k} unexpectedly marked provider=local/has_hidden=True"


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
