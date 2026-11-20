"""GD3: config.py - defaults, environment overrides, validation, hash."""
import pytest

from config import Config, load_config


def test_defaults_match_spec():
    c = load_config(env={})
    assert c.model == "qwen2.5-coder:14b" and c.judge_model == "qwen2.5:14b-instruct"
    assert c.sandbox_mode == "docker"
    assert c.max_retries == 3 and c.num_ctx == 8192 and c.seed == 42


def test_env_overrides_are_typed():
    c = load_config(env={"EDA_OLLAMA_URL": "https://abc.trycloudflare.com", "EDA_SANDBOX_MODE": "subprocess",
                         "EDA_MAX_RETRIES": "2", "EDA_TEMPERATURE": "0.0", "EDA_REFLEXION": "false",
                         "EDA_PROFILING": "1", "EDA_MODEL": ""})
    assert c.ollama_url == "https://abc.trycloudflare.com"
    assert c.sandbox_mode == "subprocess"
    assert c.max_retries == 2 and isinstance(c.max_retries, int)
    assert c.temperature == 0.0 and c.reflexion is False and c.profiling is True
    assert c.model == "qwen2.5-coder:14b"  # empty env value = not set


def test_explicit_overrides_win_over_env():
    c = load_config(env={"EDA_MAX_STEPS": "5"}, max_steps=7)
    assert c.max_steps == 7


@pytest.mark.parametrize("env", [
    {"EDA_SANDBOX_MODE": "e2b"},
    {"EDA_OLLAMA_URL": "localhost:11434"},
    {"EDA_REFLEXION": "maybe"},
    {"EDA_MAX_RETRIES": "-1"},
])
def test_invalid_values_raise(env):
    with pytest.raises(ValueError):
        load_config(env=env)


def test_config_hash_is_stable_and_sensitive():
    a, b = load_config(env={}), load_config(env={})
    assert a.config_hash() == b.config_hash() and len(a.config_hash()) == 12
    assert a.replace(reflexion=False).config_hash() != a.config_hash()
    assert isinstance(a.to_dict(), dict) and isinstance(a, Config)
