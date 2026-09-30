"""Which backend gets chosen, and why.

The default is token-driven, so the interesting cases are all about the
environment. Every one of them injects an env mapping rather than touching the
ambient environment — a test that depends on whether the developer happens to
have BODHAN_API_KEY exported is a test that passes on one machine.
"""

from __future__ import annotations

import pytest

from bodhan_genai.mt.engine.backends import (
    API_KEY_ENV_VARS,
    BACKEND_NAMES,
    find_api_key,
    resolve_backend_name,
)


def test_no_token_falls_back_to_vllm():
    assert resolve_backend_name(env={}) == "vllm"


@pytest.mark.parametrize("var", API_KEY_ENV_VARS)
def test_either_env_var_selects_the_hosted_api(var):
    """Both spellings work. Silently falling back to vLLM because someone
    exported the other one reads as a missing GPU, not a missing variable."""
    assert resolve_backend_name(env={var: "sk-live-xyz"}) == "api"


def test_both_env_vars_set_is_still_api():
    assert resolve_backend_name(env=dict.fromkeys(API_KEY_ENV_VARS, "k")) == "api"


def test_first_env_var_wins_when_both_are_set():
    env = {"BODHAN_API_KEY": "first", "BODHAN_API_TOKEN": "second"}
    assert find_api_key(env) == "first"


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_an_exported_but_empty_token_is_not_a_token(value):
    """The CI footgun this function exists for.

    `export BODHAN_API_KEY=` in a job would otherwise select the hosted backend
    and then fail every request on a 401, which reads as a broken endpoint rather
    than an unset secret.
    """
    assert find_api_key({"BODHAN_API_KEY": value}) is None
    assert resolve_backend_name(env={"BODHAN_API_KEY": value}) == "vllm"


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_an_explicit_backend_always_wins(name):
    """Naming a backend must not be overridden by an ambient token."""
    env = dict.fromkeys(API_KEY_ENV_VARS, "sk-live-xyz")
    assert resolve_backend_name(name, env=env) == name


def test_unknown_backend_name_is_rejected_and_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown backend") as excinfo:
        resolve_backend_name("tensorrt", env={})
    for name in BACKEND_NAMES:
        assert name in str(excinfo.value)


def test_api_key_is_stripped():
    assert find_api_key({"BODHAN_API_KEY": "  sk-live-xyz \n"}) == "sk-live-xyz"
