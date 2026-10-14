"""Tests for the LLM client wrapper.

No test here touches the network. The real client is exercised only through
its cache, its environment handling, and injected failures -- which is the
whole surface that matters for "the tier degrades gracefully".
"""

from __future__ import annotations

import json

import pytest

from lure.llm.client import (
    API_KEY_ENV,
    MODEL_ENV,
    FakeLLM,
    LLMClient,
    unavailable_message,
)


# -- from_env --------------------------------------------------------------


def test_from_env_returns_none_without_credentials() -> None:
    assert LLMClient.from_env(environ={}) is None


def test_from_env_returns_none_with_only_a_key() -> None:
    assert LLMClient.from_env(environ={API_KEY_ENV: "sk-test"}) is None


def test_from_env_returns_none_with_only_a_model() -> None:
    assert LLMClient.from_env(environ={MODEL_ENV: "claude-haiku-5-5"}) is None


def test_from_env_ignores_blank_values() -> None:
    assert LLMClient.from_env(environ={API_KEY_ENV: "  ", MODEL_ENV: "  "}) is None


def test_from_env_builds_a_client_when_both_are_set(tmp_path) -> None:
    client = LLMClient.from_env(
        cache_path=tmp_path / "cache.json",
        environ={API_KEY_ENV: "sk-test", MODEL_ENV: "claude-haiku-5-5"},
    )

    assert client is not None
    assert client.model == "claude-haiku-5-5"
    assert client.available is True


def test_unavailable_message_explains_both_variables() -> None:
    message = unavailable_message()

    assert API_KEY_ENV in message
    assert MODEL_ENV in message
    assert "claude-haiku-5-5" in message


# -- caching ---------------------------------------------------------------


def _cache_key(client: LLMClient, prompt: str, system: str | None) -> str:
    return client._cache_key(system, prompt)


def test_a_cache_hit_needs_no_network_or_credentials(tmp_path) -> None:
    """The property that makes a published LLM number reproducible offline."""
    cache_path = tmp_path / "cache.json"
    client = LLMClient(model="m", api_key="", cache_path=cache_path)
    key = _cache_key(client, "hello", "sys")
    cache_path.write_text(json.dumps({key: "cached reply"}), encoding="utf-8")

    assert client.complete("hello", system="sys") == "cached reply"


def test_cache_keys_separate_model_system_and_prompt(tmp_path) -> None:
    a = LLMClient(model="m1", api_key="k", cache_path=tmp_path / "c.json")
    b = LLMClient(model="m2", api_key="k", cache_path=tmp_path / "c.json")

    assert _cache_key(a, "p", "s") != _cache_key(b, "p", "s")
    assert _cache_key(a, "p", "s") != _cache_key(a, "p", "other")
    assert _cache_key(a, "p", "s") != _cache_key(a, "other", "s")
    assert _cache_key(a, "p", "s") == _cache_key(a, "p", "s")


def test_a_corrupt_cache_is_treated_as_empty(tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{not json", encoding="utf-8")
    client = LLMClient(model="m", api_key="", cache_path=cache_path)

    # Falls through to the live path, which is disabled without a key.
    client._disable("test")
    assert client.complete("hello") is None


def test_a_successful_call_is_written_to_the_cache(tmp_path) -> None:
    cache_path = tmp_path / "nested" / "cache.json"
    client = LLMClient(model="m", api_key="k", cache_path=cache_path)
    client._sdk_client = _StubSDK("live reply")

    assert client.complete("hello") == "live reply"
    assert client.complete("hello") == "live reply"
    assert cache_path.exists()
    # Second call was served from cache, so the SDK saw exactly one request.
    assert client._sdk_client.calls == 1


# -- graceful failure ------------------------------------------------------


class _StubBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _StubResponse:
    def __init__(self, text: str) -> None:
        self.content = [_StubBlock(text)]


class _StubMessages:
    def __init__(self, outer: "_StubSDK") -> None:
        self._outer = outer

    def create(self, **kwargs: object) -> _StubResponse:
        self._outer.calls += 1
        self._outer.last_kwargs = kwargs
        if self._outer.error is not None:
            raise self._outer.error
        return _StubResponse(self._outer.reply)


class _StubSDK:
    """Stands in for ``anthropic.Anthropic`` without importing or calling it."""

    def __init__(self, reply: str = "ok", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}
        self.messages = _StubMessages(self)


def test_an_api_error_disables_the_tier_instead_of_raising(tmp_path, capsys) -> None:
    client = LLMClient(model="m", api_key="k", cache_path=tmp_path / "c.json")
    client._sdk_client = _StubSDK(error=RuntimeError("boom"))

    assert client.complete("hello") is None
    assert client.available is False
    assert "disabled" in capsys.readouterr().err


def test_a_disabled_client_stops_calling_out(tmp_path) -> None:
    client = LLMClient(model="m", api_key="k", cache_path=tmp_path / "c.json")
    stub = _StubSDK(error=RuntimeError("boom"))
    client._sdk_client = stub

    client.complete("one")
    client.complete("two")

    assert stub.calls == 1


def test_requests_send_no_sampling_parameters(tmp_path) -> None:
    """Current models reject non-default temperature; determinism is the cache's job."""
    client = LLMClient(model="m", api_key="k", cache_path=tmp_path / "c.json")
    stub = _StubSDK("ok")
    client._sdk_client = stub

    client.complete("hello", system="sys")

    assert "temperature" not in stub.last_kwargs
    assert "top_p" not in stub.last_kwargs
    assert stub.last_kwargs["model"] == "m"
    assert stub.last_kwargs["system"] == "sys"


def test_requests_omit_an_empty_system_prompt(tmp_path) -> None:
    client = LLMClient(model="m", api_key="k", cache_path=tmp_path / "c.json")
    stub = _StubSDK("ok")
    client._sdk_client = stub

    client.complete("hello")

    assert "system" not in stub.last_kwargs


def test_a_refusal_is_cached_as_an_empty_reply(tmp_path) -> None:
    """Declined prompts are a stable state of the model, not a transient failure."""
    client = LLMClient(model="m", api_key="k", cache_path=tmp_path / "c.json")
    stub = _StubSDK("should not be read")
    client._sdk_client = stub
    stub.messages.create = lambda **kwargs: _RefusedResponse()  # type: ignore[method-assign]

    assert client.complete("hello") == ""
    assert client.available is True
    assert client.complete("hello") == ""


class _RefusedResponse(_StubResponse):
    def __init__(self) -> None:
        super().__init__("declined text that must be ignored")
        self.stop_reason = "refusal"


# -- FakeLLM ---------------------------------------------------------------


def test_fake_llm_returns_a_constant_response() -> None:
    fake = FakeLLM("always this")

    assert fake.complete("anything") == "always this"
    assert fake.available is True


def test_fake_llm_matches_on_prompt_substrings() -> None:
    fake = FakeLLM({"mug": "mug reply", "lamp": "lamp reply"})

    assert fake.complete("buy a mug please") == "mug reply"
    assert fake.complete("buy a lamp please") == "lamp reply"
    assert fake.complete("buy a hovercraft") is None


def test_fake_llm_supports_a_callable() -> None:
    fake = FakeLLM(lambda prompt, system: prompt.upper())

    assert fake.complete("abc") == "ABC"


def test_fake_llm_can_report_itself_unavailable() -> None:
    fake = FakeLLM("x", available=False)

    assert fake.available is False
    assert fake.complete("anything") is None


def test_fake_llm_records_calls() -> None:
    fake = FakeLLM("x")
    fake.complete("prompt one", system="sys")

    assert fake.calls == [("prompt one", "sys")]
