"""A thin, cached, fail-soft wrapper over the Anthropic SDK.

Three properties matter here, and each exists to protect a benchmark property
rather than for convenience:

* **Fail-soft.** Missing credentials, a missing SDK, or an API error make the
  tier *unavailable*; they never raise into the benchmark. An offline user
  must get a clear message and a clean exit, not a stack trace.
* **Cached.** Every response is stored under ``results/llm_cache.json`` keyed
  by ``sha256(model + system + prompt)``. A cache hit is served before
  availability is even considered, so a populated cache makes a run free,
  network-free and exactly reproducible -- which is what makes a published LLM
  number checkable by someone else.
* **temperature=0.** Sampling variance would turn a measured attack success
  rate into a noisy estimate of one.

:class:`FakeLLM` implements the identical interface with canned responses and
is what the entire test suite uses. No test in this repository can reach the
network, whether or not credentials happen to be present in the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Mapping

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_CACHE_PATH",
    "FakeLLM",
    "LLMBackend",
    "LLMClient",
    "MODEL_ENV",
    "unavailable_message",
]

API_KEY_ENV = "ANTHROPIC_API_KEY"
MODEL_ENV = "LURE_MODEL"

#: Cache lives beside the metrics so a published result and the responses that
#: produced it travel together.
DEFAULT_CACHE_PATH = Path("results") / "llm_cache.json"

#: Suggested models. Haiku is the high-volume default -- the red-team and agent
#: prompts are short, and a full LLM-tier run is many small calls.
SUGGESTED_MODELS = ("claude-haiku-4-5-20251001", "claude-sonnet-5")


def unavailable_message() -> str:
    """Return the operator-facing explanation of how to enable the LLM tier."""
    return (
        "The LLM tier is not enabled, so there is nothing to run.\n"
        "\n"
        "It is opt-in because it costs money and needs network access. The\n"
        "offline benchmark (`python -m lure.cli run`) is unaffected and needs\n"
        "neither.\n"
        "\n"
        "To enable it:\n"
        f"  export {API_KEY_ENV}=sk-ant-...\n"
        f"  export {MODEL_ENV}={SUGGESTED_MODELS[0]}\n"
        "  pip install -e '.[llm]'\n"
        "\n"
        f"Suggested models: {SUGGESTED_MODELS[0]} (cheap, the high-volume\n"
        f"default) or {SUGGESTED_MODELS[1]} (stronger, more expensive).\n"
        "\n"
        f"Responses are cached in {DEFAULT_CACHE_PATH}, so a second run over the\n"
        "same tasks costs nothing and needs no network."
    )


class LLMBackend(ABC):
    """Interface shared by the real client and the test stub."""

    #: Model identifier, reported in output.
    model: str = ""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Return whether calls can currently be served."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str | None:
        """Return the model's text response, or ``None`` if it could not be obtained.

        Returning ``None`` rather than raising is the contract: callers treat a
        missing response as a task that could not be scored, not as a crash.
        """
        raise NotImplementedError


class _JsonCache:
    """A small JSON-file cache of prompt hashes to responses."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        """Return the cache contents, reading from disk once.

        A corrupt or unreadable cache is treated as empty. A cache is an
        optimisation; letting it break a run would be worse than a cache miss.
        """
        if self._entries is None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._entries = {
                    str(k): str(v) for k, v in raw.items() if isinstance(v, str)
                }
            except (OSError, ValueError):
                self._entries = {}
        return self._entries

    def get(self, key: str) -> str | None:
        """Return the cached response for ``key``, if any."""
        return self._load().get(key)

    def put(self, key: str, value: str) -> None:
        """Store ``value`` under ``key`` and persist, ignoring write failures."""
        entries = self._load()
        entries[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError:
            return


class LLMClient(LLMBackend):
    """Cached Anthropic client that degrades to unavailable instead of raising."""

    def __init__(
        self,
        model: str,
        api_key: str,
        cache_path: Path = DEFAULT_CACHE_PATH,
        max_tokens: int = 1024,
    ) -> None:
        """Initialise with an explicit model and key.

        Args:
            model: Model identifier to call.
            api_key: Anthropic API key.
            cache_path: Where responses are cached.
            max_tokens: Response ceiling. The agent returns a small JSON object
                and the red-teamer a short list, so this is generous.
        """
        self.model = model
        self._api_key = api_key
        self._cache = _JsonCache(cache_path)
        self._max_tokens = max_tokens
        self._sdk_client: object | None = None
        self._disabled_reason: str | None = None

    @classmethod
    def from_env(
        cls,
        cache_path: Path = DEFAULT_CACHE_PATH,
        environ: Mapping[str, str] | None = None,
    ) -> "LLMClient | None":
        """Build a client from the environment, or ``None`` if not configured.

        Both variables are required. Returning ``None`` rather than a
        half-configured client keeps "is the tier on?" a single, honest check
        at the call site.
        """
        env = os.environ if environ is None else environ
        api_key = env.get(API_KEY_ENV, "").strip()
        model = env.get(MODEL_ENV, "").strip()
        if not api_key or not model:
            return None
        return cls(model=model, api_key=api_key, cache_path=cache_path)

    @property
    def available(self) -> bool:
        """Return whether live calls can still be attempted."""
        return self._disabled_reason is None

    def _client(self) -> object | None:
        """Return a lazily constructed SDK client, or ``None`` if unusable.

        Imported here rather than at module scope so the offline core never
        depends on the SDK being installed.
        """
        if self._sdk_client is not None:
            return self._sdk_client
        try:
            import anthropic
        except ImportError:
            self._disable("the `anthropic` package is not installed")
            return None
        try:
            self._sdk_client = anthropic.Anthropic(api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001 - any SDK failure disables the tier
            self._disable(f"could not create an Anthropic client: {exc}")
            return None
        return self._sdk_client

    def _disable(self, reason: str) -> None:
        """Mark the tier unavailable and explain why, once."""
        if self._disabled_reason is None:
            self._disabled_reason = reason
            sys.stderr.write(f"[lure] LLM tier disabled: {reason}\n")

    def _cache_key(self, system: str | None, prompt: str) -> str:
        """Return the cache key for one request."""
        payload = "\x00".join([self.model, system or "", prompt])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, prompt: str, system: str | None = None) -> str | None:
        """Return the model's response, from cache when possible.

        The cache is consulted before availability, so a fully cached run needs
        no key, no SDK and no network.
        """
        key = self._cache_key(system, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if not self.available:
            return None
        client = self._client()
        if client is None:
            return None

        try:
            response = client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=self._max_tokens,
                temperature=0,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
        except Exception as exc:  # noqa: BLE001 - any API failure disables the tier
            self._disable(f"API call failed ({type(exc).__name__}: {exc})")
            return None

        self._cache.put(key, text)
        return text


class FakeLLM(LLMBackend):
    """Deterministic in-memory stub with the same interface as the real client.

    Used by every test, so the suite is hermetic regardless of what is set in
    the developer's environment.
    """

    model = "fake-model"

    def __init__(
        self,
        responses: Mapping[str, str]
        | Callable[[str, str | None], str | None]
        | str
        | None,
        available: bool = True,
    ) -> None:
        """Initialise with canned behaviour.

        Args:
            responses: A single string returned for every prompt, a mapping
                from substring-of-prompt to response, a callable taking
                ``(prompt, system)``, or ``None`` to model a backend that
                returns nothing -- the case callers must survive.
            available: Whether this stub reports itself as usable, so callers'
                unavailable paths are testable too.
        """
        self._responses = responses
        self._available = available
        self.calls: list[tuple[str, str | None]] = []

    @property
    def available(self) -> bool:
        """Return the configured availability."""
        return self._available

    def complete(self, prompt: str, system: str | None = None) -> str | None:
        """Return the canned response for ``prompt`` and record the call."""
        self.calls.append((prompt, system))
        if not self._available or self._responses is None:
            return None
        if callable(self._responses):
            return self._responses(prompt, system)
        if isinstance(self._responses, str):
            return self._responses
        for needle, response in self._responses.items():
            if needle in prompt:
                return response
        return None
