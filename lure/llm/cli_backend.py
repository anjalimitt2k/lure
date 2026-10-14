"""The Claude Code CLI as an LLM backend: no API key, the user's own login.

``claude -p`` answers a prompt headlessly using whatever account the Claude
Code CLI is signed into. For someone who has Claude Code installed but no API
key, that is the difference between the LLM tier being runnable and not.

Everything that makes :class:`~lure.llm.client.LLMClient` safe to put in a
benchmark holds here too:

* **Same cache, same keys.** Responses are stored under the identical
  ``sha256(model + system + prompt)`` scheme, so a cache populated through the
  CLI replays through the SDK client and vice versa.
* **Fail-soft.** A missing binary or a signed-out CLI makes the backend
  *unavailable* with a stated reason; a single failed call is retried once and
  then recorded as "no response", never raised into the run.
* **No tools, no session.** The CLI is invoked with its tool set emptied and
  session persistence off, so the model under test can only read the prompt
  and reply. An adversarial page cannot make it touch the filesystem.

The cost is latency: each call is a fresh process and a fresh request, several
seconds against under one for the SDK. The cache pre-warmer in
:mod:`lure.eval.warm` exists to spend that time concurrently.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from lure.llm.client import DEFAULT_CACHE_PATH, LLMBackend, _JsonCache, cache_key

__all__ = ["CLI_BACKEND_NAME", "ClaudeCLIBackend", "DEFAULT_CLI_MODEL"]

CLI_BACKEND_NAME = "claude-cli"
DEFAULT_CLI_MODEL = "claude-haiku-5-5"

#: Seconds to wait for one reply before treating the call as failed.
DEFAULT_TIMEOUT = 300.0

#: The CLI refuses to start a nested session inside another one. Dropping this
#: variable is what lets the backend run from within Claude Code itself.
_NESTING_GUARD = "CLAUDECODE"

Runner = Callable[[Sequence[str], str, Mapping[str, str], float], "subprocess.CompletedProcess[str]"]


def _default_runner(
    argv: Sequence[str], stdin: str, env: Mapping[str, str], timeout: float
) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # noqa: S603 - argv is built here, never from page text
        list(argv),
        input=stdin,
        capture_output=True,
        text=True,
        env=dict(env),
        timeout=timeout,
        check=False,
    )


class ClaudeCLIBackend(LLMBackend):
    """Cached backend that answers prompts through ``claude -p``."""

    def __init__(
        self,
        model: str = DEFAULT_CLI_MODEL,
        cache_path: Path = DEFAULT_CACHE_PATH,
        executable: str = "claude",
        runner: Runner | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialise with a model and the CLI to drive.

        Args:
            model: Model identifier passed to ``--model``.
            cache_path: Where responses are cached.
            executable: Name or path of the Claude Code CLI.
            runner: Replacement for ``subprocess.run``, for tests.
            timeout: Per-call wall-clock limit in seconds.
        """
        self.model = model
        self._cache = _JsonCache(cache_path)
        self._executable = executable
        self._runner = runner or _default_runner
        self._timeout = timeout
        self._disabled_reason: str | None = None
        self._resolved: str | None = None

    @property
    def available(self) -> bool:
        """Return whether calls can still be attempted."""
        if self._disabled_reason is not None:
            return False
        if self._resolved is None:
            self._resolved = shutil.which(self._executable) or ""
            if not self._resolved:
                self._disable(
                    f"the {self._executable!r} CLI is not on PATH; install Claude Code "
                    "or set ANTHROPIC_API_KEY to use the SDK instead"
                )
                return False
        return True

    def _disable(self, reason: str) -> None:
        """Mark the backend unavailable and explain why, once."""
        if self._disabled_reason is None:
            self._disabled_reason = reason
            sys.stderr.write(f"[lure] LLM tier disabled: {reason}\n")

    def _argv(self, system: str | None) -> list[str]:
        argv = [
            self._resolved or self._executable,
            "-p",
            "--no-session-persistence",
            "--model",
            self.model,
            "--output-format",
            "text",
            "--tools",
            "",
        ]
        if system:
            argv.extend(["--system-prompt", system])
        return argv

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k != _NESTING_GUARD}
        return env

    def _call_once(self, prompt: str, system: str | None) -> str | None:
        """Run the CLI once; return its text, or ``None`` on any failure."""
        try:
            completed = self._runner(self._argv(system), prompt, self._env(), self._timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            sys.stderr.write(f"[lure] claude CLI call failed ({type(exc).__name__}: {exc})\n")
            return None
        text = (completed.stdout or "").strip()
        if completed.returncode != 0 or "Not logged in" in text:
            detail = (completed.stderr or text or "").strip().splitlines()
            if "Not logged in" in text:
                self._disable("the claude CLI is not logged in; run `claude` and /login")
            else:
                sys.stderr.write(
                    f"[lure] claude CLI exited {completed.returncode}"
                    + (f": {detail[-1]}" if detail else "")
                    + "\n"
                )
            return None
        return text

    def complete(self, prompt: str, system: str | None = None) -> str | None:
        """Return the model's response, from cache when possible.

        The cache is consulted before availability, so a fully cached run needs
        no CLI and no login. A live call that fails is retried once; a second
        failure yields ``None`` without disabling the backend, since one bad
        request is not evidence the next will fail.
        """
        key = cache_key(self.model, system, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not self.available:
            return None

        text = self._call_once(prompt, system)
        if text is None and self.available:
            text = self._call_once(prompt, system)
        if text is None:
            return None

        self._cache.put(key, text)
        return text
