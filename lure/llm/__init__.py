"""The opt-in LLM tier.

Nothing in the deterministic core imports this package at module load, and
nothing here is required to run ``python -m lure.cli run`` or the test suite.
The tier is enabled by credentials and disabled by their absence -- never by
an error.
"""

from lure.llm.client import (
    DEFAULT_CACHE_PATH,
    FakeLLM,
    LLMBackend,
    LLMClient,
    unavailable_message,
)

__all__ = [
    "DEFAULT_CACHE_PATH",
    "FakeLLM",
    "LLMBackend",
    "LLMClient",
    "unavailable_message",
]
