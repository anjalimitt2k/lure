"""LLM red-team generation of novel injections.

The fixed attack suite is, by construction, inside the content classifier's
training distribution. This package exists to leave it: a model is asked to
write fresh natural-language injections for the same fraud goals, and those
strings become extra attack cases that no defense here has seen.

Scope is deliberately narrow. Generation happens inside the sandboxed mock
marketplace, targets the scripted benchmark environment only, and exists to
measure and harden defenses.
"""

from lure.redteam.generator import (
    GeneratedInjectionAttack,
    canned_redteam_client,
    generate_injections,
)

__all__ = [
    "GeneratedInjectionAttack",
    "canned_redteam_client",
    "generate_injections",
]
