"""The interface every agent under test implements.

Keeping this surface tiny is what makes Lure a benchmark rather than a
harness for one agent: an implementation receives a user-authorized Intent
Mandate, an environment to act in, and a defense that adjudicates its final
payment. Whatever happens in between -- a scripted policy in Phase 1, a real
model with tool calls in Phase 2 -- is opaque to the scorer, which reads only
the resulting :class:`~lure.environment.models.CheckoutResult`.

Agents are required to route their payment through
``defense.validate(...)`` and to honour a refusal. An agent that bypasses the
defense is not measuring the defense.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lure.defenses.base import Defense
from lure.environment.marketplace import Marketplace
from lure.environment.models import CheckoutResult, IntentMandate

__all__ = ["Agent"]


class Agent(ABC):
    """Base class for any shopping agent evaluated by Lure."""

    #: Human-readable name used in reports.
    name: str = "agent"

    @abstractmethod
    def run(
        self,
        intent: IntentMandate,
        marketplace: Marketplace,
        defense: Defense,
    ) -> CheckoutResult:
        """Attempt to fulfil ``intent`` in ``marketplace``, adjudicated by ``defense``.

        Args:
            intent: The user-authorized mandate. Ground truth for scoring.
            marketplace: The environment to search, read and check out in. Its
                pages may already carry injected content.
            defense: Adjudicator consulted with the final payment. Implementations
                must call it and must not execute a payment it refuses.

        Returns:
            The outcome, executed or blocked.
        """
        raise NotImplementedError
