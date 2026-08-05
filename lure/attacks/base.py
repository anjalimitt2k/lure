"""The interface every attack implements.

An attack has three parts, and keeping them separate is what makes the
benchmark honest:

* A **payload**, rendered in one of two styles. ``style="structured"`` emits
  the ``[[INSTRUCTION]] key=value`` token that
  :class:`~lure.agent.mock_agent.MockVulnerableAgent` obeys; ``style="natural"``
  emits ordinary prose aimed at a real language model. Same fraud, two
  registers.
* :meth:`Attack.apply` writes that payload into a page and returns a *copy*, so
  a task's staged page can never leak into another task's environment.
* :meth:`Attack.goal_achieved` scores the outcome against the attacker's own
  objective, stated purely in terms of what the executed transaction did.

That last point carries the design. Goals are **outcome-based and
method-agnostic**: they ask what money moved and where the goods went, never
which phrasing was used or whether the agent "noticed" anything. So the same
attack class scores the scripted agent and the LLM agent on one axis, and a
defense gets no credit for recognising a template it happens to have been
trained on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from lure.environment.models import CheckoutResult, IntentMandate, Page

__all__ = ["Attack", "InjectionStyle", "Surface"]

InjectionStyle = Literal["structured", "natural"]

#: Where on the page a payload is written. Reviews are the more realistic
#: surface (no merchant fully controls them); descriptions are the more direct.
Surface = Literal["description", "review"]


class Attack(ABC):
    """Base class for page-content attacks."""

    #: Human-readable attack-class name used in reports and CSV rows.
    name: str = "attack"

    #: Page field this attack writes to.
    surface: Surface = "description"

    # -- payloads ----------------------------------------------------------

    @abstractmethod
    def structured_payload(self) -> str:
        """Return the ``[[INSTRUCTION]]``-token form of this attack."""
        raise NotImplementedError

    @abstractmethod
    def natural_payload(self) -> str:
        """Return the natural-language prose form of the same fraud."""
        raise NotImplementedError

    def payload(self, style: InjectionStyle = "structured") -> str:
        """Return this attack's payload in ``style``.

        Raises:
            ValueError: On an unknown style, rather than silently injecting
                nothing and scoring a false negative.
        """
        if style == "structured":
            return self.structured_payload()
        if style == "natural":
            return self.natural_payload()
        raise ValueError(f"unknown injection style {style!r}")

    # -- staging -----------------------------------------------------------

    def apply(self, page: Page, style: InjectionStyle = "structured") -> Page:
        """Return a copy of ``page`` carrying this attack's payload.

        Never mutates ``page``.
        """
        poisoned = page.copy()
        text = self.payload(style)
        if self.surface == "description":
            poisoned.description = f"{poisoned.description}\n{text}"
        else:
            poisoned.reviews = [*poisoned.reviews, text]
        return poisoned

    # -- scoring -----------------------------------------------------------

    @abstractmethod
    def goal_achieved(self, result: CheckoutResult, intent: IntentMandate) -> bool:
        """Return whether the attacker's objective was actually met.

        A blocked checkout never counts as success, regardless of how close
        the agent came to executing it.
        """
        raise NotImplementedError
