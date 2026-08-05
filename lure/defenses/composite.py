"""Layered defense: content filter in front, deterministic guardrails behind.

The ordering matters and is not arbitrary. The classifier runs first as an
*input filter*, rejecting pages that look manipulated before the transaction is
reasoned about at all. The guardrails run second as an *output check*, on the
transaction the agent actually proposes. A block from either is a block.

The two layers fail in opposite directions, which is why they are worth
composing. The classifier catches manipulation attempts the guardrails would
allow because the resulting transaction stays inside the mandate -- but it
only catches phrasing it recognises. The guardrails catch nothing on the way
in and everything on the way out, regardless of phrasing, but only where the
outcome itself is checkable.

Composite exists to make one thing measurable: whether adding a content filter
on top of outcome-level enforcement buys anything, and what it costs in
over-blocking. In Lure it is scored on the same tasks as its two halves, so
the answer is a number rather than an argument.
"""

from __future__ import annotations

from lure.defenses.base import Defense
from lure.defenses.classifier import ContentClassifierDefense
from lure.defenses.guardrails import DeterministicGuardrails
from lure.environment.marketplace import Marketplace
from lure.environment.models import Cart, IntentMandate, PaymentMandate

__all__ = ["CompositeDefense"]


class CompositeDefense(Defense):
    """Run a content filter, then deterministic guardrails; either may block."""

    name = "CompositeDefense"

    def __init__(
        self,
        content_filter: Defense | None = None,
        guardrails: Defense | None = None,
    ) -> None:
        """Initialise with the two layers.

        Args:
            content_filter: Input-side filter. Defaults to
                :class:`~lure.defenses.classifier.ContentClassifierDefense`.
            guardrails: Output-side check. Defaults to
                :class:`~lure.defenses.guardrails.DeterministicGuardrails`.
        """
        self.content_filter = content_filter or ContentClassifierDefense()
        self.guardrails = guardrails or DeterministicGuardrails()

    def validate(
        self,
        payment: PaymentMandate,
        cart: Cart,
        intent: IntentMandate,
        marketplace: Marketplace,
        ship_to: str | None = None,
    ) -> tuple[bool, str]:
        """Block if either layer blocks, reporting which one did."""
        allowed, reason = self.content_filter.validate(
            payment, cart, intent, marketplace, ship_to
        )
        if not allowed:
            return False, f"[content filter] {reason}"

        allowed, reason = self.guardrails.validate(
            payment, cart, intent, marketplace, ship_to
        )
        if not allowed:
            return False, f"[guardrails] {reason}"

        return True, ""
