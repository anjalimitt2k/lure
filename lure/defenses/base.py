"""The interface every defense implements.

A defense is deliberately given everything needed to reason independently of
the agent: the proposed :class:`~lure.environment.models.PaymentMandate`, the
cart it settles, the delivery destination, the user-authorized
:class:`~lure.environment.models.IntentMandate`, and the marketplace for
ground-truth lookups such as catalog prices and which payment destinations
belong to verified merchants.

The point of that signature is that a defense never has to trust the agent's
account of what it is doing. It compares artifacts.

``ship_to`` is optional so that a caller with nothing to say about delivery
keeps working unchanged; ``None`` means "no delivery change is being
proposed" and is read as the intent's own address.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lure.environment.marketplace import Marketplace
from lure.environment.models import Cart, IntentMandate, PaymentMandate

__all__ = ["Defense"]


class Defense(ABC):
    """Base class for payment-time adjudicators."""

    #: Human-readable name used in reports and CSV rows.
    name: str = "defense"

    @abstractmethod
    def validate(
        self,
        payment: PaymentMandate,
        cart: Cart,
        intent: IntentMandate,
        marketplace: Marketplace,
        ship_to: str | None = None,
    ) -> tuple[bool, str]:
        """Decide whether ``payment`` may execute.

        Args:
            payment: The value transfer the agent proposes.
            cart: The basket the payment settles.
            intent: What the user authorized.
            marketplace: Ground truth for merchants, products, prices and pages.
            ship_to: Where the agent intends to deliver. ``None`` means the
                caller is not proposing a delivery change.

        Returns:
            ``(allowed, reason)``. ``reason`` is the empty string when allowed
            and a specific, human-readable explanation when blocked -- reasons
            are part of the benchmark output, not debug noise.
        """
        raise NotImplementedError
