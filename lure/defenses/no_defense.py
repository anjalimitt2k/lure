"""The null defense: the undefended control condition.

``NoDefense`` exists to measure the attack surface itself. Its attack success
rate is the baseline every real defense is scored against, and its benign
utility confirms that the tasks are actually completable -- if a legitimate
purchase fails with no defense in the way, the task set is broken, not the
defense.
"""

from __future__ import annotations

from lure.defenses.base import Defense
from lure.environment.marketplace import Marketplace
from lure.environment.models import Cart, IntentMandate, PaymentMandate

__all__ = ["NoDefense"]


class NoDefense(Defense):
    """Allow every payment unconditionally."""

    name = "NoDefense"

    def validate(
        self,
        payment: PaymentMandate,
        cart: Cart,
        intent: IntentMandate,
        marketplace: Marketplace,
        ship_to: str | None = None,
    ) -> tuple[bool, str]:
        """Always allow."""
        return True, ""
