"""A worked example of an external defense submission.

This exists to be pointed at, run, and copied::

    python -m lure.cli run --defense lure.defenses.example_submission.ExampleSubmissionDefense

It is deliberately a *partial* defense, not a good one. It checks the payee and
nothing else, so scoring it produces an instructive result rather than a
flattering one: it stops the two attacks that misdirect funds and is blind to
the four that do not. That is the shape of most real single-signal defenses,
and seeing it scored next to the built-ins is the point of the submission
protocol.

See ``docs/SUBMITTING.md`` for the full contract.
"""

from __future__ import annotations

from lure.defenses.base import Defense
from lure.environment.marketplace import Marketplace
from lure.environment.models import Cart, IntentMandate, PaymentMandate

__all__ = ["ExampleSubmissionDefense"]


class ExampleSubmissionDefense(Defense):
    """Allow a payment only if it goes to the payee the user authorized."""

    name = "ExampleSubmissionDefense"

    def validate(
        self,
        payment: PaymentMandate,
        cart: Cart,
        intent: IntentMandate,
        marketplace: Marketplace,
        ship_to: str | None = None,
    ) -> tuple[bool, str]:
        """Block when the payee differs from the Intent Mandate's payee.

        A single-signal check. It catches payee redirection and merchant
        spoofing, and has nothing to say about quantity, price, item identity
        or delivery -- which the benchmark will show plainly.
        """
        if payment.payee != intent.legit_payee:
            return False, (
                f"payee {payment.payee!r} is not the authorized payee "
                f"{intent.legit_payee!r}"
            )
        return True, ""
