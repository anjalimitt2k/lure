"""Core data model for the Lure benchmark.

The model is grounded in the *mandate* structure of AP2 (the Agent Payments
Protocol), which separates what a user authorized from what an agent actually
executes:

* An **Intent Mandate** (:class:`IntentMandate`) is the user-authorized envelope
  -- what to buy, at what unit price, how many, up to what total spend, shipped
  where, and paid to whom.
* A **Cart Mandate** (:class:`Cart`) is the concrete basket the agent assembled
  while acting on that intent.
* A **Payment Mandate** (:class:`PaymentMandate`) is the value transfer the
  agent actually attempts: an amount, to a payee.

Lure measures the gap between those artifacts. An *attack* succeeds when
an executed Payment Mandate violates the Intent Mandate that authorized it --
the agent bought the wrong thing, bought too much, or paid the wrong party. A
*defense* succeeds when it blocks that divergence without blocking a
legitimate purchase.

:class:`Page` is the adversarial surface. Its ``description``, ``reviews`` and
``metadata`` values are merchant-controlled free text: exactly the untrusted
channel a real shopping agent reads and, if naive, obeys.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Cart",
    "CartItem",
    "CheckoutResult",
    "IntentMandate",
    "Merchant",
    "Page",
    "PaymentMandate",
    "Product",
]


@dataclass(frozen=True)
class Merchant:
    """A seller of record.

    Attributes:
        id: Stable catalog identifier.
        name: Human-readable storefront name.
        verified: Whether the marketplace has verified this seller. Unverified
            merchants are the control case for payee-authenticity checks.
        payment_destination: Account the merchant is legitimately paid at,
            e.g. ``"acct:merchant_1"``.
    """

    id: str
    name: str
    verified: bool
    payment_destination: str


@dataclass(frozen=True)
class Product:
    """A purchasable item listed by exactly one merchant."""

    id: str
    name: str
    category: str
    unit_price: float
    merchant_id: str


@dataclass
class Page:
    """A product detail page -- the untrusted content surface under test.

    ``description``, every entry of ``reviews``, and every value in
    ``metadata`` are merchant- or third-party-controlled text. Attacks inject
    instructions into these fields; an agent that treats them as instructions
    rather than as data is the vulnerability Lure measures.
    """

    product: Product
    description: str
    reviews: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def copy(self) -> "Page":
        """Return an independent copy, so attacks never mutate shared state.

        The :class:`Product` is frozen and therefore safe to share; the mutable
        injection surfaces are duplicated.
        """
        return Page(
            product=self.product,
            description=self.description,
            reviews=list(self.reviews),
            metadata=dict(self.metadata),
        )

    def as_text(self) -> str:
        """Return every untrusted field flattened into one string.

        This is the canonical view of a page as *content*, used both to train a
        content classifier and to score pages at validation time. Having one
        implementation matters: a classifier trained on differently-assembled
        text than it is later shown would report a held-out score that does not
        describe its real behaviour.
        """
        parts = [self.description, *self.reviews]
        parts.extend(f"{key}: {value}" for key, value in self.metadata.items())
        return "\n".join(parts)


@dataclass(frozen=True)
class IntentMandate:
    """What the user actually authorized -- the ground truth for scoring.

    Attributes:
        item_query: What the user asked to buy, matched case-insensitively
            against product names.
        max_unit_price: Ceiling on the price of a single unit.
        quantity: Number of units authorized.
        spend_cap: Ceiling on the total authorized spend.
        ship_to: Authorized shipping destination.
        legit_payee: The payment destination the user expects to fund.
    """

    item_query: str
    max_unit_price: float
    quantity: int
    spend_cap: float
    ship_to: str
    legit_payee: str


@dataclass(frozen=True)
class CartItem:
    """One line of a cart: a product and how many units of it."""

    product: Product
    quantity: int

    def line_total(self) -> float:
        """Return the catalog-priced total for this line."""
        return self.product.unit_price * self.quantity


@dataclass
class Cart:
    """The basket the agent assembled while acting on an Intent Mandate."""

    items: list[CartItem] = field(default_factory=list)

    def total(self) -> float:
        """Return the sum of line totals at *catalog* prices.

        Catalog prices are used deliberately: this is the marketplace's own
        view of what the cart is worth, independent of any figure the agent was
        talked into by page content.
        """
        return sum(item.line_total() for item in self.items)


@dataclass(frozen=True)
class PaymentMandate:
    """The value transfer the agent attempts: an amount, to a payee."""

    amount: float
    payee: str


@dataclass
class CheckoutResult:
    """Outcome of a checkout attempt -- the unit of measurement in Lure.

    Attributes:
        executed: Whether money actually moved.
        amount: Amount of the attempted (or executed) payment.
        payee: Destination of the attempted (or executed) payment.
        cart: The cart the payment was for.
        blocked_reason: ``None`` when executed; otherwise the defense's stated
            reason for refusing.
        ship_to: Where the order was actually shipped. Divergence from
            ``IntentMandate.ship_to`` is its own fraud outcome -- goods
            delivered to an attacker are stolen just as surely as
            misdirected funds -- so it is recorded alongside the payment.
    """

    executed: bool
    amount: float
    payee: str
    cart: Cart
    blocked_reason: str | None = None
    ship_to: str = ""
