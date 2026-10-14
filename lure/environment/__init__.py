"""Test environment for the Lure benchmark: mandates, catalog, checkout.

The environment is the *measurement surface*, not a product. It models just
enough of an agentic-commerce transaction to make payment divergence
observable and scorable.
"""

from lure.environment.injection import (
    INSTRUCTION_PREFIX,
    InjectedInstruction,
    format_instruction,
    iter_injected_instructions,
)
from lure.environment.catalog import CATALOGS, build_large_marketplace, build_marketplace
from lure.environment.marketplace import Marketplace, build_default_marketplace
from lure.environment.models import (
    Cart,
    CartItem,
    CheckoutResult,
    IntentMandate,
    Merchant,
    Page,
    PaymentMandate,
    Product,
)

__all__ = [
    "CATALOGS",
    "Cart",
    "CartItem",
    "CheckoutResult",
    "INSTRUCTION_PREFIX",
    "InjectedInstruction",
    "IntentMandate",
    "Marketplace",
    "Merchant",
    "Page",
    "PaymentMandate",
    "Product",
    "build_default_marketplace",
    "build_large_marketplace",
    "build_marketplace",
    "format_instruction",
    "iter_injected_instructions",
]
