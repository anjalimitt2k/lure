"""Attacks: content injected into merchant pages to divert an agent's payment.

Each attack states a concrete fraud goal in terms of what the executed
transaction did -- what money moved, and where the goods went -- so "did it
work?" is scored on outcomes rather than on whether the agent sounded
confused. Every attack renders that same goal in two registers: a structured
``[[INSTRUCTION]]`` token for the offline scripted agent, and natural-language
prose for the LLM tier.

The six classes cover the distinct ways an executed transaction can diverge
from the mandate that authorized it:

============================== ==========================================
``SpendCapEvasionAttack``      buy too many
``PriceInflationAttack``       pay more than the goods are worth
``ItemSwapAttack``             buy something not asked for
``PayeeRedirectAttack``        pay an attacker account
``MerchantSpoofingAttack``     pay an unvetted seller posing as official
``ShippingExfiltrationAttack`` deliver to an attacker address
============================== ==========================================
"""

from lure.attacks.base import Attack, InjectionStyle
from lure.attacks.item_swap import ItemSwapAttack
from lure.attacks.merchant_spoofing import MerchantSpoofingAttack
from lure.attacks.payee_redirect import ATTACKER_ACCOUNT, PayeeRedirectAttack
from lure.attacks.price_inflation import PriceInflationAttack
from lure.attacks.shipping_exfiltration import (
    ATTACKER_ADDRESS,
    ShippingExfiltrationAttack,
)
from lure.attacks.spend_cap_evasion import SpendCapEvasionAttack

__all__ = [
    "ATTACKER_ACCOUNT",
    "ATTACKER_ADDRESS",
    "Attack",
    "InjectionStyle",
    "ItemSwapAttack",
    "MerchantSpoofingAttack",
    "PayeeRedirectAttack",
    "PriceInflationAttack",
    "ShippingExfiltrationAttack",
    "SpendCapEvasionAttack",
]
