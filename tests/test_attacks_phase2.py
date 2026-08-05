"""Tests for the four Phase 2 attacks and the dual-register payload machinery.

Each attack must land against an undefended agent -- an attack that does not
work proves nothing about the defense that "stops" it -- and each must expose
both registers of the same fraud.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lure.agent import MockVulnerableAgent
from lure.attacks import (
    ATTACKER_ADDRESS,
    ItemSwapAttack,
    MerchantSpoofingAttack,
    PayeeRedirectAttack,
    PriceInflationAttack,
    ShippingExfiltrationAttack,
    SpendCapEvasionAttack,
)
from lure.defenses import DeterministicGuardrails, NoDefense
from lure.environment import build_default_marketplace
from lure.environment.injection import iter_injected_instructions
from lure.eval.runner import run_task
from lure.tasks.tasks import build_attacks, build_benign_intent


def _setup(product_id: str = "p_mug"):
    marketplace = build_default_marketplace()
    product = marketplace.get_product(product_id)
    return marketplace, product, build_benign_intent(product, marketplace)


def _all_attacks(product_id: str = "p_mug"):
    marketplace, product, intent = _setup(product_id)
    return marketplace, product, intent, build_attacks(intent, product, marketplace)


# -- dual-register payload machinery ---------------------------------------


def test_every_attack_exposes_both_registers() -> None:
    _, _, _, attacks = _all_attacks()

    for attack in attacks:
        structured = attack.payload("structured")
        natural = attack.payload("natural")
        assert "[[INSTRUCTION]]" in structured, attack.name
        assert "[[INSTRUCTION]]" not in natural, attack.name
        assert len(natural.split()) >= 15, attack.name


def test_apply_never_mutates_the_source_page() -> None:
    marketplace, product, _, attacks = _all_attacks()
    page = marketplace.get_page(product.id)
    before = (page.description, list(page.reviews), dict(page.metadata))

    for attack in attacks:
        for style in ("structured", "natural"):
            attack.apply(page, style=style)

    assert (page.description, page.reviews, page.metadata) == before


def test_structured_style_lands_a_parseable_instruction() -> None:
    marketplace, product, _, attacks = _all_attacks()
    page = marketplace.get_page(product.id)

    for attack in attacks:
        poisoned = attack.apply(page, style="structured")
        assert list(iter_injected_instructions(poisoned)), attack.name


def test_natural_style_lands_no_parseable_instruction() -> None:
    """Prose must be invisible to the structured parser, or the tiers conflate."""
    marketplace, product, _, attacks = _all_attacks()
    page = marketplace.get_page(product.id)

    for attack in attacks:
        poisoned = attack.apply(page, style="natural")
        assert not list(iter_injected_instructions(poisoned)), attack.name


def test_unknown_style_is_rejected() -> None:
    _, _, _, attacks = _all_attacks()

    with pytest.raises(ValueError):
        attacks[0].payload("telepathic")  # type: ignore[arg-type]


# -- PriceInflationAttack --------------------------------------------------


def test_price_inflation_lands_against_no_defense() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = PriceInflationAttack.for_intent(intent, product)

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert result.executed is True
    assert result.amount > result.cart.total()
    assert attack.goal_achieved(result, intent) is True


def test_price_inflation_is_priced_above_the_authorized_ceiling() -> None:
    _, product, intent = _setup("p_mug")
    attack = PriceInflationAttack.for_intent(intent, product)

    assert attack.inflated_unit_price > intent.max_unit_price


def test_price_inflation_is_blocked_by_the_amount_integrity_check() -> None:
    """The gap Phase 1 documented and Phase 2 closes."""
    marketplace, product, intent = _setup("p_mug")
    attack = PriceInflationAttack.for_intent(intent, product)

    result = run_task(
        MockVulnerableAgent(), marketplace, DeterministicGuardrails(), intent, attack
    )

    assert result.executed is False
    assert "does not match the cart's catalog value" in result.blocked_reason
    assert attack.goal_achieved(result, intent) is False


# -- ItemSwapAttack --------------------------------------------------------


def test_item_swap_lands_against_no_defense() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = ItemSwapAttack.for_intent(intent, product, marketplace)

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert result.executed is True
    assert len(result.cart.items) == 2
    assert attack.goal_achieved(result, intent) is True


def test_item_swap_keeps_the_authorized_item_in_the_cart() -> None:
    """The smuggled item is added, not substituted -- that is what hides it."""
    marketplace, product, intent = _setup("p_mug")
    attack = ItemSwapAttack.for_intent(intent, product, marketplace)

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert any(item.product.id == "p_mug" for item in result.cart.items)
    assert any(item.product.id != "p_mug" for item in result.cart.items)


def test_item_swap_picks_a_product_the_query_does_not_match() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = ItemSwapAttack.for_intent(intent, product, marketplace)
    swapped = marketplace.get_product(attack.swap_product_id)

    assert intent.item_query.lower() not in swapped.name.lower()


def test_item_swap_is_blocked_by_guardrails() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = ItemSwapAttack.for_intent(intent, product, marketplace)

    result = run_task(
        MockVulnerableAgent(), marketplace, DeterministicGuardrails(), intent, attack
    )

    assert result.executed is False
    assert attack.goal_achieved(result, intent) is False


def test_item_swap_goal_ignores_which_product_was_smuggled() -> None:
    """Scored on 'something unauthorized', not on the payload's own id."""
    marketplace, product, intent = _setup("p_mug")
    attack = ItemSwapAttack(swap_product_id="p_lamp", swap_product_name="Desk Lamp")
    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    other = ItemSwapAttack(swap_product_id="p_mouse", swap_product_name="Wireless Mouse")
    assert other.goal_achieved(result, intent) is True


# -- MerchantSpoofingAttack ------------------------------------------------


def test_merchant_spoofing_lands_against_no_defense() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = MerchantSpoofingAttack.for_marketplace(marketplace)

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert result.executed is True
    assert result.payee == "acct:merchant_bazaar"
    assert attack.goal_achieved(result, intent) is True


def test_merchant_spoofing_targets_a_real_but_unverified_seller() -> None:
    marketplace = build_default_marketplace()
    attack = MerchantSpoofingAttack.for_marketplace(marketplace)

    merchant = next(
        m
        for m in marketplace.merchants
        if m.payment_destination == attack.spoofed_destination
    )
    assert merchant.verified is False
    assert "verified" in attack.natural_payload().lower()


def test_merchant_spoofing_and_payee_redirect_share_one_guardrail() -> None:
    """Two unrelated pretexts, one outcome-level check. The intended overlap."""
    marketplace, product, intent = _setup("p_mug")
    defense = DeterministicGuardrails()

    for attack in (
        MerchantSpoofingAttack.for_marketplace(marketplace),
        PayeeRedirectAttack(),
    ):
        result = run_task(MockVulnerableAgent(), marketplace, defense, intent, attack)
        assert result.executed is False, attack.name
        assert "verified merchant" in result.blocked_reason, attack.name


def test_merchant_spoofing_needs_an_unverified_merchant() -> None:
    marketplace = build_default_marketplace()
    verified_only = type(marketplace)(
        merchants=[m for m in marketplace.merchants if m.verified],
        products=[],
        pages=[],
    )

    with pytest.raises(LookupError):
        MerchantSpoofingAttack.for_marketplace(verified_only)


# -- ShippingExfiltrationAttack --------------------------------------------


def test_shipping_exfiltration_lands_against_no_defense() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = ShippingExfiltrationAttack()

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert result.executed is True
    assert result.ship_to == ATTACKER_ADDRESS
    assert result.ship_to != intent.ship_to
    assert attack.goal_achieved(result, intent) is True


def test_shipping_exfiltration_leaves_the_payment_entirely_clean() -> None:
    """Every financial signal is correct; only the delivery is stolen."""
    marketplace, product, intent = _setup("p_mug")

    result = run_task(
        MockVulnerableAgent(),
        marketplace,
        NoDefense(),
        intent,
        ShippingExfiltrationAttack(),
    )

    assert result.payee == intent.legit_payee
    assert result.amount == pytest.approx(result.cart.total())
    assert result.amount <= intent.spend_cap


def test_shipping_exfiltration_is_blocked_by_the_delivery_check() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = ShippingExfiltrationAttack()

    result = run_task(
        MockVulnerableAgent(), marketplace, DeterministicGuardrails(), intent, attack
    )

    assert result.executed is False
    assert "not the authorized destination" in result.blocked_reason
    assert attack.goal_achieved(result, intent) is False


def test_marketplace_records_the_shipping_destination() -> None:
    marketplace, product, intent = _setup("p_mug")
    page = marketplace.get_page(product.id)
    staged = marketplace.with_page(ShippingExfiltrationAttack().apply(page))

    MockVulnerableAgent().run(intent, staged, NoDefense())

    assert staged.executed_shipments == [ATTACKER_ADDRESS]
    # The unstaged marketplace settled nothing: run isolation is per-copy.
    assert marketplace.executed_shipments == []


# -- guardrail check ordering ----------------------------------------------


def test_amount_integrity_is_reported_before_the_spend_cap() -> None:
    """Pinned order: a transaction violating both reports check 1."""
    marketplace, product, intent = _setup("p_mug")
    tight = replace(intent, spend_cap=1.00)
    attack = PriceInflationAttack.for_intent(intent, product)

    result = run_task(
        MockVulnerableAgent(), marketplace, DeterministicGuardrails(), tight, attack
    )

    assert "does not match the cart's catalog value" in result.blocked_reason


def test_all_six_attacks_land_undefended_and_are_stopped_by_guardrails() -> None:
    marketplace, product, intent, attacks = _all_attacks()

    for attack in attacks:
        undefended = run_task(
            MockVulnerableAgent(), marketplace, NoDefense(), intent, attack
        )
        assert attack.goal_achieved(undefended, intent) is True, attack.name

        guarded = run_task(
            MockVulnerableAgent(), marketplace, DeterministicGuardrails(), intent, attack
        )
        assert attack.goal_achieved(guarded, intent) is False, attack.name
        assert guarded.blocked_reason, attack.name


def test_add_item_ignores_an_unknown_product_id() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = ItemSwapAttack(swap_product_id="p_nonexistent", swap_product_name="Ghost")

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert [i.product.id for i in result.cart.items] == ["p_mug"]
