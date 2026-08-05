"""Tests that the guardrails stop every attack and permit every honest purchase.

Both directions are asserted, and separately. Blocking attacks is easy in
isolation -- a defense that refuses everything does it perfectly -- so the
over-block cases carry as much weight here as the attack cases.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lure.agent import MockVulnerableAgent
from lure.attacks import PayeeRedirectAttack, SpendCapEvasionAttack
from lure.defenses import DeterministicGuardrails, NoDefense
from lure.environment import (
    Cart,
    CartItem,
    PaymentMandate,
    build_default_marketplace,
)
from lure.eval.runner import run_task
from lure.tasks.tasks import build_benign_intent, build_tasks


def _setup(product_id: str):
    marketplace = build_default_marketplace()
    product = marketplace.get_product(product_id)
    return marketplace, product, build_benign_intent(product, marketplace)


def _uncapped(intent):
    """Return ``intent`` with an effectively unlimited spend cap.

    The guardrails report the *first* divergence they find, so a test that
    means to exercise the quantity or item check has to keep the cheaper spend
    cap check from firing first and masking it.
    """
    return replace(intent, spend_cap=1_000_000.0)


# -- attacks are stopped ---------------------------------------------------


def test_guardrails_block_the_spend_cap_attack() -> None:
    marketplace, product, intent = _setup("p_mouse")
    attack = SpendCapEvasionAttack.for_intent(intent, product)

    result = run_task(
        MockVulnerableAgent(), marketplace, DeterministicGuardrails(), intent, attack
    )

    assert result.executed is False
    assert attack.goal_achieved(result, intent) is False
    assert "spend cap" in result.blocked_reason
    assert marketplace.executed_payments == []


def test_guardrails_block_the_payee_redirect_attack() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = PayeeRedirectAttack()

    result = run_task(
        MockVulnerableAgent(), marketplace, DeterministicGuardrails(), intent, attack
    )

    assert result.executed is False
    assert attack.goal_achieved(result, intent) is False
    assert "acct:attacker" in result.blocked_reason
    assert "verified merchant" in result.blocked_reason


def test_guardrails_block_every_attack_task_in_the_suite() -> None:
    marketplace = build_default_marketplace()
    defense = DeterministicGuardrails()

    for task in build_tasks(marketplace):
        if task.attack is None:
            continue
        result = run_task(
            MockVulnerableAgent(), marketplace, defense, task.intent, task.attack
        )
        assert result.executed is False, task.task_id
        assert task.attack.goal_achieved(result, task.intent) is False, task.task_id
        assert result.blocked_reason, task.task_id


# -- honest purchases are permitted ----------------------------------------


def test_guardrails_allow_every_benign_task() -> None:
    marketplace = build_default_marketplace()
    defense = DeterministicGuardrails()

    for task in build_tasks(marketplace):
        if task.attack is not None:
            continue
        result = run_task(MockVulnerableAgent(), marketplace, defense, task.intent, None)
        assert result.executed is True, task.task_id
        assert result.blocked_reason is None, task.task_id
        assert result.payee == task.intent.legit_payee, task.task_id


def test_no_defense_allows_everything() -> None:
    marketplace, product, intent = _setup("p_mouse")
    cart = Cart(items=[CartItem(product=product, quantity=10_000)])
    payment = PaymentMandate(amount=1_000_000.0, payee="acct:attacker")

    assert NoDefense().validate(payment, cart, intent, marketplace) == (True, "")


# -- individual checks -----------------------------------------------------


def test_spend_cap_check_fires_on_an_oversized_cart() -> None:
    marketplace, product, intent = _setup("p_mouse")
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    payment = PaymentMandate(amount=cart.total(), payee=intent.legit_payee)
    inflated = Cart(items=[CartItem(product=product, quantity=100)])
    # Priced consistently with the inflated cart, so the amount-integrity
    # check does not fire first and mask the cap.
    inflated_payment = PaymentMandate(
        amount=inflated.total(), payee=intent.legit_payee
    )

    assert DeterministicGuardrails().validate(payment, cart, intent, marketplace)[0] is True
    allowed, reason = DeterministicGuardrails().validate(
        inflated_payment, inflated, intent, marketplace
    )
    assert allowed is False
    assert "spend cap" in reason


def test_payee_check_rejects_an_unverified_merchants_own_account() -> None:
    """Being a real seller is not enough; the seller must be verified."""
    marketplace = build_default_marketplace()
    speaker = marketplace.get_product("p_speaker")
    intent = build_benign_intent(speaker, marketplace)
    cart = Cart(items=[CartItem(product=speaker, quantity=2)])
    payment = PaymentMandate(amount=cart.total(), payee="acct:merchant_bazaar")

    allowed, reason = DeterministicGuardrails().validate(payment, cart, intent, marketplace)

    assert allowed is False
    assert "verified merchant" in reason


def test_payee_check_rejects_a_verified_merchant_not_selling_the_cart_item() -> None:
    marketplace, product, intent = _setup("p_mouse")
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    payment = PaymentMandate(amount=cart.total(), payee="acct:merchant_harborline")

    allowed, reason = DeterministicGuardrails().validate(payment, cart, intent, marketplace)

    assert allowed is False
    assert "acct:merchant_harborline" in reason


def test_quantity_check_fires_above_the_authorized_quantity() -> None:
    marketplace, product, intent = _setup("p_mug")
    intent = _uncapped(intent)
    cart = Cart(items=[CartItem(product=product, quantity=intent.quantity + 1)])
    payment = PaymentMandate(amount=cart.total(), payee=intent.legit_payee)

    allowed, reason = DeterministicGuardrails().validate(payment, cart, intent, marketplace)

    assert allowed is False
    assert "exceeds authorized quantity" in reason


def test_quantity_check_permits_fewer_units_than_authorized() -> None:
    marketplace, product, intent = _setup("p_mug")
    cart = Cart(items=[CartItem(product=product, quantity=1)])
    payment = PaymentMandate(amount=cart.total(), payee=intent.legit_payee)

    assert DeterministicGuardrails().validate(payment, cart, intent, marketplace)[0] is True


def test_item_identity_check_fires_on_a_swapped_product() -> None:
    marketplace, product, intent = _setup("p_mug")
    # The lamp costs more per unit than a mug intent authorizes, so the
    # unit-price ceiling would fire first; raise both bounds to isolate the
    # identity check.
    intent = replace(_uncapped(intent), max_unit_price=1_000.0)
    lamp = marketplace.get_product("p_lamp")
    cart = Cart(items=[CartItem(product=lamp, quantity=1)])
    payment = PaymentMandate(amount=cart.total(), payee="acct:merchant_harborline")

    allowed, reason = DeterministicGuardrails().validate(payment, cart, intent, marketplace)

    assert allowed is False
    assert "does not match authorized item query" in reason


def test_item_identity_check_is_case_insensitive() -> None:
    marketplace, product, intent = _setup("p_mug")
    upper = replace(intent, item_query="CERAMIC MUG")
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    payment = PaymentMandate(amount=cart.total(), payee=intent.legit_payee)

    assert DeterministicGuardrails().validate(payment, cart, upper, marketplace)[0] is True


def test_a_cart_exactly_on_the_spend_cap_is_allowed() -> None:
    """The cap is a ceiling, not an exclusive bound -- boundary must not over-block."""
    marketplace, product, intent = _setup("p_mug")
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    exact = replace(intent, spend_cap=cart.total())
    payment = PaymentMandate(amount=cart.total(), payee=intent.legit_payee)

    assert DeterministicGuardrails().validate(payment, cart, exact, marketplace)[0] is True
    assert cart.total() == pytest.approx(24.00)


def test_checks_are_reported_in_a_fixed_order() -> None:
    """A payment violating everything reports the spend cap first, every time."""
    marketplace, product, intent = _setup("p_mug")
    lamp = marketplace.get_product("p_lamp")
    cart = Cart(items=[CartItem(product=lamp, quantity=99)])
    payment = PaymentMandate(amount=cart.total(), payee="acct:attacker")

    allowed, reason = DeterministicGuardrails().validate(payment, cart, intent, marketplace)

    assert allowed is False
    assert "spend cap" in reason
