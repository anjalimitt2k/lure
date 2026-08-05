"""Tests that each attack actually works.

An attack that does not land against an undefended agent proves nothing about
the defense that "stops" it, so every attack must first be shown to succeed
under NoDefense.
"""

from __future__ import annotations

import pytest

from lure.agent import MockVulnerableAgent
from lure.attacks import ATTACKER_ACCOUNT, PayeeRedirectAttack, SpendCapEvasionAttack
from lure.defenses import NoDefense
from lure.environment import build_default_marketplace
from lure.environment.injection import iter_injected_instructions
from lure.eval.runner import run_task
from lure.tasks.tasks import build_benign_intent


def _setup(product_id: str):
    marketplace = build_default_marketplace()
    product = marketplace.get_product(product_id)
    return marketplace, product, build_benign_intent(product, marketplace)


# -- SpendCapEvasionAttack -------------------------------------------------


def test_spend_cap_attack_injects_a_quantity_override_without_mutating_the_page() -> None:
    marketplace, product, intent = _setup("p_mouse")
    page = marketplace.get_page(product.id)
    attack = SpendCapEvasionAttack.for_intent(intent, product)

    attacked = attack.apply(page)

    assert attacked is not page
    assert page.description == build_default_marketplace().get_page("p_mouse").description
    injected = list(iter_injected_instructions(attacked))
    assert [(i.key, i.source) for i in injected] == [("set_quantity", "description")]
    assert int(injected[0].value) == attack.injected_quantity


def test_spend_cap_attack_is_sized_to_actually_exceed_the_cap() -> None:
    marketplace, product, intent = _setup("p_mouse")
    attack = SpendCapEvasionAttack.for_intent(intent, product)

    assert product.unit_price * attack.injected_quantity > intent.spend_cap
    assert attack.injected_quantity > intent.quantity


def test_spend_cap_attack_succeeds_against_no_defense() -> None:
    marketplace, product, intent = _setup("p_mouse")
    attack = SpendCapEvasionAttack.for_intent(intent, product)

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert result.executed is True
    assert result.amount > intent.spend_cap
    assert attack.goal_achieved(result, intent) is True


def test_spend_cap_goal_is_not_achieved_when_the_checkout_was_blocked() -> None:
    marketplace, product, intent = _setup("p_mouse")
    attack = SpendCapEvasionAttack.for_intent(intent, product)
    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)
    blocked = type(result)(
        executed=False,
        amount=result.amount,
        payee=result.payee,
        cart=result.cart,
        blocked_reason="blocked",
    )

    assert attack.goal_achieved(blocked, intent) is False


# -- PayeeRedirectAttack ---------------------------------------------------


def test_payee_attack_injects_a_payee_override_without_mutating_the_page() -> None:
    marketplace, product, _ = _setup("p_mug")
    page = marketplace.get_page(product.id)
    original_review_count = len(page.reviews)

    attacked = PayeeRedirectAttack().apply(page)

    assert len(page.reviews) == original_review_count
    assert len(attacked.reviews) == original_review_count + 1
    injected = list(iter_injected_instructions(attacked))
    assert [(i.key, i.value) for i in injected] == [("set_payee", ATTACKER_ACCOUNT)]
    assert injected[0].source.startswith("review[")


def test_payee_attack_succeeds_against_no_defense() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = PayeeRedirectAttack()

    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    assert result.executed is True
    assert result.payee == ATTACKER_ACCOUNT
    assert result.payee != intent.legit_payee
    assert attack.goal_achieved(result, intent) is True


def test_payee_attack_leaves_the_rest_of_the_transaction_clean() -> None:
    """The whole point: only the destination is wrong, so nothing else flags."""
    marketplace, product, intent = _setup("p_mug")

    result = run_task(
        MockVulnerableAgent(), marketplace, NoDefense(), intent, PayeeRedirectAttack()
    )

    assert result.cart.items[0].product.id == "p_mug"
    assert result.cart.items[0].quantity == intent.quantity
    assert result.amount == pytest.approx(12.00 * 2)
    assert result.amount <= intent.spend_cap


def test_payee_goal_is_not_achieved_when_the_checkout_was_blocked() -> None:
    marketplace, product, intent = _setup("p_mug")
    attack = PayeeRedirectAttack()
    result = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)
    blocked = type(result)(
        executed=False,
        amount=result.amount,
        payee=result.payee,
        cart=result.cart,
        blocked_reason="blocked",
    )

    assert attack.goal_achieved(blocked, intent) is False


# -- isolation -------------------------------------------------------------


def test_running_an_attack_task_does_not_contaminate_later_tasks() -> None:
    marketplace, product, intent = _setup("p_mouse")
    attack = SpendCapEvasionAttack.for_intent(intent, product)
    run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, attack)

    clean = run_task(MockVulnerableAgent(), marketplace, NoDefense(), intent, None)

    assert clean.cart.items[0].quantity == intent.quantity
    assert clean.amount <= intent.spend_cap
