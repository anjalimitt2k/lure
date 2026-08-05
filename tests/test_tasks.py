"""Tests for the task set: shape, determinism, and satisfiability."""

from __future__ import annotations

import pytest

from lure.attacks import (
    ItemSwapAttack,
    MerchantSpoofingAttack,
    PayeeRedirectAttack,
    PriceInflationAttack,
    ShippingExfiltrationAttack,
    SpendCapEvasionAttack,
)
from lure.environment import build_default_marketplace
from lure.tasks import build_tasks

ATTACK_CLASS_NAMES = {
    SpendCapEvasionAttack.name,
    PriceInflationAttack.name,
    ItemSwapAttack.name,
    PayeeRedirectAttack.name,
    MerchantSpoofingAttack.name,
    ShippingExfiltrationAttack.name,
}


def test_task_set_has_the_expected_shape() -> None:
    tasks = build_tasks(build_default_marketplace())
    benign = [t for t in tasks if t.kind == "benign"]
    attacks = [t for t in tasks if t.kind == "attack"]

    assert len(benign) == 4
    assert len(attacks) == 24
    assert len(attacks) == len(benign) * len(ATTACK_CLASS_NAMES)


def test_task_ids_are_unique_and_deterministic() -> None:
    first = [t.task_id for t in build_tasks(build_default_marketplace())]
    second = [t.task_id for t in build_tasks(build_default_marketplace())]

    assert first == second
    assert len(set(first)) == len(first)


def test_benign_tasks_cover_only_verified_merchants() -> None:
    marketplace = build_default_marketplace()
    benign = [t for t in build_tasks(marketplace) if t.kind == "benign"]

    payees = {t.intent.legit_payee for t in benign}
    verified = {m.payment_destination for m in marketplace.merchants if m.verified}

    assert payees == verified
    assert "acct:merchant_bazaar" not in payees


def test_each_benign_intent_resolves_to_exactly_one_product() -> None:
    marketplace = build_default_marketplace()
    for task in build_tasks(marketplace):
        if task.kind != "benign":
            continue
        assert len(marketplace.search(task.intent.item_query)) == 1, task.task_id


def test_benign_intents_are_satisfiable_by_the_honest_purchase() -> None:
    marketplace = build_default_marketplace()
    for task in build_tasks(marketplace):
        if task.kind != "benign":
            continue
        product = marketplace.search(task.intent.item_query)[0]
        assert product.unit_price <= task.intent.max_unit_price, task.task_id
        assert product.unit_price * task.intent.quantity <= task.intent.spend_cap


def test_attack_tasks_reuse_the_matching_benign_intent_verbatim() -> None:
    tasks = build_tasks(build_default_marketplace())
    benign_by_product = {t.task_id.split("::")[1]: t.intent for t in tasks if t.kind == "benign"}

    for task in tasks:
        if task.kind != "attack":
            continue
        product_id = task.task_id.split("::")[1]
        assert task.intent == benign_by_product[product_id], task.task_id


def test_every_benign_intent_is_paired_with_every_attack_class() -> None:
    tasks = build_tasks(build_default_marketplace())
    names = {t.attack_name for t in tasks if t.kind == "attack"}

    assert names == ATTACK_CLASS_NAMES
    for product_id in {t.task_id.split("::")[1] for t in tasks if t.kind == "benign"}:
        paired = {
            t.attack_name for t in tasks if t.kind == "attack" and t.task_id.split("::")[1] == product_id
        }
        assert paired == names


def test_benign_tasks_carry_no_attack() -> None:
    for task in build_tasks(build_default_marketplace()):
        if task.kind == "benign":
            assert task.attack is None
            assert task.attack_name is None


def test_spend_cap_attacks_are_sized_per_intent() -> None:
    marketplace = build_default_marketplace()
    for task in build_tasks(marketplace):
        if task.attack_name != SpendCapEvasionAttack.name:
            continue
        product = marketplace.search(task.intent.item_query)[0]
        assert isinstance(task.attack, SpendCapEvasionAttack)
        cost = product.unit_price * task.attack.injected_quantity
        assert cost > task.intent.spend_cap, task.task_id


def test_benign_intents_have_headroom_over_the_exact_order_cost() -> None:
    marketplace = build_default_marketplace()
    for task in build_tasks(marketplace):
        if task.kind != "benign":
            continue
        product = marketplace.search(task.intent.item_query)[0]
        exact = product.unit_price * task.intent.quantity
        assert task.intent.spend_cap == pytest.approx(exact + 10.00)
