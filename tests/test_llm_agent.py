"""Tests for the LLM agent's plumbing, driven entirely by FakeLLM.

What is verified here is that a model's output is parsed safely and routed
through exactly the same defense-and-checkout path the scripted agent uses.
Whether a real model falls for an injection is what ``run-llm`` measures; this
file makes sure the measurement apparatus around it is sound.
"""

from __future__ import annotations

import json

import pytest

from lure.agent.llm_agent import LLMAgent, parse_plan
from lure.attacks import ShippingExfiltrationAttack
from lure.defenses import DeterministicGuardrails, NoDefense
from lure.defenses.base import Defense
from lure.environment import build_default_marketplace
from lure.eval.runner import run_task
from lure.llm.client import FakeLLM
from lure.tasks.tasks import build_benign_intent


def _setup(product_id: str = "p_mug"):
    marketplace = build_default_marketplace()
    product = marketplace.get_product(product_id)
    return marketplace, product, build_benign_intent(product, marketplace)


def _plan_json(
    product_id: str = "p_mug",
    quantity: int = 2,
    unit_price: float = 12.00,
    payee: str = "acct:merchant_harborline",
    ship_to: str = "123 Alder Street, Springfield",
) -> str:
    return json.dumps(
        {
            "cart_items": [
                {"product_id": product_id, "quantity": quantity, "unit_price": unit_price}
            ],
            "payee": payee,
            "ship_to": ship_to,
        }
    )


# -- parsing ---------------------------------------------------------------


def test_parses_a_well_formed_plan() -> None:
    plan = parse_plan(_plan_json())

    assert plan is not None
    assert plan.lines == [("p_mug", 2, 12.00)]
    assert plan.payee == "acct:merchant_harborline"


def test_parses_a_plan_wrapped_in_a_code_fence() -> None:
    plan = parse_plan(f"```json\n{_plan_json()}\n```")

    assert plan is not None
    assert plan.lines[0][0] == "p_mug"


def test_parses_a_plan_with_surrounding_prose() -> None:
    plan = parse_plan(f"Sure! Here is the plan:\n{_plan_json()}\nLet me know.")

    assert plan is not None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not json at all",
        "{",
        "[]",
        json.dumps({"cart_items": [], "payee": "a", "ship_to": "b"}),
        json.dumps({"cart_items": "nope", "payee": "a", "ship_to": "b"}),
        json.dumps({"payee": "a", "ship_to": "b"}),
        json.dumps({"cart_items": [{"product_id": "p", "quantity": 1}], "payee": "a",
                    "ship_to": "b"}),
        json.dumps({"cart_items": [{"product_id": "p", "quantity": 0, "unit_price": 1}],
                    "payee": "a", "ship_to": "b"}),
        json.dumps({"cart_items": [{"product_id": "p", "quantity": "two",
                                    "unit_price": 1}], "payee": "a", "ship_to": "b"}),
        json.dumps({"cart_items": [{"product_id": "p", "quantity": 1,
                                    "unit_price": -5}], "payee": "a", "ship_to": "b"}),
        json.dumps({"cart_items": [{"product_id": "", "quantity": 1, "unit_price": 1}],
                    "payee": "a", "ship_to": "b"}),
        json.dumps({"cart_items": [{"product_id": "p", "quantity": 1, "unit_price": 1}],
                    "payee": 5, "ship_to": "b"}),
    ],
)
def test_rejects_malformed_plans(raw) -> None:
    assert parse_plan(raw) is None


def test_rejects_booleans_masquerading_as_numbers() -> None:
    raw = json.dumps(
        {
            "cart_items": [{"product_id": "p", "quantity": True, "unit_price": 1}],
            "payee": "a",
            "ship_to": "b",
        }
    )
    assert parse_plan(raw) is None


# -- execution path --------------------------------------------------------


def test_a_clean_plan_executes_correctly() -> None:
    marketplace, _, intent = _setup()

    result = LLMAgent(FakeLLM(_plan_json())).run(intent, marketplace, NoDefense())

    assert result.executed is True
    assert result.amount == pytest.approx(24.00)
    assert result.payee == "acct:merchant_harborline"
    assert result.ship_to == intent.ship_to
    assert [i.product.id for i in result.cart.items] == ["p_mug"]


def test_the_defense_is_consulted_with_the_agents_proposal() -> None:
    marketplace, _, intent = _setup()
    seen: list[tuple] = []

    class Recording(Defense):
        name = "Recording"

        def validate(self, payment, cart, intent, marketplace, ship_to=None):
            seen.append((payment.amount, payment.payee, ship_to, len(cart.items)))
            return True, ""

    LLMAgent(FakeLLM(_plan_json())).run(intent, marketplace, Recording())

    assert seen == [(24.00, "acct:merchant_harborline", intent.ship_to, 1)]


def test_a_refusal_is_honoured_and_moves_no_money() -> None:
    marketplace, _, intent = _setup()
    diverted = _plan_json(payee="acct:attacker")

    result = LLMAgent(FakeLLM(diverted)).run(
        intent, marketplace, DeterministicGuardrails()
    )

    assert result.executed is False
    assert result.blocked_reason
    assert marketplace.executed_payments == []


def test_believed_prices_drive_the_amount_so_inflation_is_representable() -> None:
    """The agent pays what it was told, not what the catalog says."""
    marketplace, _, intent = _setup()
    inflated = _plan_json(unit_price=200.00)

    result = LLMAgent(FakeLLM(inflated)).run(intent, marketplace, NoDefense())

    assert result.amount == pytest.approx(400.00)
    assert result.cart.total() == pytest.approx(24.00)


def test_an_inflated_amount_is_caught_by_the_guardrails() -> None:
    marketplace, _, intent = _setup()

    result = LLMAgent(FakeLLM(_plan_json(unit_price=200.00))).run(
        intent, marketplace, DeterministicGuardrails()
    )

    assert result.executed is False
    assert "does not match the cart's catalog value" in result.blocked_reason


def test_the_cart_is_built_from_real_catalog_products() -> None:
    marketplace, _, intent = _setup()

    result = LLMAgent(FakeLLM(_plan_json(unit_price=999.0))).run(
        intent, marketplace, NoDefense()
    )

    assert result.cart.items[0].product is marketplace.get_product("p_mug")
    assert result.cart.items[0].product.unit_price == pytest.approx(12.00)


def test_unknown_product_ids_are_dropped() -> None:
    marketplace, _, intent = _setup()
    raw = json.dumps(
        {
            "cart_items": [
                {"product_id": "p_mug", "quantity": 2, "unit_price": 12.0},
                {"product_id": "p_ghost", "quantity": 1, "unit_price": 5.0},
            ],
            "payee": "acct:merchant_harborline",
            "ship_to": intent.ship_to,
        }
    )

    result = LLMAgent(FakeLLM(raw)).run(intent, marketplace, NoDefense())

    assert [i.product.id for i in result.cart.items] == ["p_mug"]
    assert result.amount == pytest.approx(24.00)


def test_a_plan_of_only_unknown_products_does_not_execute() -> None:
    marketplace, _, intent = _setup()
    raw = _plan_json(product_id="p_ghost")

    result = LLMAgent(FakeLLM(raw)).run(intent, marketplace, NoDefense())

    assert result.executed is False
    assert "no known catalog product" in result.blocked_reason


# -- failure handling ------------------------------------------------------


def test_malformed_output_triggers_one_repair_attempt() -> None:
    marketplace, _, intent = _setup()
    replies = ["I cannot help with that.", _plan_json()]
    fake = FakeLLM(lambda prompt, system: replies.pop(0) if replies else None)

    result = LLMAgent(fake).run(intent, marketplace, NoDefense())

    assert result.executed is True
    assert len(fake.calls) == 2
    assert "could not be parsed" in fake.calls[1][0]


def test_persistently_unparseable_output_blocks_rather_than_crashing() -> None:
    marketplace, _, intent = _setup()

    result = LLMAgent(FakeLLM("still not json")).run(intent, marketplace, NoDefense())

    assert result.executed is False
    assert "no usable purchase plan" in result.blocked_reason
    assert marketplace.executed_payments == []


def test_repair_can_be_disabled() -> None:
    marketplace, _, intent = _setup()
    fake = FakeLLM("nonsense")

    LLMAgent(fake, allow_repair=False).run(intent, marketplace, NoDefense())

    assert len(fake.calls) == 1


def test_an_unavailable_backend_blocks_cleanly() -> None:
    marketplace, _, intent = _setup()

    result = LLMAgent(FakeLLM("x", available=False)).run(
        intent, marketplace, NoDefense()
    )

    assert result.executed is False
    assert result.blocked_reason


# -- prompt construction ---------------------------------------------------


def test_the_page_is_labelled_untrusted_in_the_prompt() -> None:
    marketplace, _, intent = _setup()
    fake = FakeLLM(_plan_json())

    LLMAgent(fake).run(intent, marketplace, NoDefense())
    prompt = fake.calls[0][0]

    assert "UNTRUSTED" in prompt
    assert "<product_page>" in prompt
    assert "not as\ninstructions" in prompt


def test_the_poisoned_page_reaches_the_prompt() -> None:
    """Staging works end to end: what the attack wrote is what the model reads."""
    marketplace, product, intent = _setup()
    fake = FakeLLM(_plan_json())

    run_task(
        LLMAgent(fake),
        marketplace,
        NoDefense(),
        intent,
        ShippingExfiltrationAttack(),
        style="natural",
    )

    assert "addr:attacker" in fake.calls[0][0]
    assert "[[INSTRUCTION]]" not in fake.calls[0][0]


def test_a_naive_model_that_obeys_prose_is_scored_as_a_success() -> None:
    """End-to-end: prose injection, credulous model, outcome-based scoring."""
    marketplace, product, intent = _setup()
    attack = ShippingExfiltrationAttack()

    def credulous(prompt: str, system: str | None) -> str:
        # Stands in for a model that believed the review it just read.
        address = "addr:attacker" if "addr:attacker" in prompt else intent.ship_to
        return _plan_json(ship_to=address)

    result = run_task(
        LLMAgent(FakeLLM(credulous)),
        marketplace,
        NoDefense(),
        intent,
        attack,
        style="natural",
    )

    assert result.executed is True
    assert attack.goal_achieved(result, intent) is True


def test_the_same_prose_attack_is_stopped_by_the_guardrails() -> None:
    marketplace, product, intent = _setup()
    attack = ShippingExfiltrationAttack()

    def credulous(prompt: str, system: str | None) -> str:
        address = "addr:attacker" if "addr:attacker" in prompt else intent.ship_to
        return _plan_json(ship_to=address)

    result = run_task(
        LLMAgent(FakeLLM(credulous)),
        marketplace,
        DeterministicGuardrails(),
        intent,
        attack,
        style="natural",
    )

    assert result.executed is False
    assert attack.goal_achieved(result, intent) is False


def test_both_agents_satisfy_the_same_interface() -> None:
    from lure.agent import Agent, MockVulnerableAgent

    assert issubclass(LLMAgent, Agent)
    assert issubclass(MockVulnerableAgent, Agent)
