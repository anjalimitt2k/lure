"""Tests for the trained content classifier and the layered composite defense.

These check that the classifier is a real, reproducible model doing real work
-- and, just as importantly, that its limits are visible rather than papered
over.
"""

from __future__ import annotations

import pytest

from lure.attacks import PayeeRedirectAttack, ShippingExfiltrationAttack
from lure.defenses import CompositeDefense, ContentClassifierDefense, NoDefense
from lure.defenses.classifier import train_classifier
from lure.defenses.training_data import (
    LABEL_CLEAN,
    LABEL_INJECTED,
    build_corpus,
    training_attacks,
)
from lure.environment import Cart, CartItem, PaymentMandate, build_default_marketplace
from lure.tasks.tasks import build_benign_intent


def _setup(product_id: str = "p_mug"):
    marketplace = build_default_marketplace()
    product = marketplace.get_product(product_id)
    return marketplace, product, build_benign_intent(product, marketplace)


def _payment_for(cart: Cart, payee: str) -> PaymentMandate:
    return PaymentMandate(amount=cart.total(), payee=payee)


# -- corpus ----------------------------------------------------------------


def test_corpus_is_deterministic() -> None:
    first, second = build_corpus(), build_corpus()

    assert first.texts == second.texts
    assert first.labels == second.labels


def test_corpus_has_both_classes_in_workable_balance() -> None:
    corpus = build_corpus()

    assert corpus.n_clean > 50
    assert corpus.n_injected > 50
    ratio = corpus.n_injected / corpus.n_clean
    assert 0.5 < ratio < 2.0, ratio


def test_corpus_labels_align_with_texts() -> None:
    corpus = build_corpus()

    assert len(corpus.texts) == len(corpus.labels)
    assert set(corpus.labels) == {LABEL_CLEAN, LABEL_INJECTED}


def test_corpus_covers_all_six_attack_classes_in_both_registers() -> None:
    marketplace = build_default_marketplace()
    names = {attack.name for attack in training_attacks(marketplace)}

    assert len(names) == 6

    corpus = build_corpus()
    injected = [
        text for text, label in zip(corpus.texts, corpus.labels) if label == LABEL_INJECTED
    ]
    assert any("[[INSTRUCTION]]" in text for text in injected)
    assert any("[[INSTRUCTION]]" not in text for text in injected)


def test_real_catalog_pages_are_in_the_clean_class() -> None:
    """The classifier is given the favourable case on benign listings."""
    marketplace = build_default_marketplace()
    corpus = build_corpus()
    clean = {
        text for text, label in zip(corpus.texts, corpus.labels) if label == LABEL_CLEAN
    }

    for product in marketplace.products:
        assert marketplace.get_page(product.id).as_text() in clean


# -- training --------------------------------------------------------------


def test_training_is_reproducible() -> None:
    first, second = train_classifier(), train_classifier()

    assert first.precision == second.precision
    assert first.recall == second.recall
    assert first.n_train == second.n_train


def test_held_out_scores_meet_the_acceptance_bar() -> None:
    model = train_classifier()

    assert model.recall >= 0.90
    assert model.precision >= 0.90
    assert model.n_test > 0


def test_the_model_is_not_a_constant_predictor() -> None:
    """Guards against 'always injected', which would score perfect recall."""
    marketplace = build_default_marketplace()
    model = train_classifier()

    clean_page = marketplace.get_page("p_mug")
    poisoned = PayeeRedirectAttack().apply(clean_page)

    assert model.pipeline.predict([clean_page.as_text()])[0] == LABEL_CLEAN
    assert model.pipeline.predict([poisoned.as_text()])[0] == LABEL_INJECTED


# -- defense behaviour -----------------------------------------------------


def test_classifier_allows_a_clean_cart() -> None:
    marketplace, product, intent = _setup()
    cart = Cart(items=[CartItem(product=product, quantity=2)])

    allowed, reason = ContentClassifierDefense().validate(
        _payment_for(cart, intent.legit_payee), cart, intent, marketplace
    )

    assert allowed is True
    assert reason == ""


def test_classifier_blocks_a_cart_whose_page_is_poisoned() -> None:
    marketplace, product, intent = _setup()
    poisoned = ShippingExfiltrationAttack().apply(marketplace.get_page(product.id))
    marketplace = marketplace.with_page(poisoned)
    cart = Cart(items=[CartItem(product=product, quantity=2)])

    allowed, reason = ContentClassifierDefense().validate(
        _payment_for(cart, intent.legit_payee), cart, intent, marketplace
    )

    assert allowed is False
    assert "content classifier" in reason


def test_classifier_allows_an_empty_cart_rather_than_guessing() -> None:
    marketplace, _, intent = _setup()
    cart = Cart(items=[])

    allowed, _ = ContentClassifierDefense().validate(
        PaymentMandate(amount=0.0, payee=""), cart, intent, marketplace
    )

    assert allowed is True


def test_classifier_ignores_the_transaction_itself() -> None:
    """It reasons about text only -- an obviously fraudulent payment on a clean
    page is invisible to it. This is the limitation, asserted."""
    marketplace, product, intent = _setup()
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    fraudulent = PaymentMandate(amount=99_999.0, payee="acct:attacker")

    allowed, _ = ContentClassifierDefense().validate(
        fraudulent, cart, intent, marketplace, ship_to="addr:attacker"
    )

    assert allowed is True


def test_constructing_the_defense_repeatedly_reuses_the_trained_model() -> None:
    assert ContentClassifierDefense().model is ContentClassifierDefense().model


# -- composite -------------------------------------------------------------


def test_composite_blocks_when_only_the_guardrails_object() -> None:
    marketplace, product, intent = _setup()
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    fraudulent = PaymentMandate(amount=cart.total(), payee="acct:attacker")

    allowed, reason = CompositeDefense().validate(fraudulent, cart, intent, marketplace)

    assert allowed is False
    assert reason.startswith("[guardrails]")


def test_composite_blocks_when_only_the_content_filter_objects() -> None:
    marketplace, product, intent = _setup()
    poisoned = ShippingExfiltrationAttack().apply(marketplace.get_page(product.id))
    marketplace = marketplace.with_page(poisoned)
    cart = Cart(items=[CartItem(product=product, quantity=2)])

    # The transaction itself is clean, so only the content layer can object.
    allowed, reason = CompositeDefense().validate(
        _payment_for(cart, intent.legit_payee), cart, intent, marketplace
    )

    assert allowed is False
    assert reason.startswith("[content filter]")


def test_composite_allows_a_clean_transaction_on_a_clean_page() -> None:
    marketplace, product, intent = _setup()
    cart = Cart(items=[CartItem(product=product, quantity=2)])

    allowed, reason = CompositeDefense().validate(
        _payment_for(cart, intent.legit_payee), cart, intent, marketplace
    )

    assert allowed is True
    assert reason == ""


def test_composite_reports_the_content_filter_first() -> None:
    """Input filter before output check, so the earliest signal is named."""
    marketplace, product, intent = _setup()
    poisoned = ShippingExfiltrationAttack().apply(marketplace.get_page(product.id))
    marketplace = marketplace.with_page(poisoned)
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    fraudulent = PaymentMandate(amount=cart.total(), payee="acct:attacker")

    _, reason = CompositeDefense().validate(fraudulent, cart, intent, marketplace)

    assert reason.startswith("[content filter]")


def test_composite_layers_can_be_swapped_out() -> None:
    marketplace, product, intent = _setup()
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    composite = CompositeDefense(content_filter=NoDefense(), guardrails=NoDefense())

    allowed, _ = composite.validate(
        PaymentMandate(amount=1.0, payee="acct:attacker"), cart, intent, marketplace
    )

    assert allowed is True
