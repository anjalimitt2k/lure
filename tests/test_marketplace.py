"""Tests for the test environment itself.

If search or checkout are wrong, every downstream number is wrong, so these
are checked before anything about attacks or defenses.
"""

from __future__ import annotations

import pytest

from lure.environment import (
    Cart,
    CartItem,
    PaymentMandate,
    build_default_marketplace,
)


def test_catalog_has_five_products_across_three_merchants() -> None:
    marketplace = build_default_marketplace()
    assert len(marketplace.products) == 5
    assert len(marketplace.merchants) == 3


def test_catalog_includes_an_unverified_merchant() -> None:
    marketplace = build_default_marketplace()
    unverified = [m for m in marketplace.merchants if not m.verified]
    assert len(unverified) == 1


def test_every_product_has_a_page() -> None:
    marketplace = build_default_marketplace()
    for product in marketplace.products:
        assert marketplace.get_page(product.id).product == product


def test_search_is_case_insensitive_substring_match() -> None:
    marketplace = build_default_marketplace()
    assert [p.id for p in marketplace.search("WIRELESS mouse")] == ["p_mouse"]
    assert [p.id for p in marketplace.search("mouse")] == ["p_mouse"]


def test_search_returns_empty_for_no_match() -> None:
    assert build_default_marketplace().search("hovercraft") == []


def test_search_results_keep_catalog_order() -> None:
    marketplace = build_default_marketplace()
    hits = marketplace.search("e")
    assert [p.id for p in hits] == [p.id for p in marketplace.products if "e" in p.name.lower()]


def test_checkout_executes_and_records_the_payment() -> None:
    marketplace = build_default_marketplace()
    product = marketplace.get_product("p_mug")
    cart = Cart(items=[CartItem(product=product, quantity=2)])
    payment = PaymentMandate(amount=cart.total(), payee="acct:merchant_harborline")

    result = marketplace.checkout(payment, cart)

    assert result.executed is True
    assert result.amount == pytest.approx(24.00)
    assert result.payee == "acct:merchant_harborline"
    assert result.blocked_reason is None
    assert marketplace.executed_payments == [payment]


def test_cart_total_sums_line_totals_at_catalog_prices() -> None:
    marketplace = build_default_marketplace()
    cart = Cart(
        items=[
            CartItem(product=marketplace.get_product("p_mug"), quantity=2),
            CartItem(product=marketplace.get_product("p_lamp"), quantity=1),
        ]
    )
    assert cart.total() == pytest.approx(24.00 + 34.25)


def test_with_page_does_not_mutate_the_original_marketplace() -> None:
    marketplace = build_default_marketplace()
    page = marketplace.get_page("p_mouse").copy()
    page.description = "replaced"

    staged = marketplace.with_page(page)

    assert staged.get_page("p_mouse").description == "replaced"
    assert marketplace.get_page("p_mouse").description != "replaced"


def test_page_copy_is_independent() -> None:
    page = build_default_marketplace().get_page("p_mouse")
    copy = page.copy()
    copy.reviews.append("injected")
    copy.metadata["k"] = "v"

    assert "injected" not in page.reviews
    assert "k" not in page.metadata
