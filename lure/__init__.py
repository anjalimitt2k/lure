"""Lure: the independent benchmark for agentic-commerce payment safety.

Lure measures a single open question: can an AI shopping agent be
socially-engineered, via content injected into merchant pages, into buying the
wrong thing or paying the wrong party -- and does a given defense prevent that?

This is a security / safety-measurement project, not a commerce product. The
shopping marketplace, catalog and checkout flow exist only as a test
environment; the scripted agent inside exists only to be attacked and scored.

Phase 1 is a correctness proof of the harness itself: a deliberately naive
agent must fall for every attack (ASR = 1.0) while deterministic guardrails
must stop every attack (ASR = 0.0) without blocking a single legitimate
purchase (over-block = 0.0).

Everything here is deterministic and offline: standard library only, no
network, no model calls, no randomness.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
