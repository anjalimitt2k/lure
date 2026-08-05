"""Defenses under measurement.

A defense in Lure is the last adjudicator before money moves. It sees the
proposed payment, the cart, the delivery address, the user's Intent Mandate and
the marketplace, and returns an allow/block decision with a reason.

Four defenses, chosen to bracket the design space rather than to enumerate it:

* :class:`~lure.defenses.no_defense.NoDefense` -- the undefended control.
* :class:`~lure.defenses.guardrails.DeterministicGuardrails` -- outcome-level
  enforcement that never asks whether the agent was fooled.
* :class:`~lure.defenses.classifier.ContentClassifierDefense` -- a trained
  model that inspects the untrusted content itself.
* :class:`~lure.defenses.composite.CompositeDefense` -- both, layered.

The interesting comparison is not which scores best on the fixed suite. It is
which keeps scoring well when the attacks stop resembling the training data.
"""

from lure.defenses.base import Defense
from lure.defenses.classifier import ContentClassifierDefense, TrainedClassifier
from lure.defenses.composite import CompositeDefense
from lure.defenses.guardrails import DeterministicGuardrails
from lure.defenses.no_defense import NoDefense

__all__ = [
    "CompositeDefense",
    "ContentClassifierDefense",
    "Defense",
    "DeterministicGuardrails",
    "NoDefense",
    "TrainedClassifier",
]
