"""Loading an external defense from a dotted path.

This is the seam a third party plugs into. They write a
:class:`~lure.defenses.base.Defense` subclass anywhere importable, then run::

    python -m lure.cli run --defense mypkg.mymodule.MyDefense

Errors are the whole job here. A submission protocol whose failure mode is a
raw ``ImportError`` traceback wastes the submitter's time, so every way this
can go wrong -- bad path shape, missing module, missing attribute, wrong type,
constructor that needs arguments -- produces a specific message naming what to
fix.
"""

from __future__ import annotations

import importlib
import inspect

from lure.defenses.base import Defense

__all__ = ["DefenseLoadError", "load_defense"]


class DefenseLoadError(Exception):
    """Raised when a ``--defense`` dotted path cannot be turned into a Defense."""


def load_defense(dotted_path: str) -> Defense:
    """Import and instantiate the :class:`Defense` named by ``dotted_path``.

    Args:
        dotted_path: Fully qualified path such as
            ``lure.defenses.example_submission.ExampleSubmissionDefense``.
            The final segment is the class name; everything before it is the
            module.

    Returns:
        An instance, ready to score.

    Raises:
        DefenseLoadError: On any failure, with a message naming the cause.
    """
    path = dotted_path.strip()
    if "." not in path:
        raise DefenseLoadError(
            f"{dotted_path!r} is not a dotted path. Expected "
            "'module.path.ClassName', e.g. "
            "'lure.defenses.example_submission.ExampleSubmissionDefense'."
        )

    module_name, _, class_name = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DefenseLoadError(
            f"could not import module {module_name!r}: {exc}. Is it on the "
            "Python path? Try `pip install -e .` from your package root."
        ) from exc

    try:
        candidate = getattr(module, class_name)
    except AttributeError as exc:
        raise DefenseLoadError(
            f"module {module_name!r} has no attribute {class_name!r}."
        ) from exc

    if not inspect.isclass(candidate) or not issubclass(candidate, Defense):
        raise DefenseLoadError(
            f"{dotted_path!r} is not a subclass of lure.defenses.base.Defense. "
            "A submission must subclass Defense and implement "
            "validate(payment, cart, intent, marketplace, ship_to=None)."
        )

    try:
        return candidate()
    except TypeError as exc:
        raise DefenseLoadError(
            f"could not instantiate {dotted_path!r} with no arguments: {exc}. "
            "A submitted defense must be constructible as MyDefense(); put any "
            "configuration in default arguments."
        ) from exc
