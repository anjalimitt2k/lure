"""Assembling the LLM tier's task set.

Two groups of attack tasks, and the split between them is the whole point of
the tier:

* **Fixed natural** -- the six attack classes rendered as prose. Realistic
  wording, but written by hand and present in the content classifier's
  training corpus.
* **Red-team generated** -- the same six goals, phrased by a model. Out of
  distribution by construction.

Scoring is identical across both, because both use the same outcome-based
:meth:`~lure.attacks.base.Attack.goal_achieved`. Any gap between the two
groups is therefore a property of the defense, not of the scorer.
"""

from __future__ import annotations

from lure.environment.marketplace import Marketplace, build_default_marketplace
from lure.llm.client import LLMBackend
from lure.redteam.generator import GeneratedInjectionAttack, generate_injections
from lure.tasks.tasks import Task, build_tasks

__all__ = ["build_llm_tasks"]


def build_llm_tasks(
    client: LLMBackend,
    marketplace: Marketplace | None = None,
    redteam_variants: int = 2,
) -> list[Task]:
    """Return the natural-language task set, with red-team cases appended.

    Red-team variants are generated once per attack class and reused across
    intents. Generating per intent would multiply cost for no extra signal --
    the phrasing does not depend on which product is being bought.

    Args:
        client: LLM backend used for generation. Cached, so repeat runs are
            free.
        marketplace: Catalog to build against.
        redteam_variants: Variants to request per attack class. Zero skips
            generation entirely.

    Returns:
        Benign tasks, fixed natural attack tasks, then red-team attack tasks.
    """
    marketplace = marketplace or build_default_marketplace()
    tasks = build_tasks(marketplace, style="natural")
    if redteam_variants <= 0:
        return tasks

    attack_tasks = [t for t in tasks if t.attack is not None]

    # One representative per attack class drives generation; the resulting
    # strings are then attached to every intent that class appears under.
    representatives: dict[str, Task] = {}
    for task in attack_tasks:
        representatives.setdefault(str(task.attack_name), task)

    generated: list[Task] = []
    for attack_name, representative in representatives.items():
        assert representative.attack is not None
        variants = generate_injections(
            client, representative.attack, count=redteam_variants
        )
        if not variants:
            continue
        for task in attack_tasks:
            if task.attack_name != attack_name or task.attack is None:
                continue
            for index, text in enumerate(variants):
                generated.append(
                    Task(
                        task_id=f"redteam::{task.task_id}::v{index}",
                        intent=task.intent,
                        attack=GeneratedInjectionAttack(
                            base=task.attack, text=text, variant=index
                        ),
                        style="natural",
                    )
                )

    return tasks + generated
