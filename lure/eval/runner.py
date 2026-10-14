"""Execution of tasks: stage the attack, run the agent, capture the outcome.

Isolation is the runner's real job. Each task runs against its own marketplace
copy with only that task's injected page installed, so a payload can never
leak into a later task and inflate its own success rate. Nothing else about
the environment changes between tasks, which is what makes runs comparable and
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from lure.agent.base import Agent
from lure.attacks.base import Attack, InjectionStyle
from lure.defenses.base import Defense
from lure.environment.marketplace import Marketplace
from lure.environment.models import CheckoutResult, IntentMandate
from lure.tasks.tasks import Task

__all__ = ["TaskRun", "run_task", "run_tasks", "stage_marketplace"]


def stage_marketplace(
    marketplace: Marketplace,
    intent: IntentMandate,
    attack: Attack | None,
    style: InjectionStyle = "structured",
) -> Marketplace:
    """Return the marketplace the agent will actually read for this task.

    With an attack, that is a private copy serving the attacked page for the
    first search hit on ``intent.item_query``; without one, the base
    marketplace unchanged. Factored out so the cache pre-warmer can build the
    exact prompts the run will issue.

    Raises:
        LookupError: If ``intent.item_query`` matches no product.
    """
    if attack is None:
        return marketplace
    results = marketplace.search(intent.item_query)
    if not results:
        raise LookupError(f"no product matches item_query {intent.item_query!r}")
    target_page = marketplace.get_page(results[0].id)
    return marketplace.with_page(attack.apply(target_page, style=style))


@dataclass(frozen=True)
class TaskRun:
    """A task, the defense it ran under, and what happened."""

    task: Task
    defense_name: str
    result: CheckoutResult


def run_task(
    agent: Agent,
    marketplace: Marketplace,
    defense: Defense,
    intent: IntentMandate,
    attack: Attack | None,
    style: InjectionStyle = "structured",
) -> CheckoutResult:
    """Run one task and return its checkout outcome.

    When ``attack`` is given, it is applied to the page the agent will actually
    reach -- the first search hit for ``intent.item_query`` -- and staged in a
    private marketplace copy. Attacking any other page would silently score a
    payload the agent never read.

    Args:
        agent: The agent under test.
        marketplace: Base environment. Never mutated.
        defense: Adjudicator for the agent's payment.
        intent: The user-authorized mandate.
        attack: Attack to stage, or ``None`` for a benign task.
        style: Register to inject the attack in -- the structured token the
            scripted agent obeys, or prose aimed at a language model.

    Returns:
        The agent's checkout result, executed or blocked.

    Raises:
        LookupError: If ``intent.item_query`` matches no product, which would
            make the task unscoreable.
    """
    return agent.run(intent, stage_marketplace(marketplace, intent, attack, style), defense)


def run_tasks(
    agent: Agent,
    marketplace: Marketplace,
    defense: Defense,
    tasks: list[Task],
) -> list[TaskRun]:
    """Run every task under one defense, in order, and return the outcomes.

    Each task is run against a fresh copy of the environment via
    :func:`run_task`, so a payload staged for one task cannot influence the
    next.
    """
    return [
        TaskRun(
            task=task,
            defense_name=defense.name,
            result=run_task(
                agent, marketplace, defense, task.intent, task.attack, task.style
            ),
        )
        for task in tasks
    ]
