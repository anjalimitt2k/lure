"""Pre-warming the response cache concurrently.

The benchmark runs tasks sequentially, defense by defense, because the
scoring path is simple and order-independent and a sequential loop is the
easiest thing to reason about. That is the wrong shape for a backend where
every call takes seconds: a run of several hundred tasks would spend most of
its wall-clock waiting on one request at a time.

The agent's prompt for a task does not depend on which defense adjudicates
it, and the judge's prompt for a page does not depend on which cart it backs.
So every prompt the run will issue can be computed up front and answered
concurrently, after which the sequential run is served entirely from cache.
The run itself is unchanged; only the order in which the model is asked is.

Prompts are built by the same code that builds them during the run --
:meth:`~lure.agent.llm_agent.LLMAgent._build_prompt` on the exact staged
marketplace -- so a pre-warmed key is byte-identical to the key the run looks
up. The one thing not pre-warmed is the agent's repair prompt, issued only
when a first reply fails to parse; those are rare and are answered live.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

from lure.agent import llm_agent
from lure.agent.llm_agent import LLMAgent
from lure.defenses import llm_judge
from lure.defenses.llm_judge import build_judge_prompt
from lure.environment.marketplace import Marketplace
from lure.eval.runner import stage_marketplace
from lure.llm.client import FakeLLM, LLMBackend
from lure.redteam.ood_eval import in_distribution_samples, out_of_distribution_samples
from lure.tasks.tasks import Task

__all__ = ["Prompt", "collect_prompts", "warm_cache"]

Prompt = tuple[str, str | None]


def collect_prompts(
    tasks: Sequence[Task],
    marketplace: Marketplace,
    include_judge: bool = True,
) -> list[Prompt]:
    """Return every ``(prompt, system)`` pair the run will issue, deduplicated.

    Args:
        tasks: The task set about to be run.
        marketplace: The clean catalog the tasks were built against.
        include_judge: Whether the LLM judge will be scored too, in which case
            its prompts for every staged page and every content-filter probe
            page are included.
    """
    builder = LLMAgent(FakeLLM(None))
    seen: set[Prompt] = set()
    prompts: list[Prompt] = []

    def add(prompt: str, system: str | None) -> None:
        pair = (prompt, system)
        if pair not in seen:
            seen.add(pair)
            prompts.append(pair)

    for task in tasks:
        staged = stage_marketplace(marketplace, task.intent, task.attack, task.style)
        product = staged.search(task.intent.item_query)[0]
        page_text = staged.get_page(product.id).as_text()
        add(
            builder._build_prompt(task.intent, product, page_text, staged),
            llm_agent.SYSTEM_PROMPT,
        )
        if include_judge:
            add(build_judge_prompt(page_text), llm_judge.SYSTEM_PROMPT)

    if include_judge:
        for sample in [*in_distribution_samples(marketplace), *out_of_distribution_samples(marketplace)]:
            add(build_judge_prompt(sample.text), llm_judge.SYSTEM_PROMPT)

    return prompts


def warm_cache(
    backend: LLMBackend,
    prompts: Sequence[Prompt],
    workers: int = 4,
    report: Callable[[str], None] | None = None,
) -> int:
    """Answer ``prompts`` concurrently so the run that follows is cache-served.

    Args:
        backend: Backend whose cache to fill. Its ``complete`` must be safe to
            call from several threads, which both shipped backends are.
        prompts: Pairs from :func:`collect_prompts`.
        workers: Concurrent calls in flight.
        report: Sink for progress lines. Defaults to stdout.

    Returns:
        The number of prompts that received a response.
    """
    emit = report or (lambda line: (sys.stdout.write(line + "\n"), sys.stdout.flush()))
    total = len(prompts)
    if total == 0:
        return 0
    answered = 0
    done = 0
    step = max(1, total // 20)

    def one(pair: Prompt) -> bool:
        prompt, system = pair
        return backend.complete(prompt, system=system) is not None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for ok in pool.map(one, prompts):
            done += 1
            answered += int(ok)
            if done % step == 0 or done == total:
                emit(f"      warmed {done}/{total} prompts ({answered} answered)")
            if not backend.available:
                break
    return answered
