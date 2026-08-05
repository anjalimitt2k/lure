"""Running the benchmark and reporting it.

Output is plain text and CSV, and identical on every offline run.
Reproducibility is the product here: a Lure number is only worth quoting if
anyone can regenerate it byte-for-byte on their own machine.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from lure.agent.base import Agent
from lure.agent.mock_agent import MockVulnerableAgent
from lure.attacks.base import InjectionStyle
from lure.defenses.base import Defense
from lure.defenses.classifier import ContentClassifierDefense
from lure.defenses.composite import CompositeDefense
from lure.defenses.guardrails import DeterministicGuardrails
from lure.defenses.no_defense import NoDefense
from lure.environment.marketplace import Marketplace, build_default_marketplace
from lure.eval.metrics import DefenseMetrics, compute_metrics
from lure.eval.runner import TaskRun, run_tasks
from lure.tasks.tasks import Task, build_tasks

__all__ = [
    "BenchmarkReport",
    "DEFAULT_CSV_PATH",
    "default_defenses",
    "format_report",
    "run_benchmark",
    "write_metrics_csv",
]

#: Where the CLI writes results, relative to the current working directory.
DEFAULT_CSV_PATH = Path("results") / "metrics.csv"

_CSV_HEADER = [
    "tier",
    "agent",
    "style",
    "defense",
    "scope",
    "benign_utility",
    "attack_success_rate",
    "over_block_rate",
    "benign_tasks",
    "attack_tasks",
]


@dataclass(frozen=True)
class BenchmarkReport:
    """Everything one benchmark run produced.

    Attributes:
        metrics: Scores, one entry per defense, in the order run.
        runs: Every individual task outcome, retained so a headline number can
            always be traced back to the transactions behind it.
        agent_name: Agent under test.
        style: Injection register the attacks were run in.
        tier: ``"offline"`` for the deterministic core, ``"llm"`` otherwise.
        classifier_precision: Held-out precision of the content classifier, if
            one was among the defenses.
        classifier_recall: Held-out recall of the content classifier, if one
            was among the defenses.
        notes: Free-text caveats printed under the tables.
    """

    metrics: list[DefenseMetrics]
    runs: list[TaskRun]
    agent_name: str = MockVulnerableAgent.name
    style: InjectionStyle = "structured"
    tier: str = "offline"
    classifier_precision: float | None = None
    classifier_recall: float | None = None
    notes: list[str] = field(default_factory=list)


def default_defenses() -> list[Defense]:
    """Return the four defenses, weakest control first.

    Ordered so a reader sees the undefended baseline, then each single-layer
    defense, then the composition -- the sequence the comparison is meant to
    be read in.
    """
    return [
        NoDefense(),
        DeterministicGuardrails(),
        ContentClassifierDefense(),
        CompositeDefense(),
    ]


def _classifier_scores(
    defenses: Sequence[Defense],
) -> tuple[float | None, float | None]:
    """Return held-out precision and recall from any classifier in ``defenses``."""
    for defense in defenses:
        if isinstance(defense, ContentClassifierDefense):
            return defense.precision, defense.recall
        if isinstance(defense, CompositeDefense) and isinstance(
            defense.content_filter, ContentClassifierDefense
        ):
            return defense.content_filter.precision, defense.content_filter.recall
    return None, None


def run_benchmark(
    agent_factory: Callable[[], Agent] = MockVulnerableAgent,
    marketplace_factory: Callable[[], Marketplace] = build_default_marketplace,
    defenses: Sequence[Defense] | None = None,
    tasks: Sequence[Task] | None = None,
    style: InjectionStyle = "structured",
    tier: str = "offline",
) -> BenchmarkReport:
    """Run every task under every defense and score the results.

    Each defense gets a freshly built agent and marketplace, so settled
    payments from one condition cannot influence another.

    Args:
        agent_factory: Builds the agent under test. The LLM tier passes a real
            model agent here and changes nothing else.
        marketplace_factory: Builds a clean environment per defense.
        defenses: Defenses to score. Defaults to all four.
        tasks: Task set. Defaults to the standard set in ``style``.
        style: Injection register for the default task set.
        tier: Label recorded in the report and CSV.

    Returns:
        Scores and the underlying task outcomes.
    """
    defenses = list(defenses) if defenses is not None else default_defenses()
    task_list = (
        list(tasks)
        if tasks is not None
        else build_tasks(marketplace_factory(), style=style)
    )

    metrics: list[DefenseMetrics] = []
    all_runs: list[TaskRun] = []
    for defense in defenses:
        runs = run_tasks(agent_factory(), marketplace_factory(), defense, task_list)
        all_runs.extend(runs)
        metrics.append(compute_metrics(runs, defense_name=defense.name))

    precision, recall = _classifier_scores(defenses)
    return BenchmarkReport(
        metrics=metrics,
        runs=all_runs,
        agent_name=agent_factory().name,
        style=style,
        tier=tier,
        classifier_precision=precision,
        classifier_recall=recall,
    )


# -- text rendering --------------------------------------------------------


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Return a left-aligned fixed-width table as a list of lines."""
    if not rows:
        return ["  ".join(headers), "(no rows)"]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]

    def render(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [render(headers), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    lines.extend(render(row) for row in rows)
    return lines


def format_report(
    metrics: Sequence[DefenseMetrics],
    report: BenchmarkReport | None = None,
) -> str:
    """Render the defense-by-metric table, per-attack ASR, and classifier scores."""
    if not metrics:
        return "No results.\n"

    title = "Lure -- agentic-commerce payment safety"
    if report is not None:
        title += f"  [{report.tier} tier | {report.agent_name} | {report.style}]"

    lines = [
        title,
        f"benign tasks: {metrics[0].benign_total}   "
        f"attack tasks: {metrics[0].attack_total}",
        "",
    ]

    lines.extend(
        _render_table(
            ["defense", "benign_utility", "attack_success_rate", "over_block_rate"],
            [
                [
                    m.defense_name,
                    f"{m.benign_utility:.2f}",
                    f"{m.attack_success_rate:.2f}",
                    f"{m.over_block_rate:.2f}",
                ]
                for m in metrics
            ],
        )
    )

    lines.extend(["", "Attack success rate by attack class (lower is better)"])
    attack_names = sorted({name for m in metrics for name in m.per_attack_asr})
    if not attack_names:
        lines.append("  (no attack tasks scored)")
    else:
        lines.extend(
            _render_table(
                ["defense", *attack_names],
                [
                    [
                        m.defense_name,
                        *(f"{m.per_attack_asr.get(n, 0.0):.2f}" for n in attack_names),
                    ]
                    for m in metrics
                ],
            )
        )

    if report is not None and report.classifier_precision is not None:
        lines.extend(
            [
                "",
                "ContentClassifierDefense -- held-out scores on its own training "
                "distribution",
                f"  precision {report.classifier_precision:.2f}   "
                f"recall {report.classifier_recall:.2f}",
                "  These describe injections drawn from the corpus it was trained on.",
                "  Novel phrasing is what the LLM tier is built to measure.",
            ]
        )

    if report is not None and report.notes:
        lines.append("")
        lines.extend(f"  {note}" for note in report.notes)

    return "\n".join(lines) + "\n"


# -- csv -------------------------------------------------------------------


def write_metrics_csv(
    metrics: Sequence[DefenseMetrics],
    path: Path = DEFAULT_CSV_PATH,
    report: BenchmarkReport | None = None,
    append: bool = False,
) -> Path:
    """Write (or append) the scores to ``path``, creating parents as needed.

    One ``overall`` row per defense, followed by one row per attack class with
    only the attack success rate populated -- benign utility and over-block are
    properties of a defense, not of an attack class. Two
    ``classifier_holdout`` rows carry the content classifier's precision and
    recall when one was scored.

    Args:
        metrics: Scores to write.
        path: Destination CSV.
        report: Run context used to label rows by tier, agent and style.
        append: Append to an existing file rather than replacing it, so the
            LLM tier can extend the offline core's results instead of
            overwriting them.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tier = report.tier if report else "offline"
    agent_name = report.agent_name if report else ""
    style = report.style if report else "structured"

    appending = append and path.exists()
    mode = "a" if appending else "w"

    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not appending:
            writer.writerow(_CSV_HEADER)
        for m in metrics:
            writer.writerow(
                [
                    tier,
                    agent_name,
                    style,
                    m.defense_name,
                    "overall",
                    f"{m.benign_utility:.4f}",
                    f"{m.attack_success_rate:.4f}",
                    f"{m.over_block_rate:.4f}",
                    m.benign_total,
                    m.attack_total,
                ]
            )
            for name in sorted(m.per_attack_asr):
                writer.writerow(
                    [
                        tier,
                        agent_name,
                        style,
                        m.defense_name,
                        name,
                        "",
                        f"{m.per_attack_asr[name]:.4f}",
                        "",
                        "",
                        "",
                    ]
                )
        if report is not None and report.classifier_precision is not None:
            for scope, value in (
                ("classifier_holdout_precision", report.classifier_precision),
                ("classifier_holdout_recall", report.classifier_recall),
            ):
                writer.writerow(
                    [
                        tier,
                        agent_name,
                        style,
                        "ContentClassifierDefense",
                        scope,
                        f"{value:.4f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
    return path
