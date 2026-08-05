"""Command-line entry point.

Three commands, in the order a newcomer meets them:

``reproduce``
    The one command that matters. Runs the entire offline pipeline -- the
    deterministic core, the in- versus out-of-distribution comparison, the
    figures and ``RESULTS.md`` -- and writes every artifact under ``results/``.
    Deterministic, offline, no credentials.

``run``
    Just the deterministic core, optionally against an external defense via
    ``--defense``.

``run-llm``
    The opt-in tier. Without credentials it explains how to enable itself and
    exits 0, because "not configured" is a state, not a failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from lure.agent.llm_agent import LLMAgent
from lure.eval.llm_suite import build_llm_tasks
from lure.eval.loader import DefenseLoadError, load_defense
from lure.eval.report import (
    DEFAULT_CSV_PATH,
    format_report,
    run_benchmark,
    write_metrics_csv,
)
from lure.eval.results_doc import write_results_markdown
from lure.llm.client import DEFAULT_CACHE_PATH, LLMClient, unavailable_message
from lure.redteam.ood_eval import run_ood_comparison
from lure.viz.figures import generate_all_figures

__all__ = ["build_parser", "cli_entry", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m lure.cli",
        description=(
            "Lure: the independent benchmark for agentic-commerce payment safety."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="run the whole offline pipeline and regenerate every artifact",
    )
    reproduce_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="directory for all artifacts (default: results)",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="offline core: structured attacks against the scripted agent",
    )
    run_parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"path for the metrics CSV (default: {DEFAULT_CSV_PATH})",
    )
    run_parser.add_argument(
        "--defense",
        action="append",
        metavar="DOTTED.PATH.ToClass",
        help=(
            "score an external Defense subclass instead of the built-ins; "
            "repeatable. See docs/SUBMITTING.md"
        ),
    )

    llm_parser = subparsers.add_parser(
        "run-llm",
        help="opt-in tier: natural-language attacks against a real LLM agent",
    )
    llm_parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"metrics CSV to append to (default: {DEFAULT_CSV_PATH})",
    )
    llm_parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"LLM response cache (default: {DEFAULT_CACHE_PATH})",
    )
    llm_parser.add_argument(
        "--redteam-variants",
        type=int,
        default=2,
        help="model-generated injections per attack class (default: 2, 0 to skip)",
    )
    return parser


# -- commands --------------------------------------------------------------


def _resolve_defenses(paths: Sequence[str] | None):
    """Return the defenses to score, or ``None`` to use the built-ins.

    Raises:
        DefenseLoadError: If any dotted path cannot be loaded.
    """
    if not paths:
        return None
    return [load_defense(path) for path in paths]


def _run_offline(csv_path: Path, defense_paths: Sequence[str] | None) -> int:
    """Run the deterministic core and report it."""
    try:
        defenses = _resolve_defenses(defense_paths)
    except DefenseLoadError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if defenses is not None:
        names = ", ".join(defense.name for defense in defenses)
        sys.stdout.write(f"Scoring submitted defenses: {names}\n\n")

    report = run_benchmark(defenses=defenses)
    sys.stdout.write(format_report(report.metrics, report))
    written = write_metrics_csv(report.metrics, csv_path, report)
    sys.stdout.write(f"\nWrote {written}\n")
    return 0


def _run_reproduce(results_dir: Path) -> int:
    """Run the full offline pipeline and regenerate every artifact.

    Deliberately sequential and chatty: this is the command a stranger runs
    first, and it should say what it is doing at each stage.
    """
    results_dir = Path(results_dir)
    csv_path = results_dir / "metrics.csv"
    results_path = results_dir / "RESULTS.md"
    figure_dir = results_dir / "figures"

    sys.stdout.write("Lure -- full offline reproduction\n")
    sys.stdout.write("=" * 62 + "\n\n")

    sys.stdout.write("[1/4] Deterministic core: 4 defenses x 6 attack classes\n\n")
    report = run_benchmark()
    sys.stdout.write(format_report(report.metrics, report))
    write_metrics_csv(report.metrics, csv_path, report)
    sys.stdout.write(f"\n      wrote {csv_path}\n\n")

    sys.stdout.write("[2/4] In- vs out-of-distribution comparison\n")
    comparison = run_ood_comparison()
    sys.stdout.write(
        f"      in-distribution : miss {comparison.in_distribution.miss_rate:.2f}"
        f"   mean confidence {comparison.in_distribution.mean_confidence:.2f}"
        f"   n={comparison.in_distribution.n}\n"
        f"      out-of-dist.    : miss {comparison.out_of_distribution.miss_rate:.2f}"
        f"   mean confidence {comparison.out_of_distribution.mean_confidence:.2f}"
        f"   n={comparison.out_of_distribution.n}\n"
        f"      guardrails      : ASR {comparison.guardrail_asr_in:.2f} in-dist, "
        f"{comparison.guardrail_asr_out:.2f} out-of-dist.\n\n"
    )

    sys.stdout.write("[3/4] Figures\n")
    figures = generate_all_figures(report, comparison, figure_dir)
    for figure in figures:
        sys.stdout.write(f"      wrote {figure}\n")
    sys.stdout.write("\n")

    sys.stdout.write("[4/4] Results snapshot\n")
    write_results_markdown(report, comparison, results_path)
    sys.stdout.write(f"      wrote {results_path}\n\n")

    sys.stdout.write("=" * 62 + "\n")
    sys.stdout.write(
        f"Done. {2 + len(figures)} artifacts under {results_dir}/\n"
        f"  {results_path}  -- the readable snapshot\n"
        f"  {csv_path}  -- the portable interface\n"
        f"  {figure_dir}/ -- {len(figures)} figures\n"
        "\nEverything above regenerates offline and byte-identically.\n"
    )
    return 0


def _run_llm(csv_path: Path, cache_path: Path, redteam_variants: int) -> int:
    """Run the LLM tier, or explain how to enable it.

    Returns 0 in both cases. An unconfigured optional tier is not an error, and
    making it one would break any CI that runs the suite without secrets.
    """
    client = LLMClient.from_env(cache_path=cache_path)
    if client is None:
        sys.stdout.write(unavailable_message() + "\n")
        return 0

    sys.stdout.write(
        f"Running the LLM tier with model {client.model}.\n"
        f"Responses are cached in {cache_path}; cache hits cost nothing.\n\n"
    )

    tasks = build_llm_tasks(client, redteam_variants=redteam_variants)
    if not client.available:
        sys.stdout.write(
            "The LLM tier became unavailable during setup; nothing was scored.\n"
        )
        return 0

    report = run_benchmark(
        agent_factory=lambda: LLMAgent(client),
        tasks=tasks,
        style="natural",
        tier="llm",
    )
    report.notes.append(
        "Red-team rows (RedTeam:*) are model-written phrasings the content "
        "classifier was not trained on."
    )
    sys.stdout.write(format_report(report.metrics, report))
    written = write_metrics_csv(report.metrics, csv_path, report, append=True)
    sys.stdout.write(f"\nAppended to {written}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "reproduce":
        return _run_reproduce(args.results_dir)
    if args.command == "run":
        return _run_offline(args.csv, args.defense)
    if args.command == "run-llm":
        return _run_llm(args.csv, args.cache, args.redteam_variants)
    return 1


def cli_entry() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
