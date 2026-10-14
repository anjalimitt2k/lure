"""Command-line entry point.

Three commands, in the order a newcomer meets them:

``reproduce``
    The one command that matters. Runs the entire offline pipeline -- the
    deterministic core, the in- versus out-of-distribution comparison on both
    catalogs, the figures and ``RESULTS.md`` -- and writes every artifact
    under ``results/``. Deterministic, offline, no credentials.

``run``
    Just the deterministic core, on either catalog, optionally against an
    external defense via ``--defense``.

``run-llm``
    The opt-in tier. Puts a real model in the agent's seat and in the content
    filter's, and writes ``results/llm_results.md``. Without credentials it
    explains how to enable itself and exits 0, because "not configured" is a
    state, not a failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from lure.agent.llm_agent import LLMAgent
from lure.defenses.llm_judge import LLMJudgeDefense
from lure.environment.catalog import CATALOGS, build_large_marketplace, build_marketplace
from lure.eval.llm_results import (
    DEFAULT_LLM_RESULTS_PATH,
    score_content_filters,
    write_llm_results_markdown,
)
from lure.eval.llm_suite import build_llm_tasks, redteam_sources
from lure.eval.loader import DefenseLoadError, load_defense
from lure.eval.report import (
    DEFAULT_CSV_PATH,
    default_defenses,
    format_report,
    run_benchmark,
    write_metrics_csv,
)
from lure.eval.results_doc import write_results_markdown
from lure.eval.warm import collect_prompts, warm_cache
from lure.llm.client import DEFAULT_CACHE_PATH, backend_from_env, unavailable_message
from lure.redteam.ood_eval import reference_llm_report, run_ood_comparison
from lure.viz.figures import generate_all_figures

__all__ = ["build_parser", "cli_entry", "main"]

#: Where the LLM tier writes its own metrics, so ``reproduce`` never clobbers
#: them and they never leak into the offline CSV that CI diffs.
DEFAULT_LLM_CSV_PATH = Path("results") / "llm_metrics.csv"


def _add_catalog_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog",
        choices=sorted(CATALOGS),
        default="small",
        help="catalog to build tasks against (default: small)",
    )


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
    _add_catalog_option(run_parser)

    llm_parser = subparsers.add_parser(
        "run-llm",
        help="opt-in tier: natural-language attacks against a real LLM agent",
    )
    llm_parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_LLM_CSV_PATH,
        help=f"metrics CSV to write (default: {DEFAULT_LLM_CSV_PATH})",
    )
    llm_parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_LLM_RESULTS_PATH,
        help=f"markdown results document to write (default: {DEFAULT_LLM_RESULTS_PATH})",
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
    llm_parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the LLM judge defense and the content-filter comparison",
    )
    llm_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "concurrent calls used to pre-fill the response cache before the "
            "run (default: 4, 0 to answer every prompt in run order)"
        ),
    )
    _add_catalog_option(llm_parser)
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


def _run_offline(csv_path: Path, defense_paths: Sequence[str] | None, catalog: str) -> int:
    """Run the deterministic core and report it."""
    try:
        defenses = _resolve_defenses(defense_paths)
    except DefenseLoadError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if defenses is not None:
        names = ", ".join(defense.name for defense in defenses)
        sys.stdout.write(f"Scoring submitted defenses: {names}\n\n")

    report = run_benchmark(
        marketplace_factory=lambda: build_marketplace(catalog), defenses=defenses
    )
    sys.stdout.write(format_report(report.metrics, report))
    written = write_metrics_csv(report.metrics, csv_path, report)
    sys.stdout.write(f"\nWrote {written}\n")
    return 0


def _describe_comparison(comparison) -> str:
    return (
        f"      in-distribution : miss {comparison.in_distribution.miss_rate:.2f}"
        f"   mean confidence {comparison.in_distribution.mean_confidence:.2f}"
        f"   n={comparison.in_distribution.n}\n"
        f"      out-of-dist.    : miss {comparison.out_of_distribution.miss_rate:.2f}"
        f"   mean confidence {comparison.out_of_distribution.mean_confidence:.2f}"
        f"   n={comparison.out_of_distribution.n}\n"
        f"      guardrails      : ASR {comparison.guardrail_asr_in:.2f} in-dist, "
        f"{comparison.guardrail_asr_out:.2f} out-of-dist.\n\n"
    )


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

    sys.stdout.write("[1/5] Deterministic core: 4 defenses x 6 attack classes\n\n")
    report = run_benchmark()
    sys.stdout.write(format_report(report.metrics, report))
    write_metrics_csv(report.metrics, csv_path, report)
    sys.stdout.write(f"\n      wrote {csv_path}\n\n")

    sys.stdout.write("[2/5] In- vs out-of-distribution comparison, small catalog\n")
    comparison = run_ood_comparison()
    sys.stdout.write(_describe_comparison(comparison))

    sys.stdout.write("[3/5] In- vs out-of-distribution comparison, scaled catalog\n")
    scaled = run_ood_comparison(build_large_marketplace)
    sys.stdout.write(_describe_comparison(scaled))

    # The reference-agent rows behind both comparisons go into the same CSV,
    # labelled by tier and catalog, so every number in RESULTS.md is in the
    # portable interface too.
    for factory in (build_marketplace, build_large_marketplace):
        reference = reference_llm_report(marketplace_factory=factory)
        write_metrics_csv(reference.metrics, csv_path, reference, append=True)

    sys.stdout.write("[4/5] Figures\n")
    figures = generate_all_figures(report, comparison, figure_dir, scaled_comparison=scaled)
    for figure in figures:
        sys.stdout.write(f"      wrote {figure}\n")
    sys.stdout.write("\n")

    sys.stdout.write("[5/5] Results snapshot\n")
    write_results_markdown(report, comparison, results_path, scaled_comparison=scaled)
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


def _run_llm(
    csv_path: Path,
    results_path: Path,
    cache_path: Path,
    redteam_variants: int,
    catalog: str,
    include_judge: bool,
    workers: int = 4,
) -> int:
    """Run the LLM tier, or explain how to enable it.

    Returns 0 in both cases. An unconfigured optional tier is not an error, and
    making it one would break any CI that runs the suite without secrets.
    """
    client = backend_from_env(cache_path=cache_path)
    if client is None:
        sys.stdout.write(unavailable_message() + "\n")
        return 0

    sys.stdout.write(
        f"Running the LLM tier with model {client.model} on the {catalog} catalog.\n"
        f"Responses are cached in {cache_path}; cache hits cost nothing.\n\n"
    )

    marketplace = build_marketplace(catalog)
    tasks = build_llm_tasks(client, marketplace=marketplace, redteam_variants=redteam_variants)
    if not client.available:
        sys.stdout.write(
            "The LLM tier became unavailable during setup; nothing was scored.\n"
        )
        return 0
    sources = redteam_sources(tasks)
    if sources["canned"] and not sources["model"]:
        sys.stdout.write(
            "The model declined to write red-team injections; using the bundled "
            f"canned variants for all {sources['canned']} red-team tasks.\n\n"
        )
    elif sources["canned"]:
        sys.stdout.write(
            f"Red-team tasks: {sources['model']} model-written, {sources['canned']} "
            "canned where the model declined.\n\n"
        )

    defenses = default_defenses(marketplace)
    judge = LLMJudgeDefense(client) if include_judge else None
    if judge is not None:
        defenses.append(judge)

    if workers > 0:
        prompts = collect_prompts(tasks, marketplace, include_judge=judge is not None)
        sys.stdout.write(
            f"Pre-filling the cache: {len(prompts)} distinct prompts, {workers} in flight.\n"
        )
        warm_cache(client, prompts, workers=workers)
        sys.stdout.write("\n")
        if not client.available:
            sys.stdout.write(
                "The LLM tier became unavailable while pre-filling; nothing was scored.\n"
            )
            return 0

    report = run_benchmark(
        agent_factory=lambda: LLMAgent(client),
        marketplace_factory=lambda: build_marketplace(catalog),
        defenses=defenses,
        tasks=tasks,
        style="natural",
        tier="llm",
    )
    report.notes.append(
        "Red-team rows (RedTeam:*) are rephrasings the content classifier was not "
        "trained on"
        + (" (bundled canned variants: the model declined to write its own)."
           if sources["canned"] and not sources["model"] else ".")
    )
    sys.stdout.write(format_report(report.metrics, report))

    filters = score_content_filters(marketplace, judge) if judge is not None else []
    if filters:
        sys.stdout.write("\nContent filters on the same poisoned pages:\n")
        for scores in filters:
            sys.stdout.write(
                f"  {scores.name:<26} miss in-dist {scores.in_distribution.miss_rate:.2f}"
                f"   out-of-dist {scores.out_of_distribution.miss_rate:.2f}"
                f"   gap {scores.miss_rate_gap:+.2f}\n"
            )

    if not client.available:
        sys.stdout.write(
            "\nThe LLM tier became unavailable part-way through; the numbers above "
            "include unscored tasks and were not written.\n"
        )
        return 0

    written_csv = write_metrics_csv(report.metrics, csv_path, report)
    written_doc = write_llm_results_markdown(
        report, filters, client.model, results_path, redteam_sources=sources
    )
    sys.stdout.write(f"\nWrote {written_csv}\nWrote {written_doc}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "reproduce":
        return _run_reproduce(args.results_dir)
    if args.command == "run":
        return _run_offline(args.csv, args.defense, args.catalog)
    if args.command == "run-llm":
        return _run_llm(
            args.csv,
            args.results,
            args.cache,
            args.redteam_variants,
            args.catalog,
            include_judge=not args.no_judge,
            workers=args.workers,
        )
    return 1


def cli_entry() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
