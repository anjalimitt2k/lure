"""Generation of ``results/RESULTS.md``, the regenerated results snapshot.

This file is written by the pipeline and must never be hand-edited. Every
number in it is read out of the same objects the test suite asserts on -- a
:class:`~lure.eval.report.BenchmarkReport` and an
:class:`~lure.redteam.ood_eval.OODComparison` -- so a published table cannot
drift away from a pinned acceptance value. If someone changes a defense and
forgets to regenerate, the tests fail rather than the document quietly lying.

Deliberately **timestamp-free**. A generated-on line would change every run and
make ``git diff`` useless for the one question that matters: did any number
move?
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from lure.eval.metrics import DefenseMetrics
from lure.eval.report import BenchmarkReport
from lure.redteam.ood_eval import OODComparison
from lure.viz.figures import FIGURE_NAMES

__all__ = ["DEFAULT_RESULTS_PATH", "build_results_markdown", "write_results_markdown"]

#: Where the snapshot is written, relative to the working directory.
DEFAULT_RESULTS_PATH = Path("results") / "RESULTS.md"

#: Defenses shipped with the benchmark, in report order. Anything else scored
#: through ``--defense`` is an external submission and is labelled as such.
BUILT_IN_DEFENSES = frozenset(
    {
        "NoDefense",
        "DeterministicGuardrails",
        "ContentClassifierDefense",
        "CompositeDefense",
    }
)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Return a GitHub-flavoured markdown table as a list of lines."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _short(attack_name: str) -> str:
    """Shorten an attack class name for a table header."""
    return attack_name.removeprefix("RedTeam:").removesuffix("Attack")


def _headline_section(comparison: OODComparison) -> list[str]:
    """Render the in- versus out-of-distribution comparison."""
    in_scores = comparison.in_distribution
    out_scores = comparison.out_of_distribution
    return [
        "## Headline: the content filter does not survive new wording",
        "",
        "The same six fraud goals, phrased two ways. In-distribution injections are",
        "the fixed templates the classifier's training corpus was generated from;",
        "out-of-distribution injections are red-team phrasings it has not seen.",
        "",
        *_table(
            ["injections", "n", "classifier miss rate", "mean confidence"],
            [
                [
                    "in-distribution (templates it trained on)",
                    str(in_scores.n),
                    f"{in_scores.miss_rate:.2f}",
                    f"{in_scores.mean_confidence:.2f}",
                ],
                [
                    "out-of-distribution (red-team phrasings)",
                    str(out_scores.n),
                    f"{out_scores.miss_rate:.2f}",
                    f"{out_scores.mean_confidence:.2f}",
                ],
            ],
        ),
        "",
        f"Miss-rate gap: **{comparison.miss_rate_gap:+.2f}**. "
        f"Mean-confidence drop: **{comparison.confidence_drop:.2f}**.",
        "",
        "Attack success rate against the offline reference agent, split the same way:",
        "",
        *_table(
            ["defense", "in-distribution ASR", "out-of-distribution ASR"],
            [
                [
                    "ContentClassifierDefense",
                    f"{comparison.classifier_asr_in:.2f}",
                    f"{comparison.classifier_asr_out:.2f}",
                ],
                [
                    "DeterministicGuardrails",
                    f"{comparison.guardrail_asr_in:.2f}",
                    f"{comparison.guardrail_asr_out:.2f}",
                ],
            ],
        ),
        "",
        "The guardrails read no text at all, so rephrasing an attack changes nothing",
        "for them. That contrast is the finding.",
        "",
        f"![in- versus out-of-distribution miss rate](figures/{FIGURE_NAMES[2]})",
    ]


def _core_section(report: BenchmarkReport) -> list[str]:
    """Render the four-defense summary and the per-attack matrix."""
    attack_names = sorted({n for m in report.metrics for n in m.per_attack_asr})

    lines = [
        "## Deterministic core",
        "",
        f"Scripted `{report.agent_name}`, `{report.style}` injections, "
        f"{report.metrics[0].benign_total} benign tasks and "
        f"{report.metrics[0].attack_total} attack tasks.",
        "",
        *_table(
            ["defense", "benign utility", "attack success rate", "over-block rate"],
            [
                [
                    f"`{m.defense_name}`",
                    f"{m.benign_utility:.2f}",
                    f"{m.attack_success_rate:.2f}",
                    f"{m.over_block_rate:.2f}",
                ]
                for m in report.metrics
            ],
        ),
        "",
        f"![attack success rate by defense](figures/{FIGURE_NAMES[0]})",
        "",
        f"![utility versus attack success](figures/{FIGURE_NAMES[1]})",
        "",
        "### Attack success rate by attack class",
        "",
        *_table(
            ["defense", *(_short(n) for n in attack_names)],
            [
                [
                    f"`{m.defense_name}`",
                    *(f"{m.per_attack_asr.get(n, 0.0):.2f}" for n in attack_names),
                ]
                for m in report.metrics
            ],
        ),
    ]

    if report.classifier_precision is not None:
        lines.extend(
            [
                "",
                "### Content classifier, held-out split",
                "",
                *_table(
                    ["metric", "value"],
                    [
                        ["precision", f"{report.classifier_precision:.2f}"],
                        ["recall", f"{report.classifier_recall:.2f}"],
                    ],
                ),
                "",
                "Measured on injections drawn from the classifier's own training",
                "distribution -- its most favourable case, and the reason the headline",
                "table above exists.",
                "",
                f"![classifier held-out scores](figures/{FIGURE_NAMES[3]})",
            ]
        )
    return lines


def _scaled_section(comparison: OODComparison) -> list[str]:
    """Render the same comparison on the scaled catalog, per attack class.

    The small catalog quantises every per-attack rate to a quarter. This
    section is the same measurement with nine times as many clean backgrounds,
    so a rate here can move by 1/36 rather than 1/4 -- and the leakage can be
    located in the goals that produce it rather than averaged over six.
    """
    in_scores = comparison.in_distribution
    out_scores = comparison.out_of_distribution
    rows = [
        [_short(name), f"{scores[0]:.2f}", f"{scores[1]:.2f}"]
        for name, scores in comparison.per_attack_classifier_asr.items()
    ]
    return [
        f"## Scaled catalog: {comparison.probe_pages} probe pages instead of four",
        "",
        "The identical comparison on the forty-product catalog (eight merchants, two",
        "unverified), with the classifier retrained on that catalog's own corpus so it",
        "keeps the same in-distribution advantage. Each payload is measured against",
        f"{comparison.probe_pages} clean listings rather than four, so per-attack rates",
        "are no longer multiples of a quarter.",
        "",
        *_table(
            ["injections", "n", "classifier miss rate", "mean confidence"],
            [
                [
                    "in-distribution (templates it trained on)",
                    str(in_scores.n),
                    f"{in_scores.miss_rate:.2f}",
                    f"{in_scores.mean_confidence:.2f}",
                ],
                [
                    "out-of-distribution (red-team phrasings)",
                    str(out_scores.n),
                    f"{out_scores.miss_rate:.2f}",
                    f"{out_scores.mean_confidence:.2f}",
                ],
            ],
        ),
        "",
        f"Miss-rate gap: **{comparison.miss_rate_gap:+.2f}**. "
        f"Mean-confidence drop: **{comparison.confidence_drop:.2f}**. "
        f"Guardrail ASR: {comparison.guardrail_asr_in:.2f} in-distribution, "
        f"{comparison.guardrail_asr_out:.2f} out-of-distribution.",
        "",
        "### Where the classifier leaks, by attack class",
        "",
        "Attack success rate against the reference agent under `ContentClassifierDefense`,",
        "fixed natural-language template versus red-team phrasing of the same goal:",
        "",
        *_table(["attack class", "in-distribution ASR", "out-of-distribution ASR"], rows),
        "",
        "Payee redirect and merchant spoofing stay at zero because the reference agent",
        "only acts on the two account identifiers the canned variants name, and both",
        "are tokens the classifier was trained on. The leakage sits in the goals that",
        "can be written in plain English.",
        "",
        f"![classifier leakage by attack class, scaled catalog](figures/{FIGURE_NAMES[4]})",
    ]


def _submissions_section(metrics: Sequence[DefenseMetrics]) -> list[str]:
    """Render the submitted-defenses table.

    Currently the four built-ins. External submissions scored through
    ``--defense`` appear here as they arrive.
    """
    rows = []
    for m in metrics:
        origin = "built-in" if m.defense_name in BUILT_IN_DEFENSES else "submitted"
        rows.append(
            [
                f"`{m.defense_name}`",
                origin,
                f"{m.benign_utility:.2f}",
                f"{m.attack_success_rate:.2f}",
                f"{m.over_block_rate:.2f}",
            ]
        )

    return [
        "## Submitted defenses",
        "",
        *_table(
            [
                "defense",
                "origin",
                "benign utility",
                "attack success rate",
                "over-block rate",
            ],
            rows,
        ),
        "",
        "External submissions are welcome -- see [docs/SUBMITTING.md](../docs/SUBMITTING.md).",
        "**This ordering is not a certification.** It records how a defense scored on",
        "this benchmark's own scenarios, which is a much narrower claim than being",
        "safe in deployment. Lure currently measures defenses written by its own",
        "author against attacks written by its own author; see the limitations",
        "section of [docs/FINDINGS.md](../docs/FINDINGS.md) before drawing",
        "conclusions from any row here.",
    ]


def build_results_markdown(
    report: BenchmarkReport,
    comparison: OODComparison,
    scaled_comparison: OODComparison | None = None,
) -> str:
    """Return the full ``RESULTS.md`` body.

    Args:
        report: Offline benchmark results.
        comparison: In- versus out-of-distribution comparison on the small
            catalog.
        scaled_comparison: The same comparison on the scaled catalog, if run.

    Returns:
        Markdown text, deterministic and free of timestamps.
    """
    lines = [
        "# Lure results",
        "",
        "Generated by `python -m lure.cli reproduce`. **Do not edit by hand** --",
        "every number here is read from the same code paths the test suite asserts",
        "on, and any edit is overwritten on the next run.",
        "",
        "Deliberately timestamp-free, so `git diff` shows only numbers that moved.",
        "",
        "---",
        "",
        *_headline_section(comparison),
        "",
        "---",
        "",
        *_core_section(report),
        "",
        "---",
        "",
        *(
            [*_scaled_section(scaled_comparison), "", "---", ""]
            if scaled_comparison is not None
            else []
        ),
        *_submissions_section(report.metrics),
        "",
        "---",
        "",
        "## Reproducing",
        "",
        "```bash",
        "make reproduce      # or: python -m lure.cli reproduce",
        "```",
        "",
        "Runs fully offline with no API key, and regenerates every table, CSV and",
        "figure byte-identically. `results/metrics.csv` is the portable interface if",
        "you would rather plot these numbers yourself.",
        "",
    ]
    return "\n".join(lines)


def write_results_markdown(
    report: BenchmarkReport,
    comparison: OODComparison,
    path: Path = DEFAULT_RESULTS_PATH,
    scaled_comparison: OODComparison | None = None,
) -> Path:
    """Write ``RESULTS.md`` to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_results_markdown(report, comparison, scaled_comparison), encoding="utf-8"
    )
    return path
