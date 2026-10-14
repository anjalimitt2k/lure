"""Assembling and writing the LLM tier's results.

The offline core writes ``results/RESULTS.md`` and must regenerate it
byte-identically with no credentials. The LLM tier cannot make that promise
from a cold cache, so it writes its own document, ``results/llm_results.md``,
beside the response cache that produced it. With the cache committed, a
second person regenerates the same document with no key and no network.

Two things are reported that the offline core cannot report:

* **What a real model does** when it reads an injected page: attack success
  per defense and per attack class, with the shopping agent driven by the
  configured model rather than by a scripted stand-in.
* **A model-based content filter**, :class:`~lure.defenses.llm_judge.LLMJudgeDefense`,
  scored on the identical in- and out-of-distribution pages as the n-gram
  classifier. This is the test of the write-up's concession that a stronger
  content filter would generalise further.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from lure.defenses.classifier import ContentClassifierDefense
from lure.defenses.llm_judge import LLMJudgeDefense
from lure.environment.marketplace import Marketplace
from lure.eval.report import BenchmarkReport
from lure.redteam.ood_eval import (
    DistributionScores,
    InjectionSample,
    classify_samples,
    in_distribution_samples,
    out_of_distribution_samples,
    per_attack_by_distribution,
)

__all__ = [
    "DEFAULT_LLM_RESULTS_PATH",
    "FilterScores",
    "build_llm_results_markdown",
    "judge_samples",
    "score_content_filters",
    "write_llm_results_markdown",
]

#: Where the LLM tier's document is written, relative to the working directory.
DEFAULT_LLM_RESULTS_PATH = Path("results") / "llm_results.md"


@dataclass(frozen=True)
class FilterScores:
    """One content filter's miss rates in and out of distribution."""

    name: str
    in_distribution: DistributionScores
    out_of_distribution: DistributionScores

    @property
    def miss_rate_gap(self) -> float:
        """Return how much more the filter misses out of distribution."""
        return self.out_of_distribution.miss_rate - self.in_distribution.miss_rate


def judge_samples(samples: list[InjectionSample], judge: LLMJudgeDefense) -> DistributionScores:
    """Score ``samples`` with the LLM judge.

    The judge returns a verdict, not a probability, so ``mean_confidence`` is
    the fraction flagged -- identical to the catch rate. It is reported in the
    same field so the two filters share one table shape.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("cannot score an empty sample set")
    caught = 0
    for sample in samples:
        verdict = judge.judge_text(sample.text)
        if verdict is not None and verdict[0]:
            caught += 1
    return DistributionScores(
        n=len(samples),
        caught=caught,
        miss_rate=1.0 - caught / len(samples),
        mean_confidence=caught / len(samples),
    )


def score_content_filters(
    marketplace: Marketplace, judge: LLMJudgeDefense
) -> list[FilterScores]:
    """Score the n-gram classifier and the LLM judge on the same pages."""
    in_samples = in_distribution_samples(marketplace)
    out_samples = out_of_distribution_samples(marketplace)
    classifier = ContentClassifierDefense(marketplace=marketplace)
    return [
        FilterScores(
            name=ContentClassifierDefense.name,
            in_distribution=classify_samples(in_samples, classifier),
            out_of_distribution=classify_samples(out_samples, classifier),
        ),
        FilterScores(
            name=LLMJudgeDefense.name,
            in_distribution=judge_samples(in_samples, judge),
            out_of_distribution=judge_samples(out_samples, judge),
        ),
    ]


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


def _redteam_provenance(sources: dict[str, int] | None) -> list[str]:
    """Say where the out-of-distribution tasks came from, honestly."""
    if not sources:
        return []
    model_n = sources.get("model", 0)
    canned_n = sources.get("canned", 0)
    if model_n and not canned_n:
        return [f"All {model_n} red-team tasks use injections written by the model.", ""]
    if canned_n and not model_n:
        return [
            f"**The model declined to write injections.** All {canned_n} red-team tasks",
            "use the bundled canned variants instead -- the same author-written",
            "out-of-distribution set the offline comparison uses. The agent and judge",
            "numbers below are real-model numbers; the rephrasings are not.",
            "",
        ]
    return [
        f"Red-team tasks: {model_n} with model-written injections, {canned_n} falling",
        "back to the bundled canned variants where the model declined.",
        "",
    ]


def build_llm_results_markdown(
    report: BenchmarkReport,
    filters: Sequence[FilterScores],
    model: str,
    redteam_sources: dict[str, int] | None = None,
) -> str:
    """Return the full ``llm_results.md`` body.

    Args:
        report: The LLM-tier benchmark run.
        filters: Content-filter scores on the shared probe pages.
        model: Model identifier the tier ran with.
        redteam_sources: ``{"model": n, "canned": m}`` -- how many red-team
            tasks the model wrote and how many fell back to the bundled set.

    Returns:
        Markdown text. Carries the model name, since the numbers are a
        property of it, but no timestamp.
    """
    fixed_names = sorted(
        {n for m in report.metrics for n in m.per_attack_asr if not n.startswith("RedTeam:")}
    )
    lines = [
        "# Lure results: LLM tier",
        "",
        "Generated by `python -m lure.cli run-llm`. **Do not edit by hand.** Every",
        "model response that produced these numbers is in `results/llm_cache.json`;",
        "with that file present the document regenerates with no key and no network.",
        "",
        f"Model: `{model}`. Catalog: `{report.catalog}`. Agent: `{report.agent_name}`.",
        f"{report.metrics[0].benign_total} benign tasks, "
        f"{report.metrics[0].attack_total} attack tasks (fixed natural-language "
        "templates plus red-team variants).",
        "",
        *_redteam_provenance(redteam_sources),
        "---",
        "",
        "## A real model in the agent's seat",
        "",
        "Unlike the offline core, the agent here is not scripted to obey. Attack",
        "success under `NoDefense` is therefore the model's own credulity, and the",
        "gap between it and each defense is what that defense bought.",
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
        "### Attack success rate by attack class",
        "",
        "Fixed natural-language template (`in`) and red-team rephrasing of the same",
        "goal (`out`), per defense:",
        "",
    ]

    header = ["defense", *(f"{_short(n)} in / out" for n in fixed_names)]
    rows = []
    for m in report.metrics:
        split = per_attack_by_distribution(report, m.defense_name)
        rows.append(
            [
                f"`{m.defense_name}`",
                *(
                    f"{split.get(n, (0.0, 0.0))[0]:.2f} / {split.get(n, (0.0, 0.0))[1]:.2f}"
                    for n in fixed_names
                ),
            ]
        )
    lines.extend(_table(header, rows))

    lines.extend(
        [
            "",
            "---",
            "",
            "## Two content filters on the same pages",
            "",
            "The n-gram classifier and the LLM judge, scored on identical poisoned",
            "pages: the fixed templates the classifier was trained on, and the canned",
            "red-team phrasings it was not. The judge was trained on neither. For the",
            "judge, *mean confidence* is simply its catch rate, since it returns a",
            "verdict rather than a probability.",
            "",
            *_table(
                ["filter", "distribution", "n", "miss rate", "mean confidence"],
                [
                    row
                    for f in filters
                    for row in (
                        [
                            f"`{f.name}`",
                            "in",
                            str(f.in_distribution.n),
                            f"{f.in_distribution.miss_rate:.2f}",
                            f"{f.in_distribution.mean_confidence:.2f}",
                        ],
                        [
                            f"`{f.name}`",
                            "out",
                            str(f.out_of_distribution.n),
                            f"{f.out_of_distribution.miss_rate:.2f}",
                            f"{f.out_of_distribution.mean_confidence:.2f}",
                        ],
                    )
                ],
            ),
            "",
            *(
                f"`{f.name}` miss-rate gap: **{f.miss_rate_gap:+.2f}**."
                for f in filters
            ),
            "",
            "---",
            "",
            "## Reading this",
            "",
            "These numbers are one model, one run, one sampled response per prompt,",
            "on a simulated marketplace. They say what *this* model did on *these*",
            "pages. The limitations in `docs/FINDINGS.md` apply in full.",
            "",
        ]
    )
    return "\n".join(lines)


def write_llm_results_markdown(
    report: BenchmarkReport,
    filters: Sequence[FilterScores],
    model: str,
    path: Path = DEFAULT_LLM_RESULTS_PATH,
    redteam_sources: dict[str, int] | None = None,
) -> Path:
    """Write ``llm_results.md`` to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_llm_results_markdown(report, filters, model, redteam_sources), encoding="utf-8"
    )
    return path
