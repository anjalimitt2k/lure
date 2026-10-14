"""Figures for the Lure write-up, regenerated from live results.

Every number plotted here is read out of a
:class:`~lure.eval.report.BenchmarkReport` or an
:class:`~lure.redteam.ood_eval.OODComparison` produced by the same code paths
the tests assert on. Nothing is hardcoded, so a figure cannot quietly disagree
with the table beside it.

**The CSV is the portable interface.** ``results/metrics.csv`` carries every
number these figures show, in long format with ``tier``/``agent``/``style``
labels. Anyone who would rather work in R and ggplot2, or in a spreadsheet, can
read that file directly and ignore this module entirely; matplotlib is an
implementation detail of the bundled figures, not a dependency of the results.

Determinism: the Agg backend is selected explicitly, style is set from a fixed
dictionary rather than a named stylesheet that may vary by version, and PNG
metadata is suppressed so no build timestamp is embedded. Repeated runs on one
machine produce byte-identical files.

Colour follows a validated categorical palette (checked for colour-vision
separation on the light surface). Two of the four slots fall below 3:1 contrast
on that surface, so every non-zero bar carries a visible value label and the
same numbers appear as tables in ``results/RESULTS.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from lure.eval.report import BenchmarkReport  # noqa: E402
from lure.redteam.ood_eval import OODComparison  # noqa: E402

__all__ = [
    "DEFAULT_FIGURE_DIR",
    "FIGURE_NAMES",
    "generate_all_figures",
    "plot_asr_by_defense",
    "plot_classifier_pr",
    "plot_defense_tradeoff",
    "plot_ood_by_attack",
    "plot_ood_gap",
]

#: Where figures are written, relative to the working directory.
DEFAULT_FIGURE_DIR = Path("results") / "figures"

#: Filenames produced, in generation order.
FIGURE_NAMES = (
    "fig_asr_by_defense.png",
    "fig_defense_tradeoff.png",
    "fig_ood_gap.png",
    "fig_classifier_pr.png",
    "fig_ood_by_attack.png",
)

# -- fixed style -----------------------------------------------------------

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_GRID = "#dcdbd6"

#: Categorical slots in fixed order. Never cycled, never reordered per chart.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

FIGSIZE = (9.0, 5.0)
DPI = 150

#: Suppresses matplotlib's default Software chunk, which would otherwise embed
#: a version string and break byte-identical reruns across environments.
_PNG_METADATA: dict[str, str] = {"Software": ""}


def _apply_style() -> None:
    """Set every rcParam the figures depend on, explicitly.

    Setting these by hand rather than via a named stylesheet keeps output
    stable across matplotlib versions that revise their built-in styles.
    """
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": INK_GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK_GRID,
            "grid.linewidth": 0.6,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def _finish(ax: plt.Axes, subtitle: str) -> None:
    """Strip chart junk and add a recessive explanatory subtitle."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_axisbelow(True)
    if subtitle:
        ax.set_xlabel(subtitle, color=INK_SECONDARY, fontsize=9, labelpad=12)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write ``fig`` to ``path`` deterministically and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", metadata=_PNG_METADATA)
    plt.close(fig)
    return path


def _short(attack_name: str) -> str:
    """Shorten an attack class name for an axis label."""
    return attack_name.removeprefix("RedTeam:").removesuffix("Attack")


# -- figures ---------------------------------------------------------------


def plot_asr_by_defense(report: BenchmarkReport, path: Path) -> Path:
    """Grouped bars: attack success rate per defense across the attack classes.

    Reads the per-attack breakdown straight off the report, so adding an attack
    class widens the chart with no edit here.
    """
    _apply_style()
    attack_names = sorted({n for m in report.metrics for n in m.per_attack_asr})
    defenses = list(report.metrics)

    fig, ax = plt.subplots()
    group_width = 0.8
    bar_width = group_width / len(defenses) * 0.92

    for index, metrics in enumerate(defenses):
        offset = -group_width / 2 + bar_width / 2 + index * (group_width / len(defenses))
        positions = [i + offset for i in range(len(attack_names))]
        values = [metrics.per_attack_asr.get(name, 0.0) for name in attack_names]
        ax.bar(
            positions,
            values,
            width=bar_width,
            label=metrics.defense_name,
            color=SERIES[index % len(SERIES)],
        )
        # Relief rule: label the bars that have height. Zero bars have no fill
        # to misread, and labelling all 24 would be noise.
        for x, value in zip(positions, values):
            if value > 0:
                ax.text(
                    x,
                    value + 0.02,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK_PRIMARY,
                )

    ax.set_xticks(range(len(attack_names)))
    ax.set_xticklabels([_short(n) for n in attack_names], rotation=20, ha="right")
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_title("Attack success rate by defense and attack class")
    # Outside the axes: at ASR 1.00 the tallest bars reach the top of the plot,
    # so any in-axes placement collides with them or their value labels.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1)
    _finish(ax, "Unlabelled bars are 0.00. Lower is better.")
    return _save(fig, path)


def plot_defense_tradeoff(report: BenchmarkReport, path: Path) -> Path:
    """Scatter: benign utility against overall ASR, one point per defense.

    The question this answers is whether a defense had to break shopping in
    order to stop fraud. Identity is carried by direct labels rather than by
    colour, so no colour-separation cap applies.
    """
    _apply_style()
    fig, ax = plt.subplots()

    # Defenses that score identically land on the same point. Grouping them
    # into one labelled marker keeps the coordinates exact -- jittering would
    # invent separation that the results do not contain -- and the shared label
    # states the tie outright, which is itself a finding.
    groups: dict[tuple[float, float], list[str]] = {}
    for metrics in report.metrics:
        key = (round(metrics.attack_success_rate, 4), round(metrics.benign_utility, 4))
        groups.setdefault(key, []).append(metrics.defense_name)

    points = sorted(groups.items())
    ax.scatter(
        [x for (x, _), _ in points],
        [y for (_, y), _ in points],
        s=90,
        color=SERIES[0],
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.5,
    )

    # Fixed alternating offsets, indexed rather than searched, so label
    # placement is deterministic across runs.
    offsets = ((12, -30), (12, 14), (12, -14), (12, 30))
    for index, ((x, y), names) in enumerate(points):
        ax.annotate(
            "\n".join(names),
            (x, y),
            textcoords="offset points",
            xytext=offsets[index % len(offsets)],
            fontsize=9,
            color=INK_PRIMARY,
            linespacing=1.4,
        )

    ax.set_xlabel("attack success rate")
    ax.set_ylabel("benign utility")
    ax.set_xlim(-0.08, 1.25)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Does the defense break shopping in order to stop fraud?")
    _finish(
        ax,
        "Top-left is best: stops every attack, blocks no honest purchase. "
        "Coincident defenses share one marker.",
    )
    return _save(fig, path)


def plot_ood_gap(comparison: OODComparison, path: Path) -> Path:
    """The headline: classifier miss rate in versus out of distribution.

    The deterministic guardrails are drawn as a reference line at their
    measured attack success rate, which is what makes the contrast legible --
    one defense degrades when the wording changes and the other does not.
    """
    _apply_style()
    fig, ax = plt.subplots()

    labels = ["in-distribution\n(templates it trained on)", "out-of-distribution\n(red-team phrasings)"]
    values = [
        comparison.in_distribution.miss_rate,
        comparison.out_of_distribution.miss_rate,
    ]
    confidences = [
        comparison.in_distribution.mean_confidence,
        comparison.out_of_distribution.mean_confidence,
    ]
    counts = [comparison.in_distribution.n, comparison.out_of_distribution.n]

    bars = ax.bar([0, 1], values, width=0.5, color=SERIES[0], zorder=3)
    for bar, value, confidence, count in zip(bars, values, confidences, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=INK_PRIMARY,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.028,
            f"mean confidence {confidence:.2f}   n={count}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
        )

    guardrail = max(comparison.guardrail_asr_in, comparison.guardrail_asr_out)
    ax.axhline(guardrail, linestyle="--", linewidth=1.4, color=SERIES[1], zorder=4)
    # Below the line: the only band guaranteed clear of both bars at any scale.
    ax.text(
        1.45,
        guardrail - 0.030,
        f"DeterministicGuardrails: {guardrail:.2f} on both",
        ha="right",
        va="bottom",
        fontsize=9,
        color=SERIES[1],
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("content classifier miss rate")
    ax.set_ylim(-0.045, max(values) * 1.6 + 0.02)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="x", visible=False)
    ax.set_title("A content filter's strength does not survive new wording")
    _finish(
        ax,
        "Same six fraud goals, rephrased. Outcome-level guardrails read no text "
        "and do not degrade.",
    )
    return _save(fig, path)


def plot_classifier_pr(report: BenchmarkReport, path: Path) -> Path:
    """Bars: the classifier's held-out precision and recall.

    Reported for context rather than as a headline. These are the scores that
    look reassuring right up until the distribution shifts.
    """
    _apply_style()
    fig, ax = plt.subplots()

    precision = report.classifier_precision or 0.0
    recall = report.classifier_recall or 0.0
    values = [precision, recall]

    bars = ax.bar([0, 1], values, width=0.5, color=SERIES[0], zorder=3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=INK_PRIMARY,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["precision", "recall"])
    ax.set_ylabel("score on held-out split")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="x", visible=False)
    ax.set_title("ContentClassifierDefense held-out scores")
    _finish(
        ax,
        "Measured on injections drawn from its own training distribution -- "
        "its most favourable case.",
    )
    return _save(fig, path)


def plot_ood_by_attack(comparison: OODComparison, path: Path) -> Path:
    """Paired bars: classifier ASR per attack class, in versus out of distribution.

    Drawn from the scaled catalog so the bars are not quantised to quarters.
    This is the figure that locates the leakage: an averaged miss rate says
    the filter degrades; this says which fraud goals it degrades on.
    """
    _apply_style()
    fig, ax = plt.subplots()

    names = list(comparison.per_attack_classifier_asr)
    in_values = [comparison.per_attack_classifier_asr[n][0] for n in names]
    out_values = [comparison.per_attack_classifier_asr[n][1] for n in names]
    width = 0.38
    left = [i - width / 2 for i in range(len(names))]
    right = [i + width / 2 for i in range(len(names))]

    ax.bar(left, in_values, width=width, color=SERIES[0], label="in-distribution", zorder=3)
    ax.bar(right, out_values, width=width, color=SERIES[1], label="out-of-distribution", zorder=3)
    for positions, values in ((left, in_values), (right, out_values)):
        for x, value in zip(positions, values):
            if value > 0:
                ax.text(x, value + 0.01, f"{value:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK_PRIMARY)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_short(n) for n in names], rotation=20, ha="right")
    ax.set_ylabel("attack success rate under ContentClassifierDefense")
    top = max([*in_values, *out_values, 0.05])
    ax.set_ylim(0, top * 1.5)
    ax.grid(axis="x", visible=False)
    ax.set_title(
        f"Where the content filter leaks ({comparison.catalog} catalog, "
        f"{comparison.probe_pages} probe pages)"
    )
    ax.legend(loc="upper left")
    _finish(ax, "Unlabelled bars are 0.00. Same six goals, template wording versus red-team wording.")
    return _save(fig, path)


def generate_all_figures(
    report: BenchmarkReport,
    comparison: OODComparison,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    scaled_comparison: OODComparison | None = None,
) -> list[Path]:
    """Generate every figure and return the paths written, in order.

    Args:
        report: Offline benchmark results.
        comparison: In- versus out-of-distribution comparison.
        output_dir: Directory to write PNGs into. Created if absent.
        scaled_comparison: The comparison on the scaled catalog. When omitted
            the per-attack figure is drawn from ``comparison`` instead.
    """
    output_dir = Path(output_dir)
    return [
        plot_asr_by_defense(report, output_dir / FIGURE_NAMES[0]),
        plot_defense_tradeoff(report, output_dir / FIGURE_NAMES[1]),
        plot_ood_gap(comparison, output_dir / FIGURE_NAMES[2]),
        plot_classifier_pr(report, output_dir / FIGURE_NAMES[3]),
        plot_ood_by_attack(scaled_comparison or comparison, output_dir / FIGURE_NAMES[4]),
    ]
