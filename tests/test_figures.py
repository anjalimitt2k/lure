"""Tests for figure generation.

Figures are checked for existence, non-emptiness, PNG validity and
determinism. Nothing here inspects pixels: the point is that the files are
produced from live results and are stable, not that they look a particular way.
"""

from __future__ import annotations

import pytest

from lure.eval.report import run_benchmark
from lure.redteam.ood_eval import run_ood_comparison
from lure.viz.figures import (
    FIGURE_NAMES,
    generate_all_figures,
    plot_asr_by_defense,
    plot_classifier_pr,
    plot_defense_tradeoff,
    plot_ood_gap,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def report():
    return run_benchmark()


@pytest.fixture(scope="module")
def comparison():
    return run_ood_comparison()


def test_all_four_figures_are_produced(tmp_path, report, comparison) -> None:
    paths = generate_all_figures(report, comparison, tmp_path)

    assert [p.name for p in paths] == list(FIGURE_NAMES)
    for path in paths:
        assert path.exists(), path.name


def test_every_figure_is_a_non_empty_png(tmp_path, report, comparison) -> None:
    for path in generate_all_figures(report, comparison, tmp_path):
        data = path.read_bytes()
        assert len(data) > 1000, path.name
        assert data.startswith(PNG_MAGIC), path.name


def test_figures_are_byte_identical_across_runs(tmp_path, report, comparison) -> None:
    """No embedded timestamp, so `git diff` only moves when a number does."""
    first = generate_all_figures(report, comparison, tmp_path / "a")
    second = generate_all_figures(report, comparison, tmp_path / "b")

    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes(), left.name


def test_the_output_directory_is_created(tmp_path, report, comparison) -> None:
    target = tmp_path / "not" / "yet" / "there"
    paths = generate_all_figures(report, comparison, target)

    assert all(p.exists() for p in paths)


def test_each_figure_can_be_generated_alone(tmp_path, report, comparison) -> None:
    assert plot_asr_by_defense(report, tmp_path / "a.png").exists()
    assert plot_defense_tradeoff(report, tmp_path / "b.png").exists()
    assert plot_ood_gap(comparison, tmp_path / "c.png").exists()
    assert plot_classifier_pr(report, tmp_path / "d.png").exists()


def test_figures_track_live_results_rather_than_hardcoded_values(
    tmp_path, comparison
) -> None:
    """A changed result must change the figure, or the figure is decoration."""
    full = run_benchmark()
    subset = run_benchmark(defenses=[d for d in _first_two_defenses()])

    a = plot_asr_by_defense(full, tmp_path / "full.png")
    b = plot_asr_by_defense(subset, tmp_path / "subset.png")

    assert a.read_bytes() != b.read_bytes()


def _first_two_defenses():
    from lure.defenses import DeterministicGuardrails, NoDefense

    return [NoDefense(), DeterministicGuardrails()]
