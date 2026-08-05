"""Tests for one-command reproducibility.

The promise Phase 3 makes is that a stranger can clone the repo, run one
command, and regenerate every headline number and figure. These tests hold
that promise to its two hard parts: the command produces every artifact, and
it produces the *same* artifact twice.
"""

from __future__ import annotations

import csv
import re

from lure.cli import main
from lure.eval.report import run_benchmark
from lure.eval.results_doc import build_results_markdown
from lure.redteam.ood_eval import run_ood_comparison
from lure.viz.figures import FIGURE_NAMES

EXPECTED_ARTIFACTS = ("metrics.csv", "RESULTS.md")


def _reproduce_into(tmp_path):
    results_dir = tmp_path / "results"
    exit_code = main(["reproduce", "--results-dir", str(results_dir)])
    return exit_code, results_dir


# -- the command -----------------------------------------------------------


def test_reproduce_exits_zero(tmp_path, capsys) -> None:
    exit_code, _ = _reproduce_into(tmp_path)
    capsys.readouterr()

    assert exit_code == 0


def test_reproduce_writes_every_artifact(tmp_path, capsys) -> None:
    _, results_dir = _reproduce_into(tmp_path)
    capsys.readouterr()

    for name in EXPECTED_ARTIFACTS:
        artifact = results_dir / name
        assert artifact.exists(), name
        assert artifact.stat().st_size > 0, name

    for name in FIGURE_NAMES:
        figure = results_dir / "figures" / name
        assert figure.exists(), name
        assert figure.stat().st_size > 0, name


def test_reproduce_prints_a_summary(tmp_path, capsys) -> None:
    _reproduce_into(tmp_path)
    out = capsys.readouterr().out

    assert "Deterministic core" in out
    assert "out-of-dist." in out
    assert "RESULTS.md" in out
    assert "byte-identically" in out


def test_reproduce_is_byte_identical_across_runs(tmp_path, capsys) -> None:
    """The property that makes a published number checkable."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    main(["reproduce", "--results-dir", str(first)])
    main(["reproduce", "--results-dir", str(second)])
    capsys.readouterr()

    for name in EXPECTED_ARTIFACTS:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    for name in FIGURE_NAMES:
        assert (first / "figures" / name).read_bytes() == (
            second / "figures" / name
        ).read_bytes(), name


def test_reproduce_needs_no_credentials(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LURE_MODEL", raising=False)

    exit_code, _ = _reproduce_into(tmp_path)
    capsys.readouterr()

    assert exit_code == 0


def test_reproduce_creates_a_missing_results_directory(tmp_path, capsys) -> None:
    nested = tmp_path / "deeply" / "nested" / "results"
    exit_code = main(["reproduce", "--results-dir", str(nested)])
    capsys.readouterr()

    assert exit_code == 0
    assert (nested / "RESULTS.md").exists()


# -- RESULTS.md content ----------------------------------------------------


def _results_text(tmp_path, capsys) -> str:
    _, results_dir = _reproduce_into(tmp_path)
    capsys.readouterr()
    return (results_dir / "RESULTS.md").read_text(encoding="utf-8")


def test_results_md_carries_every_table(tmp_path, capsys) -> None:
    text = _results_text(tmp_path, capsys)

    assert "Deterministic core" in text
    assert "Attack success rate by attack class" in text
    assert "held-out" in text
    assert "Submitted defenses" in text
    assert "out-of-distribution" in text


def test_results_md_lists_all_four_defenses(tmp_path, capsys) -> None:
    text = _results_text(tmp_path, capsys)

    for name in (
        "NoDefense",
        "DeterministicGuardrails",
        "ContentClassifierDefense",
        "CompositeDefense",
    ):
        assert name in text, name


def test_results_md_embeds_every_figure(tmp_path, capsys) -> None:
    text = _results_text(tmp_path, capsys)

    for name in FIGURE_NAMES:
        assert name in text, name


def test_results_md_has_no_timestamp(tmp_path, capsys) -> None:
    """A generated-on line would make every diff noisy and hide real changes."""
    text = _results_text(tmp_path, capsys)

    # Actual date/time shapes, not the word "timestamp" -- the document says
    # "timestamp-free" in its own header, which is the opposite of a violation.
    for pattern in (
        r"\d{4}-\d{2}-\d{2}",  # ISO date
        r"\d{2}:\d{2}",  # clock time
        r"(?i)generated on",
        r"(?i)\bUTC\b",
    ):
        assert not re.search(pattern, text), pattern


def test_results_md_states_the_non_certification_scope(tmp_path, capsys) -> None:
    text = _results_text(tmp_path, capsys)

    assert "not a certification" in text
    assert "SUBMITTING.md" in text


def test_results_md_numbers_match_the_live_report(tmp_path, capsys) -> None:
    """RESULTS.md cannot drift from the values the acceptance tests pin."""
    report = run_benchmark()
    text = build_results_markdown(report, run_ood_comparison())

    for metrics in report.metrics:
        row = (
            f"| `{metrics.defense_name}` | {metrics.benign_utility:.2f} | "
            f"{metrics.attack_success_rate:.2f} | {metrics.over_block_rate:.2f} |"
        )
        assert row in text, metrics.defense_name


def test_results_md_headline_matches_the_comparison(tmp_path, capsys) -> None:
    comparison = run_ood_comparison()
    text = build_results_markdown(run_benchmark(), comparison)

    assert f"{comparison.out_of_distribution.miss_rate:.2f}" in text
    assert f"{comparison.in_distribution.mean_confidence:.2f}" in text


# -- the CSV as portable interface ----------------------------------------


def test_metrics_csv_is_readable_and_labelled(tmp_path, capsys) -> None:
    _, results_dir = _reproduce_into(tmp_path)
    capsys.readouterr()

    with (results_dir / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert {"tier", "agent", "style", "defense", "scope"} <= set(rows[0])
    overall = {r["defense"]: r for r in rows if r["scope"] == "overall"}
    assert float(overall["NoDefense"]["attack_success_rate"]) == 1.0
    assert float(overall["DeterministicGuardrails"]["attack_success_rate"]) == 0.0
