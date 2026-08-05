"""Tests for the external-defense submission path.

The submission protocol is only real if a third party can point the CLI at a
class and get scored. These tests exercise that end to end with the in-repo
example, and check that every way of getting the dotted path wrong produces a
message naming the problem rather than a traceback.
"""

from __future__ import annotations

import csv

import pytest

from lure.cli import main
from lure.defenses.example_submission import ExampleSubmissionDefense
from lure.eval.loader import DefenseLoadError, load_defense
from lure.eval.report import run_benchmark

EXAMPLE_PATH = "lure.defenses.example_submission.ExampleSubmissionDefense"


# -- loading ---------------------------------------------------------------


def test_loads_a_defense_from_a_dotted_path() -> None:
    defense = load_defense(EXAMPLE_PATH)

    assert isinstance(defense, ExampleSubmissionDefense)
    assert defense.name == "ExampleSubmissionDefense"


def test_loads_a_built_in_defense_too() -> None:
    defense = load_defense("lure.defenses.guardrails.DeterministicGuardrails")

    assert defense.name == "DeterministicGuardrails"


def test_rejects_a_path_with_no_dot() -> None:
    with pytest.raises(DefenseLoadError, match="dotted path"):
        load_defense("NotADottedPath")


def test_rejects_a_missing_module() -> None:
    with pytest.raises(DefenseLoadError, match="could not import module"):
        load_defense("lure.defenses.no_such_module.Thing")


def test_rejects_a_missing_class() -> None:
    with pytest.raises(DefenseLoadError, match="has no attribute"):
        load_defense("lure.defenses.guardrails.NoSuchDefense")


def test_rejects_something_that_is_not_a_defense() -> None:
    with pytest.raises(DefenseLoadError, match="not a subclass"):
        load_defense("lure.environment.models.Cart")


def test_rejects_a_defense_that_needs_constructor_arguments() -> None:
    with pytest.raises(DefenseLoadError, match="no arguments"):
        load_defense("tests.test_submission.NeedsArguments")


def test_error_messages_name_the_expected_shape() -> None:
    with pytest.raises(DefenseLoadError) as excinfo:
        load_defense("lure.environment.models.Cart")

    assert "validate(" in str(excinfo.value)


# -- scoring ---------------------------------------------------------------


def test_a_submitted_defense_can_be_scored() -> None:
    report = run_benchmark(defenses=[ExampleSubmissionDefense()])

    assert [m.defense_name for m in report.metrics] == ["ExampleSubmissionDefense"]
    assert report.metrics[0].attack_total == 24


def test_the_example_submission_scores_as_documented() -> None:
    """A partial defense should look partial -- that is why it ships."""
    metrics = run_benchmark(defenses=[ExampleSubmissionDefense()]).metrics[0]

    assert metrics.per_attack_asr["PayeeRedirectAttack"] == 0.0
    assert metrics.per_attack_asr["MerchantSpoofingAttack"] == 0.0
    assert metrics.per_attack_asr["ShippingExfiltrationAttack"] == 1.0
    assert metrics.per_attack_asr["PriceInflationAttack"] == 1.0
    assert metrics.benign_utility == 1.0
    assert 0.0 < metrics.attack_success_rate < 1.0


# -- through the CLI -------------------------------------------------------


def test_cli_scores_a_submitted_defense(tmp_path, capsys) -> None:
    csv_path = tmp_path / "metrics.csv"

    exit_code = main(["run", "--defense", EXAMPLE_PATH, "--csv", str(csv_path)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Scoring submitted defenses: ExampleSubmissionDefense" in out
    assert "ExampleSubmissionDefense" in out
    assert csv_path.exists()


def test_cli_writes_the_submitted_defense_to_csv(tmp_path, capsys) -> None:
    csv_path = tmp_path / "metrics.csv"
    main(["run", "--defense", EXAMPLE_PATH, "--csv", str(csv_path)])
    capsys.readouterr()

    with csv_path.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["scope"] == "overall"]

    assert [r["defense"] for r in rows] == ["ExampleSubmissionDefense"]


def test_cli_accepts_several_defenses(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--defense",
            EXAMPLE_PATH,
            "--defense",
            "lure.defenses.no_defense.NoDefense",
            "--csv",
            str(tmp_path / "m.csv"),
        ]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "ExampleSubmissionDefense" in out
    assert "NoDefense" in out


def test_cli_reports_a_bad_path_clearly(tmp_path, capsys) -> None:
    exit_code = main(["run", "--defense", "nope.NotReal", "--csv", str(tmp_path / "m.csv")])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error:" in captured.err
    assert "could not import module" in captured.err


def test_cli_without_defense_scores_the_built_ins(tmp_path, capsys) -> None:
    main(["run", "--csv", str(tmp_path / "m.csv")])
    out = capsys.readouterr().out

    assert "NoDefense" in out
    assert "CompositeDefense" in out
    assert "Scoring submitted defenses" not in out


class NeedsArguments(ExampleSubmissionDefense):
    """A defense that cannot be constructed bare, for the error-path test."""

    def __init__(self, required: str) -> None:
        self.required = required
