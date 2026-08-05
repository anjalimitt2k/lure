"""Tests for the ``run-llm`` command's behaviour without credentials.

The contract is that an unconfigured optional tier is a *state*, not a
failure: it explains itself and exits 0. Anything else would break CI that
runs the suite without secrets, and would make the offline core feel
conditional on the online one.
"""

from __future__ import annotations

import csv

import pytest

from lure.cli import build_parser, main
from lure.llm.client import API_KEY_ENV, MODEL_ENV


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch):
    """Guarantee the LLM tier is off, whatever the developer's shell has set."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)


def test_run_llm_without_credentials_exits_zero(capsys) -> None:
    exit_code = main(["run-llm"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip()


def test_run_llm_without_credentials_explains_how_to_enable(capsys) -> None:
    main(["run-llm"])
    out = capsys.readouterr().out

    assert API_KEY_ENV in out
    assert MODEL_ENV in out
    assert "claude-haiku-4-5-20251001" in out
    assert "not enabled" in out


def test_run_llm_without_credentials_writes_nothing(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"

    main(["run-llm", "--csv", str(csv_path)])

    assert not csv_path.exists()


def test_run_llm_with_only_a_key_still_declines_cleanly(monkeypatch, capsys) -> None:
    monkeypatch.setenv(API_KEY_ENV, "sk-test")

    assert main(["run-llm"]) == 0
    assert "not enabled" in capsys.readouterr().out


def test_the_offline_core_is_unaffected_by_the_missing_tier(tmp_path, capsys) -> None:
    csv_path = tmp_path / "metrics.csv"

    assert main(["run", "--csv", str(csv_path)]) == 0
    assert "DeterministicGuardrails" in capsys.readouterr().out
    assert csv_path.exists()


def test_run_llm_appends_rather_than_replacing(tmp_path) -> None:
    """The LLM tier extends the offline results; it must not clobber them."""
    csv_path = tmp_path / "metrics.csv"
    main(["run", "--csv", str(csv_path)])

    with csv_path.open(encoding="utf-8") as handle:
        before = list(csv.reader(handle))

    main(["run-llm", "--csv", str(csv_path)])

    with csv_path.open(encoding="utf-8") as handle:
        after = list(csv.reader(handle))

    assert after == before


def test_both_subcommands_are_registered() -> None:
    parser = build_parser()

    assert parser.parse_args(["run"]).command == "run"
    assert parser.parse_args(["run-llm"]).command == "run-llm"
    assert parser.parse_args(["run-llm", "--redteam-variants", "0"]).redteam_variants == 0


def test_no_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
