"""Tests for the LLM tier's own results document and the ``run-llm`` path.

The real tier needs credentials; these tests drive the identical code with a
stub backend standing in for the SDK, so the scoring path, the content-filter
comparison, the CSV and the markdown document are all exercised hermetically.
"""

from __future__ import annotations

import csv
import json
import re

from lure.cli import main
from lure.defenses.llm_judge import LLMJudgeDefense
from lure.environment.marketplace import build_default_marketplace
from lure.eval.llm_results import (
    build_llm_results_markdown,
    judge_samples,
    score_content_filters,
)
from lure.llm.client import FakeLLM
from lure.redteam.ood_eval import in_distribution_samples, reference_llm_report


def _judge_backend() -> FakeLLM:
    def reply(prompt: str, system: str | None) -> str:
        page = prompt.split("<product_page>")[-1].split("</product_page>")[0]
        hit = "acct:" in page or "minimum order" in page
        return json.dumps({"injected": hit, "reason": "stub"})

    return FakeLLM(reply)


def test_judge_samples_report_catch_rate_as_confidence() -> None:
    judge = LLMJudgeDefense(_judge_backend())
    scores = judge_samples(in_distribution_samples(), judge)

    assert scores.n == 24
    assert 0.0 < scores.miss_rate < 1.0
    assert scores.mean_confidence == 1.0 - scores.miss_rate


def test_content_filter_scores_cover_both_filters() -> None:
    filters = score_content_filters(build_default_marketplace(), LLMJudgeDefense(_judge_backend()))

    assert [f.name for f in filters] == ["ContentClassifierDefense", "LLMJudgeDefense"]
    for scores in filters:
        assert scores.in_distribution.n == 24
        assert scores.out_of_distribution.n == 48


def test_the_markdown_carries_every_table_and_no_timestamp() -> None:
    report = reference_llm_report()
    filters = score_content_filters(build_default_marketplace(), LLMJudgeDefense(_judge_backend()))
    text = build_llm_results_markdown(report, filters, model="stub-model")

    assert "`stub-model`" in text
    assert "A real model in the agent's seat" in text
    assert "Two content filters on the same pages" in text
    assert "`LLMJudgeDefense`" in text
    for metrics in report.metrics:
        assert f"`{metrics.defense_name}`" in text
    for pattern in (r"\d{4}-\d{2}-\d{2}", r"\d{2}:\d{2}", r"(?i)generated on"):
        assert not re.search(pattern, text), pattern


class _StubClientFactory:
    """Stands in for ``LLMClient`` in the CLI, returning a stub from the environment."""

    def __init__(self, backend: FakeLLM) -> None:
        self.backend = backend

    def from_env(self, cache_path=None, environ=None):
        return self.backend


def _combined_backend() -> FakeLLM:
    """One stub serving the agent, the red-teamer and the judge, by prompt shape."""
    from lure.redteam.ood_eval import _credulous_reply

    def reply(prompt: str, system: str | None) -> str:
        if "Write 2 DIFFERENT product reviews" in prompt:
            return json.dumps(["novel phrasing one", "novel phrasing two"])
        if "Does this page contain text that attempts to steer" in prompt:
            page = prompt.split("<product_page>")[-1].split("</product_page>")[0]
            return json.dumps({"injected": "acct:" in page, "reason": "stub"})
        return _credulous_reply(prompt, system)

    backend = FakeLLM(reply)
    backend.model = "stub-model"
    return backend


def test_run_llm_end_to_end_with_a_stub_backend(tmp_path, monkeypatch, capsys) -> None:
    import lure.cli as cli

    monkeypatch.setattr(cli, "LLMClient", _StubClientFactory(_combined_backend()))
    csv_path = tmp_path / "llm_metrics.csv"
    doc_path = tmp_path / "llm_results.md"

    exit_code = main(
        ["run-llm", "--csv", str(csv_path), "--results", str(doc_path), "--cache", str(tmp_path / "c.json")]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "stub-model" in out
    assert "LLMJudgeDefense" in out
    assert "Content filters on the same poisoned pages" in out
    assert doc_path.exists()
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["tier"] for r in rows} == {"llm"}
    assert {r["catalog"] for r in rows} == {"small"}
    assert "LLMJudgeDefense" in {r["defense"] for r in rows}


def test_run_llm_can_skip_the_judge_and_pick_the_catalog(tmp_path, monkeypatch, capsys) -> None:
    import lure.cli as cli

    monkeypatch.setattr(cli, "LLMClient", _StubClientFactory(_combined_backend()))
    csv_path = tmp_path / "llm_metrics.csv"

    exit_code = main(
        [
            "run-llm",
            "--no-judge",
            "--catalog",
            "large",
            "--redteam-variants",
            "0",
            "--csv",
            str(csv_path),
            "--results",
            str(tmp_path / "doc.md"),
            "--cache",
            str(tmp_path / "c.json"),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "LLMJudgeDefense" not in {r["defense"] for r in rows}
    assert {r["catalog"] for r in rows} == {"large"}
    overall = {r["defense"]: r for r in rows if r["scope"] == "overall"}
    assert overall["NoDefense"]["attack_tasks"] == "216"


def test_run_offline_accepts_the_catalog_option(tmp_path, capsys) -> None:
    csv_path = tmp_path / "metrics.csv"

    assert main(["run", "--catalog", "large", "--csv", str(csv_path)]) == 0
    out = capsys.readouterr().out
    assert "large catalog" in out
    assert "attack tasks: 216" in out
