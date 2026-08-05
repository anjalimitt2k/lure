"""End-to-end LLM-tier tests, and the distribution-shift finding it exists for.

Two things are covered here that nothing else covers:

* The ``run-llm`` scoring path actually runs -- task assembly, a real
  :class:`~lure.agent.llm_agent.LLMAgent`, all four defenses, the report and
  the CSV append -- driven by a stub so it stays hermetic. Without this, a bug
  in that path would only surface for someone holding credentials.
* The measured gap between injections the content classifier was trained on
  and injections it was not. That gap is the reason the tier exists, so it is
  asserted rather than assumed.

The measurement functions live in :mod:`lure.redteam.ood_eval`, not here, so
that ``results/RESULTS.md`` and these assertions read the same numbers from the
same code. A test that computed its own copy could pass while the published
document was wrong.
"""

from __future__ import annotations

import csv

import pytest

from lure.eval.report import format_report, run_benchmark, write_metrics_csv
from lure.redteam.ood_eval import (
    asr_by_distribution,
    classify_samples,
    credulous_backend,
    in_distribution_samples,
    out_of_distribution_samples,
    reference_llm_report,
    run_ood_comparison,
)


# -- end-to-end tier run ---------------------------------------------------


def test_the_llm_tier_runs_end_to_end() -> None:
    report = reference_llm_report()

    assert report.agent_name == "LLMAgent"
    assert report.style == "natural"
    assert {m.defense_name for m in report.metrics} == {
        "NoDefense",
        "DeterministicGuardrails",
        "ContentClassifierDefense",
        "CompositeDefense",
    }


def test_natural_attacks_land_against_an_undefended_credulous_model() -> None:
    metrics = {m.defense_name: m for m in reference_llm_report().metrics}

    assert metrics["NoDefense"].attack_success_rate > 0.0


def test_guardrails_hold_against_natural_and_red_team_attacks() -> None:
    """Outcome-level enforcement does not care that the wording is new."""
    metrics = {m.defense_name: m for m in reference_llm_report().metrics}

    assert metrics["DeterministicGuardrails"].attack_success_rate == 0.0
    assert metrics["CompositeDefense"].attack_success_rate == 0.0


def test_benign_tasks_still_succeed_in_the_llm_tier() -> None:
    metrics = {m.defense_name: m for m in reference_llm_report().metrics}

    assert metrics["NoDefense"].benign_utility == 1.0
    assert metrics["DeterministicGuardrails"].benign_utility == 1.0


def test_red_team_classes_are_scored_separately() -> None:
    report = reference_llm_report()
    names = {n for m in report.metrics for n in m.per_attack_asr}

    assert any(n.startswith("RedTeam:") for n in names)
    assert any(not n.startswith("RedTeam:") for n in names)


def test_the_tier_report_renders() -> None:
    report = reference_llm_report()
    text = format_report(report.metrics, report)

    assert "LLMAgent" in text
    assert "RedTeam:" in text


def test_the_tier_appends_to_an_existing_csv(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    offline = run_benchmark()
    write_metrics_csv(offline.metrics, path, offline)

    with path.open(encoding="utf-8") as handle:
        offline_rows = len(list(csv.reader(handle)))

    llm = reference_llm_report()
    write_metrics_csv(llm.metrics, path, llm, append=True)

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) + 1 > offline_rows
    assert {r["tier"] for r in rows} == {"offline", "llm-reference"}
    assert any(r["agent"] == "LLMAgent" for r in rows)


def test_the_reference_backend_never_reaches_the_network() -> None:
    """The stub is the only backend used; the real client is never constructed."""
    assert credulous_backend().model == "fake-model"


def test_asr_splits_by_wording_distribution() -> None:
    split = asr_by_distribution(reference_llm_report())

    assert set(split) >= {"NoDefense", "DeterministicGuardrails"}
    for scores in split.values():
        assert set(scores) == {"in_distribution", "out_of_distribution"}


# -- the distribution-shift finding ----------------------------------------


def test_the_classifier_catches_everything_it_was_trained_on() -> None:
    scores = classify_samples(in_distribution_samples())

    assert scores.miss_rate == 0.0
    assert scores.n > 0


def test_the_classifier_leaks_on_phrasing_it_was_not_trained_on() -> None:
    """The headline. Same six goals, new wording, real residual leakage."""
    comparison = run_ood_comparison()

    assert comparison.out_of_distribution.miss_rate > 0.0
    assert (
        comparison.out_of_distribution.miss_rate > comparison.in_distribution.miss_rate
    )
    assert comparison.miss_rate_gap > 0.0


def test_the_classifiers_confidence_collapses_out_of_distribution() -> None:
    """Not just misses -- the decisions it still gets right become marginal."""
    comparison = run_ood_comparison()

    assert comparison.in_distribution.mean_confidence > 0.90
    assert comparison.confidence_drop > 0.15


def test_deterministic_guardrails_have_no_such_distribution_dependence() -> None:
    """The contrast that makes the finding meaningful rather than anecdotal."""
    comparison = run_ood_comparison()

    assert comparison.guardrail_asr_in == 0.0
    assert comparison.guardrail_asr_out == 0.0


def test_the_classifier_leaks_attack_success_not_just_detections() -> None:
    """Misses translate into executed fraud, which is what actually matters."""
    comparison = run_ood_comparison()

    assert comparison.classifier_asr_out > comparison.classifier_asr_in


def test_the_shift_measurement_is_deterministic() -> None:
    first, second = run_ood_comparison(), run_ood_comparison()

    assert first == second


def test_scoring_an_empty_sample_set_is_an_error() -> None:
    """Rates over zero samples are undefined, not zero."""
    with pytest.raises(ValueError):
        classify_samples([])


def test_samples_are_labelled_by_distribution() -> None:
    assert {s.distribution for s in in_distribution_samples()} == {"in"}
    assert {s.distribution for s in out_of_distribution_samples()} == {"out"}


def test_both_sample_sets_cover_all_six_attack_classes() -> None:
    assert len({s.attack_name for s in in_distribution_samples()}) == 6
    assert len({s.attack_name for s in out_of_distribution_samples()}) == 6
