from __future__ import annotations
import json
from pathlib import Path
import pytest
from dotenv import load_dotenv
from driftlens.output.schema import DriftReport, JudgeScore

load_dotenv()


def get_latest_report() -> DriftReport | None:
    reports_dir = Path("outputs/reports")
    if not reports_dir.exists():
        return None
    paths = sorted(reports_dir.glob("report_*.json"), reverse=True)
    if not paths:
        return None
    return DriftReport(**json.loads(paths[0].read_text()))


def test_reports_exist():
    reports_dir = Path("outputs/reports")
    assert reports_dir.exists(), "outputs/reports directory not found"
    reports = list(reports_dir.glob("report_*.json"))
    assert len(reports) >= 5, f"Expected at least 5 reports, found {len(reports)}"


def test_report_schema_valid():
    report = get_latest_report()
    assert report is not None
    assert report.batch_id != ""
    assert report.alert is not None
    assert report.explanation is not None
    assert report.recommendation is not None


def test_report_explanation_not_empty():
    report = get_latest_report()
    assert report is not None
    assert len(report.explanation.explanation) > 50
    assert len(report.explanation.features_cited) > 0


def test_report_recommendation_not_empty():
    report = get_latest_report()
    assert report is not None
    assert len(report.recommendation.recommendation) > 20
    assert len(report.recommendation.features_implicated) > 0


def test_judge_score_on_report():
    from driftlens.eval.judge import score_report
    report = get_latest_report()
    assert report is not None
    score: JudgeScore = score_report(report)
    assert 1 <= score.factual_grounding <= 5
    assert 1 <= score.causal_coherence <= 5
    assert 1 <= score.recommendation_appropriateness <= 5
    assert score.average >= 1.0
    assert score.rationale != ""


def test_eval_results_saved():
    results_path = Path("outputs/eval_results.json")
    assert results_path.exists(), "eval_results.json not found — run run_eval.py first"
    results = json.loads(results_path.read_text())
    assert len(results) == 5
    for r in results:
        assert "average" in r
        assert r["average"] >= 1.0