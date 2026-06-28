from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import joblib
import pytest
from driftlens.config import get_config
from driftlens.ingestion import DataIngestion
from driftlens.detection.statistical import StatisticalDriftDetector
from driftlens.detection.shap_monitor import SHAPMonitor
from driftlens.agents.monitor_agent import run_monitor_agent
from driftlens.output.schema import DriftSignal, Severity
import uuid
from dotenv import load_dotenv
load_dotenv()


def make_signal(psi_scores: dict, kl_scores: dict, shap_deltas: dict) -> DriftSignal:
    return DriftSignal(
        batch_id=str(uuid.uuid4())[:8],
        psi_scores=psi_scores,
        kl_scores=kl_scores,
        shap_deltas=shap_deltas,
        adwin_drift_detected=False,
        label_available=False,
    )


def test_monitor_agent_low_severity():
    psi = {f: 0.05 for f in ["LIMIT_BAL", "AGE", "PAY_0"]}
    kl = {f: 0.01 for f in ["LIMIT_BAL", "AGE", "PAY_0"]}
    shap = {f: 0.001 for f in ["LIMIT_BAL", "AGE", "PAY_0"]}
    signal = make_signal(psi, kl, shap)
    alert, payload = run_monitor_agent(signal)
    assert payload.severity == Severity.LOW
    assert alert.severity == Severity.LOW


def test_monitor_agent_high_severity():
    psi = {"LIMIT_BAL": 1.5, "AGE": 0.5, "PAY_0": 0.3}
    kl = {"LIMIT_BAL": 0.8, "AGE": 0.2, "PAY_0": 0.1}
    shap = {"LIMIT_BAL": 0.05, "AGE": 0.01, "PAY_0": 0.005}
    signal = make_signal(psi, kl, shap)
    alert, payload = run_monitor_agent(signal)
    assert payload.severity == Severity.HIGH
    assert alert.top_drifted_features[0] == "LIMIT_BAL"


def test_monitor_agent_ranks_features_correctly():
    psi = {"A": 0.1, "B": 0.5, "C": 0.3}
    kl = {"A": 0.01, "B": 0.05, "C": 0.03}
    shap = {"A": 0.0, "B": 0.0, "C": 0.0}
    signal = make_signal(psi, kl, shap)
    alert, payload = run_monitor_agent(signal)
    ranked_names = [f for f, _ in payload.ranked_features]
    assert ranked_names[0] == "B"
    assert ranked_names[1] == "C"


def test_full_pipeline_no_drift():
    from driftlens.agents.graph import run_pipeline
    from driftlens.agents.tools import register_context

    ingestion = DataIngestion()
    reference = ingestion.get_reference()
    current = pd.read_csv("data/drift_scenarios/scenario_00_no_drift.csv")
    feature_names = ingestion.get_feature_names()

    stat = StatisticalDriftDetector()
    psi, kl = stat.compute(reference, current, feature_names)

    model = joblib.load("models/credit_default.joblib")
    shap_monitor = SHAPMonitor(model, reference)
    shap_deltas = shap_monitor.compute_delta(current)
    shap_ref = shap_monitor.get_ref_mean_abs()
    shap_cur = {f: shap_ref[f] + shap_deltas[f] for f in shap_ref}

    signal = DriftSignal(
        batch_id="test_00",
        psi_scores=psi,
        kl_scores=kl,
        shap_deltas=shap_deltas,
        adwin_drift_detected=False,
        label_available=False,
    )

    report = run_pipeline(signal, reference, current, shap_ref, shap_cur)
    assert report.alert.severity == Severity.LOW
    assert report.explanation.explanation != ""
    assert report.recommendation.recommendation != ""
    assert len(report.explanation.features_cited) > 0