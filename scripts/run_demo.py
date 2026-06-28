from __future__ import annotations
import os
import uuid
from pathlib import Path
import joblib
import pandas as pd
from driftlens.config import get_config
from driftlens.ingestion import DataIngestion
from driftlens.detection.statistical import StatisticalDriftDetector
from driftlens.detection.shap_monitor import SHAPMonitor
from driftlens.detection.online import OnlineDriftDetector
from driftlens.agents.graph import run_pipeline
from driftlens.output.schema import DriftSignal
from driftlens.output.sinks import save_alert, save_report, log_to_wandb
from dotenv import load_dotenv
load_dotenv()


SCENARIOS = [
    ("data/drift_scenarios/scenario_00_no_drift.csv", False),
    ("data/drift_scenarios/scenario_01_feature_drift_mild.csv", False),
    ("data/drift_scenarios/scenario_02_feature_drift_severe.csv", False),
    ("data/drift_scenarios/scenario_03_payment_drift.csv", False),
    ("data/drift_scenarios/scenario_04_concept_drift.csv", False),
]


def run_scenario(
    batch_path: str,
    has_labels: bool,
    ingestion: DataIngestion,
    stat_detector: StatisticalDriftDetector,
    shap_monitor: SHAPMonitor,
    online_detector: OnlineDriftDetector,
    model,
) -> None:
    print(f"\n{'='*60}")
    print(f"Scenario: {batch_path}")
    print(f"{'='*60}")

    current, labels = ingestion.load_batch(batch_path, has_target=has_labels)
    reference = ingestion.get_reference()
    feature_names = ingestion.get_feature_names()

    psi_scores, kl_scores = stat_detector.compute(reference, current, feature_names)

    top_psi = sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"Top PSI scores: {[(f, round(v, 4)) for f, v in top_psi]}")

    shap_deltas = shap_monitor.compute_delta(current)
    shap_ref = shap_monitor.get_ref_mean_abs()
    shap_cur = {f: shap_ref[f] + shap_deltas[f] for f in shap_ref}

    adwin_detected = False
    adwin_timestep = None
    if has_labels and labels is not None:
        adwin_detected, adwin_timestep = online_detector.run_on_batch(
            model, current, labels
        )
        print(f"ADWIN drift detected: {adwin_detected} (timestep={adwin_timestep})")

    batch_id = str(uuid.uuid4())[:8]
    signal = DriftSignal(
        batch_id=batch_id,
        psi_scores=psi_scores,
        kl_scores=kl_scores,
        shap_deltas=shap_deltas,
        adwin_drift_detected=adwin_detected,
        adwin_drifted_timestep=adwin_timestep,
        label_available=has_labels,
    )

    print(f"Running agent pipeline (batch_id={batch_id})...")
    report = run_pipeline(
        signal=signal,
        reference=reference,
        current=current,
        shap_ref=shap_ref,
        shap_cur=shap_cur,
    )

    print(f"Severity: {report.alert.severity.value}")
    print(f"Triggered by: {report.alert.triggered_by}")
    print(f"\nExplanation:\n{report.explanation.explanation}")
    print(f"\nRecommendation:\n{report.recommendation.recommendation}")

    save_alert(report.alert)
    save_report(report)
    log_to_wandb(report)
    print(f"Report saved for batch {batch_id}")


def main() -> None:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set")

    cfg = get_config()
    print("Loading components...")

    ingestion = DataIngestion()
    stat_detector = StatisticalDriftDetector()
    online_detector = OnlineDriftDetector()

    model = joblib.load(cfg.model.path)
    shap_monitor = SHAPMonitor(model, ingestion.get_reference())

    print("Components loaded. Running scenarios...")

    for batch_path, has_labels in SCENARIOS:
        if not Path(batch_path).exists():
            print(f"Skipping {batch_path} — file not found")
            continue
        run_scenario(
            batch_path=batch_path,
            has_labels=has_labels,
            ingestion=ingestion,
            stat_detector=stat_detector,
            shap_monitor=shap_monitor,
            online_detector=online_detector,
            model=model,
        )

    print("\nAll scenarios complete.")


if __name__ == "__main__":
    main()