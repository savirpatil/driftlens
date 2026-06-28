from __future__ import annotations
import asyncio
import json
import uuid
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from driftlens.config import get_config
from driftlens.ingestion import DataIngestion
from driftlens.detection.statistical import StatisticalDriftDetector
from driftlens.detection.shap_monitor import SHAPMonitor
from driftlens.detection.online import OnlineDriftDetector
from driftlens.agents.graph import run_pipeline
from driftlens.output.schema import DriftSignal, DriftReport
from driftlens.output.sinks import save_alert, save_report, log_to_wandb, set_sse_queue
from dotenv import load_dotenv
load_dotenv()

import joblib

app = FastAPI(title="DriftLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sse_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
set_sse_queue(_sse_queue)

_ingestion: DataIngestion | None = None
_stat_detector: StatisticalDriftDetector | None = None
_shap_monitor: SHAPMonitor | None = None
_online_detector: OnlineDriftDetector | None = None
_model = None


def _get_components():
    global _ingestion, _stat_detector, _shap_monitor, _online_detector, _model
    if _ingestion is None:
        cfg = get_config()
        _ingestion = DataIngestion()
        _stat_detector = StatisticalDriftDetector()
        _online_detector = OnlineDriftDetector()
        model_path = Path(cfg.model.path)
        if model_path.exists():
            _model = joblib.load(model_path)
            _shap_monitor = SHAPMonitor(_model, _ingestion.get_reference())
    return _ingestion, _stat_detector, _shap_monitor, _online_detector, _model


class RunBatchRequest(BaseModel):
    batch_path: str
    has_labels: bool = False


class RunBatchResponse(BaseModel):
    batch_id: str
    severity: str
    max_psi: float
    triggered_by: list[str]
    explanation: str
    recommendation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html_path = Path(__file__).parent / "dashboard" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/reports")
def list_reports():
    cfg = get_config()
    reports_dir = Path(cfg.output.reports_dir)
    if not reports_dir.exists():
        return []
    paths = sorted(reports_dir.glob("report_*.json"), reverse=True)
    reports = []
    for p in paths[:20]:
        try:
            reports.append(json.loads(p.read_text()))
        except Exception:
            continue
    return reports


@app.get("/reports/{batch_id}")
def get_report(batch_id: str):
    cfg = get_config()
    path = Path(cfg.output.reports_dir) / f"report_{batch_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return json.loads(path.read_text())


@app.post("/run-batch", response_model=RunBatchResponse)
def run_batch(request: RunBatchRequest):
    ingestion, stat_detector, shap_monitor, online_detector, model = _get_components()

    current, labels = ingestion.load_batch(
        request.batch_path, has_target=request.has_labels
    )
    reference = ingestion.get_reference()
    feature_names = ingestion.get_feature_names()

    psi_scores, kl_scores = stat_detector.compute(reference, current, feature_names)

    shap_deltas: dict[str, float] = {}
    shap_ref: dict[str, float] = {}
    shap_cur: dict[str, float] = {}

    if shap_monitor is not None:
        shap_deltas = shap_monitor.compute_delta(current)
        shap_ref = shap_monitor.get_ref_mean_abs()
        shap_cur = {
            f: shap_ref[f] + shap_deltas[f] for f in shap_ref
        }

    adwin_detected = False
    adwin_timestep = None
    if request.has_labels and labels is not None and model is not None:
        adwin_detected, adwin_timestep = online_detector.run_on_batch(
            model, current, labels
        )

    batch_id = str(uuid.uuid4())[:8]
    signal = DriftSignal(
        batch_id=batch_id,
        psi_scores=psi_scores,
        kl_scores=kl_scores,
        shap_deltas=shap_deltas,
        adwin_drift_detected=adwin_detected,
        adwin_drifted_timestep=adwin_timestep,
        label_available=request.has_labels,
    )

    report: DriftReport = run_pipeline(
        signal=signal,
        reference=reference,
        current=current,
        shap_ref=shap_ref,
        shap_cur=shap_cur,
    )

    save_alert(report.alert)
    save_report(report)
    log_to_wandb(report)

    return RunBatchResponse(
        batch_id=report.batch_id,
        severity=report.alert.severity.value,
        max_psi=report.alert.max_psi,
        triggered_by=report.alert.triggered_by,
        explanation=report.explanation.explanation,
        recommendation=report.recommendation.recommendation,
    )


@app.get("/stream")
async def stream(request_obj: None = None):
    async def event_generator() -> AsyncGenerator[dict, None]:
        while True:
            try:
                event = await asyncio.wait_for(_sse_queue.get(), timeout=30.0)
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"]),
                }
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}

    return EventSourceResponse(event_generator())