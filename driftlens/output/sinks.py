from __future__ import annotations
import json
import asyncio
from pathlib import Path
from datetime import datetime
import wandb
from driftlens.config import get_config
from driftlens.output.schema import Alert, DriftReport


_sse_queue: asyncio.Queue | None = None


def set_sse_queue(queue: asyncio.Queue) -> None:
    global _sse_queue
    _sse_queue = queue


def _push_sse(event_type: str, data: dict) -> None:
    if _sse_queue is not None:
        try:
            _sse_queue.put_nowait({"event": event_type, "data": data})
        except asyncio.QueueFull:
            pass


def save_alert(alert: Alert) -> Path:
    cfg = get_config()
    alerts_dir = Path(cfg.output.alerts_dir)
    alerts_dir.mkdir(parents=True, exist_ok=True)
    path = alerts_dir / f"alert_{alert.batch_id}.json"
    path.write_text(alert.model_dump_json(indent=2))
    _push_sse("alert", alert.model_dump(mode="json"))
    return path


def save_report(report: DriftReport) -> Path:
    cfg = get_config()
    reports_dir = Path(cfg.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"report_{report.batch_id}.json"
    path.write_text(report.model_dump_json(indent=2))
    _push_sse("report", report.model_dump(mode="json"))
    return path


def log_to_wandb(report: DriftReport) -> None:
    cfg = get_config()

    if wandb.run is None:
        wandb.init(
            project=cfg.output.wandb_project,
            entity=cfg.output.wandb_entity,
            name=f"batch_{report.batch_id}",
            reinit=True,
        )

    psi_metrics = {
        f"psi/{k}": v
        for k, v in report.monitor_payload.psi_scores.items()
    }
    kl_metrics = {
        f"kl/{k}": v
        for k, v in report.monitor_payload.kl_scores.items()
    }
    shap_metrics = {
        f"shap_delta/{k}": v
        for k, v in report.monitor_payload.shap_deltas.items()
    }

    wandb.log({
        "batch_id": report.batch_id,
        "severity": report.alert.severity.value,
        "max_psi": report.alert.max_psi,
        "adwin_drift": report.monitor_payload.adwin_drift_detected,
        **psi_metrics,
        **kl_metrics,
        **shap_metrics,
    })

    if report.judge_scores:
        wandb.log({
            f"judge/{k}": v
            for k, v in report.judge_scores.items()
        })

    report_artifact = wandb.Artifact(
        name=f"drift_report_{report.batch_id}",
        type="drift_report",
    )
    report_path = Path(cfg.output.reports_dir) / f"report_{report.batch_id}.json"
    if report_path.exists():
        report_artifact.add_file(str(report_path))
        wandb.log_artifact(report_artifact)