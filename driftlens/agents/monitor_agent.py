from __future__ import annotations
import uuid
from driftlens.config import get_config
from driftlens.output.schema import (
    Alert,
    DriftSignal,
    MonitorPayload,
    Severity,
)


def _compute_severity(max_psi: float, cfg) -> Severity:
    if max_psi < cfg.severity.low_psi_max:
        return Severity.LOW
    elif max_psi < cfg.severity.med_psi_max:
        return Severity.MED
    return Severity.HIGH


def run_monitor_agent(signal: DriftSignal) -> tuple[Alert, MonitorPayload]:
    cfg = get_config()

    ranked = sorted(
        signal.psi_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_k = cfg.agents.top_k_features
    top_features = [f for f, _ in ranked[:top_k]]
    max_psi = ranked[0][1] if ranked else 0.0

    severity = _compute_severity(max_psi, cfg)

    triggered_by: list[str] = []
    psi_threshold = cfg.detection.label_free.psi_threshold
    for feature, psi in ranked:
        if psi > psi_threshold:
            triggered_by.append(f"{feature} (PSI={psi:.3f})")

    if signal.adwin_drift_detected:
        triggered_by.append(
            f"ADWIN detected drift at timestep {signal.adwin_drifted_timestep}"
        )

    alert = Alert(
        batch_id=signal.batch_id,
        timestamp=signal.timestamp,
        severity=severity,
        top_drifted_features=top_features,
        max_psi=max_psi,
        triggered_by=triggered_by,
    )

    payload = MonitorPayload(
        batch_id=signal.batch_id,
        severity=severity,
        ranked_features=ranked,
        psi_scores=signal.psi_scores,
        kl_scores=signal.kl_scores,
        shap_deltas=signal.shap_deltas,
        adwin_drift_detected=signal.adwin_drift_detected,
        label_available=signal.label_available,
    )

    return alert, payload