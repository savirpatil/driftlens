from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class DriftSignal(BaseModel):
    batch_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    psi_scores: dict[str, float]
    kl_scores: dict[str, float]
    shap_deltas: dict[str, float]
    adwin_drift_detected: bool = False
    adwin_drifted_timestep: int | None = None
    label_available: bool = False


class Alert(BaseModel):
    batch_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: Severity
    top_drifted_features: list[str]
    max_psi: float
    triggered_by: list[str]


class MonitorPayload(BaseModel):
    batch_id: str
    severity: Severity
    ranked_features: list[tuple[str, float]]
    psi_scores: dict[str, float]
    kl_scores: dict[str, float]
    shap_deltas: dict[str, float]
    adwin_drift_detected: bool
    label_available: bool


class FeatureProfile(BaseModel):
    feature_name: str
    ref_mean: float
    cur_mean: float
    ref_std: float
    cur_std: float
    ref_min: float
    cur_min: float
    ref_max: float
    cur_max: float
    psi: float
    direction: str


class SHAPComparison(BaseModel):
    feature_name: str
    ref_mean_abs_shap: float
    cur_mean_abs_shap: float
    delta: float
    delta_pct: float


class ExplanationOutput(BaseModel):
    batch_id: str
    explanation: str
    features_cited: list[str]
    feature_profiles: list[FeatureProfile]
    shap_comparisons: list[SHAPComparison]


class RecommendationOutput(BaseModel):
    batch_id: str
    severity: Severity
    recommendation: str
    features_implicated: list[str]


class DriftReport(BaseModel):
    batch_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    alert: Alert
    monitor_payload: MonitorPayload
    explanation: ExplanationOutput
    recommendation: RecommendationOutput
    judge_scores: dict[str, Any] | None = None


class JudgeScore(BaseModel):
    batch_id: str
    factual_grounding: int = Field(ge=1, le=5)
    causal_coherence: int = Field(ge=1, le=5)
    recommendation_appropriateness: int = Field(ge=1, le=5)
    average: float
    rationale: str