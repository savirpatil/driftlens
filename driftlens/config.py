from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    version: str
    path: str
    type: str


class DataConfig(BaseModel):
    schema_path: str
    reference_path: str
    target_column: str
    batch_size: int


class LabelFreeConfig(BaseModel):
    enabled: bool
    psi_threshold: float
    kl_threshold: float


class LabelAvailableConfig(BaseModel):
    enabled: bool
    adwin_delta: float


class ShapConfig(BaseModel):
    enabled: bool
    background_samples: int
    delta_threshold: float


class DetectionConfig(BaseModel):
    label_free: LabelFreeConfig
    label_available: LabelAvailableConfig
    shap: ShapConfig


class SeverityConfig(BaseModel):
    low_psi_max: float
    med_psi_max: float


class AgentsConfig(BaseModel):
    llm_provider: str
    model: str
    top_k_features: int


class OutputConfig(BaseModel):
    wandb_project: str
    wandb_entity: str
    reports_dir: str
    alerts_dir: str


class ApiConfig(BaseModel):
    host: str
    port: int


class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
    detection: DetectionConfig
    severity: SeverityConfig
    agents: AgentsConfig
    output: OutputConfig
    api: ApiConfig


@lru_cache(maxsize=1)
def get_config(path: str = "driftlens.yaml") -> Config:
    raw = Path(path).read_text()
    data = yaml.safe_load(raw)
    return Config(**data)