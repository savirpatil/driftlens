import numpy as np
import pandas as pd
import pytest
from driftlens.config import get_config
from driftlens.ingestion import DataIngestion, SchemaValidationError
from driftlens.detection.statistical import StatisticalDriftDetector
from driftlens.detection.online import OnlineDriftDetector


# ── config ────────────────────────────────────────────────────────────────────
def test_config_loads():
    cfg = get_config()
    assert cfg.model.name == "credit_default_xgb"
    assert cfg.detection.label_free.psi_threshold == 0.2


# ── schema validation ─────────────────────────────────────────────────────────
def test_schema_rejects_missing_column():
    ingestion = DataIngestion()
    bad_df = pd.DataFrame({"LIMIT_BAL": [50000]})  # missing all other cols
    with pytest.raises(SchemaValidationError):
        ingestion.schema.validate(bad_df)


# ── statistical drift ─────────────────────────────────────────────────────────
def test_psi_no_drift():
    detector = StatisticalDriftDetector()
    ingestion = DataIngestion()
    ref = ingestion.get_reference()
    # same data = no drift
    psi, kl = detector.compute(ref, ref.copy(), ingestion.get_feature_names())
    assert all(v < 0.05 for v in psi.values())


def test_psi_detects_drift():
    detector = StatisticalDriftDetector()
    ingestion = DataIngestion()
    ref = ingestion.get_reference()
    drifted = ref.copy()
    drifted["LIMIT_BAL"] = drifted["LIMIT_BAL"] * 5  # inject obvious drift
    psi, _ = detector.compute(ref, drifted, ingestion.get_feature_names())
    assert psi["LIMIT_BAL"] > 0.2


# ── online drift ──────────────────────────────────────────────────────────────
def test_adwin_no_drift():
    detector = OnlineDriftDetector()
    stable_errors = [0.1] * 200
    detected, timestep = detector.update(stable_errors)
    assert detected is False
    assert timestep is None


def test_adwin_detects_drift():
    detector = OnlineDriftDetector()
    error_stream = [0.05] * 100 + [0.9] * 100  # sudden jump in error
    detected, timestep = detector.update(error_stream)
    assert detected is True
    assert timestep is not None