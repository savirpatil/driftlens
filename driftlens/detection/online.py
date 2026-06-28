from __future__ import annotations
import numpy as np
import pandas as pd
from river.drift import ADWIN
from driftlens.config import get_config


class OnlineDriftDetector:
    def __init__(self) -> None:
        self.config = get_config()
        delta = self.config.detection.label_available.adwin_delta
        self.detector = ADWIN(delta=delta)
        self.drift_detected: bool = False
        self.drift_timestep: int | None = None

    def update(self, error_stream: list[float]) -> tuple[bool, int | None]:
        self.detector = ADWIN(
            delta=self.config.detection.label_available.adwin_delta
        )
        self.drift_detected = False
        self.drift_timestep = None

        for i, error in enumerate(error_stream):
            self.detector.update(error)
            if self.detector.drift_detected:
                self.drift_detected = True
                self.drift_timestep = i
                break

        return self.drift_detected, self.drift_timestep

    def run_on_batch(
        self,
        model,
        current: pd.DataFrame,
        labels: pd.Series,
    ) -> tuple[bool, int | None]:
        preds = model.predict(current)
        errors = (preds != labels.values).astype(float).tolist()
        return self.update(errors)

    def reset(self) -> None:
        self.drift_detected = False
        self.drift_timestep = None
        self.detector = ADWIN(
            delta=self.config.detection.label_available.adwin_delta
        )