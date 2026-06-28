from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from driftlens.config import get_config


class SHAPMonitor:
    def __init__(self, model, reference: pd.DataFrame) -> None:
        self.config = get_config()
        self.model = model
        n = min(
            self.config.detection.shap.background_samples, len(reference)
        )
        background = reference.sample(n=n, random_state=42)
        self.explainer = shap.TreeExplainer(model, background)
        self.ref_mean_abs: dict[str, float] = self._compute_mean_abs(reference)

    def _compute_mean_abs(self, df: pd.DataFrame) -> dict[str, float]:
        shap_values = self.explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        return {col: float(mean_abs[i]) for i, col in enumerate(df.columns)}

    def compute_delta(
        self, current: pd.DataFrame
    ) -> dict[str, float]:
        cur_mean_abs = self._compute_mean_abs(current)
        deltas: dict[str, float] = {}
        for feature in self.ref_mean_abs:
            ref_val = self.ref_mean_abs[feature]
            cur_val = cur_mean_abs.get(feature, 0.0)
            deltas[feature] = float(cur_val - ref_val)
        return deltas

    def get_ref_mean_abs(self) -> dict[str, float]:
        return self.ref_mean_abs

    def get_shap_comparison(
        self, feature: str, current: pd.DataFrame
    ) -> tuple[float, float, float, float]:
        cur_mean_abs = self._compute_mean_abs(current)
        ref_val = self.ref_mean_abs.get(feature, 0.0)
        cur_val = cur_mean_abs.get(feature, 0.0)
        delta = cur_val - ref_val
        delta_pct = (delta / ref_val * 100) if ref_val != 0 else 0.0
        return ref_val, cur_val, delta, delta_pct