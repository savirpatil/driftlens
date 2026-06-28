from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import entropy
from driftlens.config import get_config


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref_counts, bin_edges = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _kl_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.histogram_bin_edges(
        np.concatenate([reference, current]), bins=bins
    )
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    return float(entropy(cur_pct, ref_pct))


class StatisticalDriftDetector:
    def __init__(self) -> None:
        self.config = get_config()

    def compute(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        feature_names: list[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        psi_scores: dict[str, float] = {}
        kl_scores: dict[str, float] = {}

        for feature in feature_names:
            ref_vals = reference[feature].dropna().values
            cur_vals = current[feature].dropna().values
            psi_scores[feature] = _psi(ref_vals, cur_vals)
            kl_scores[feature] = _kl_divergence(ref_vals, cur_vals)

        return psi_scores, kl_scores

    def any_drift(self, psi_scores: dict[str, float]) -> bool:
        threshold = self.config.detection.label_free.psi_threshold
        return any(v > threshold for v in psi_scores.values())