from __future__ import annotations
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from driftlens.output.schema import FeatureProfile, SHAPComparison


_reference: pd.DataFrame | None = None
_current: pd.DataFrame | None = None
_psi_scores: dict[str, float] = {}
_shap_ref: dict[str, float] = {}
_shap_cur: dict[str, float] = {}


def register_context(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_scores: dict[str, float],
    shap_ref: dict[str, float],
    shap_cur: dict[str, float],
) -> None:
    global _reference, _current, _psi_scores, _shap_ref, _shap_cur
    _reference = reference
    _current = current
    _psi_scores = psi_scores
    _shap_ref = shap_ref
    _shap_cur = shap_cur


@tool
def profile_feature(feature_name: str) -> dict:
    """Return current vs reference distribution stats for a feature."""
    if _reference is None or _current is None:
        return {"error": "Context not registered"}
    if feature_name not in _reference.columns:
        return {"error": f"Feature {feature_name} not found"}

    ref_col = _reference[feature_name].dropna()
    cur_col = _current[feature_name].dropna()

    ref_mean = float(ref_col.mean())
    cur_mean = float(cur_col.mean())
    direction = "increased" if cur_mean > ref_mean else "decreased"

    profile = FeatureProfile(
        feature_name=feature_name,
        ref_mean=ref_mean,
        cur_mean=cur_mean,
        ref_std=float(ref_col.std()),
        cur_std=float(cur_col.std()),
        ref_min=float(ref_col.min()),
        cur_min=float(cur_col.min()),
        ref_max=float(ref_col.max()),
        cur_max=float(cur_col.max()),
        psi=_psi_scores.get(feature_name, 0.0),
        direction=direction,
    )
    return profile.model_dump()


@tool
def get_shap_comparison(feature_name: str) -> dict:
    """Return mean absolute SHAP value current vs reference for a feature."""
    if feature_name not in _shap_ref:
        return {"error": f"Feature {feature_name} not in SHAP reference"}

    ref_val = _shap_ref.get(feature_name, 0.0)
    cur_val = _shap_cur.get(feature_name, 0.0)
    delta = cur_val - ref_val
    delta_pct = (delta / ref_val * 100) if ref_val != 0 else 0.0

    comparison = SHAPComparison(
        feature_name=feature_name,
        ref_mean_abs_shap=ref_val,
        cur_mean_abs_shap=cur_val,
        delta=delta,
        delta_pct=delta_pct,
    )
    return comparison.model_dump()