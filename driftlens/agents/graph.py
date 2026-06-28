from __future__ import annotations
from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph, END
from driftlens.output.schema import (
    Alert,
    DriftSignal,
    DriftReport,
    ExplanationOutput,
    MonitorPayload,
    RecommendationOutput,
)
from driftlens.agents.monitor_agent import run_monitor_agent
from driftlens.agents.explanation_agent import run_explanation_agent
from driftlens.agents.recommendation_agent import run_recommendation_agent
from driftlens.agents.tools import register_context


class GraphState(TypedDict):
    signal: DriftSignal
    reference: pd.DataFrame
    current: pd.DataFrame
    shap_ref: dict[str, float]
    shap_cur: dict[str, float]
    alert: Alert | None
    payload: MonitorPayload | None
    explanation: ExplanationOutput | None
    recommendation: RecommendationOutput | None


def monitor_node(state: GraphState) -> GraphState:
    alert, payload = run_monitor_agent(state["signal"])
    return {**state, "alert": alert, "payload": payload}


def explanation_node(state: GraphState) -> GraphState:
    register_context(
        reference=state["reference"],
        current=state["current"],
        psi_scores=state["signal"].psi_scores,
        shap_ref=state["shap_ref"],
        shap_cur=state["shap_cur"],
    )
    explanation = run_explanation_agent(state["payload"])
    return {**state, "explanation": explanation}


def recommendation_node(state: GraphState) -> GraphState:
    recommendation = run_recommendation_agent(
        state["explanation"], state["payload"]
    )
    return {**state, "recommendation": recommendation}


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("monitor_node", monitor_node)
    graph.add_node("explanation_node", explanation_node)
    graph.add_node("recommendation_node", recommendation_node)

    graph.set_entry_point("monitor_node")
    graph.add_edge("monitor_node", "explanation_node")
    graph.add_edge("explanation_node", "recommendation_node")
    graph.add_edge("recommendation_node", END)

    return graph.compile()


def run_pipeline(
    signal: DriftSignal,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    shap_ref: dict[str, float],
    shap_cur: dict[str, float],
) -> DriftReport:
    graph = build_graph()

    initial_state: GraphState = {
        "signal": signal,
        "reference": reference,
        "current": current,
        "shap_ref": shap_ref,
        "shap_cur": shap_cur,
        "alert": None,
        "payload": None,
        "explanation": None,
        "recommendation": None,
    }

    final_state = graph.invoke(initial_state)

    return DriftReport(
        batch_id=signal.batch_id,
        timestamp=signal.timestamp,
        alert=final_state["alert"],
        monitor_payload=final_state["payload"],
        explanation=final_state["explanation"],
        recommendation=final_state["recommendation"],
    )