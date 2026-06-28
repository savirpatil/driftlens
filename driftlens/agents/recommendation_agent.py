from __future__ import annotations
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from driftlens.config import get_config
from driftlens.output.schema import (
    ExplanationOutput,
    MonitorPayload,
    RecommendationOutput,
    Severity,
)


LOW_PROMPT = """You are an ML monitoring assistant. Drift has been detected at LOW severity.
Write a single concise recommendation (2-3 sentences) telling the team to:
- Continue monitoring
- Re-evaluate in the next few batches
- Watch specifically the features listed
Do not recommend retraining. Be specific about the features."""

MED_PROMPT = """You are an ML monitoring assistant. Drift has been detected at MEDIUM severity.
Write a single concise recommendation (2-3 sentences) telling the team to:
- Schedule retraining on a recent data window
- Prioritize the specific drifted features listed
- Increase monitoring frequency in the meantime
Be specific about the features and practical about the action."""

HIGH_PROMPT = """You are an ML monitoring assistant. Drift has been detected at HIGH severity.
Write a single concise recommendation (2-3 sentences) telling the team to:
- Trigger immediate retraining
- Escalate to human review
- Consider rolling back the model if retraining is not immediate
Name the specific features that have drifted critically. Be direct and urgent."""


_SEVERITY_PROMPTS: dict[Severity, str] = {
    Severity.LOW: LOW_PROMPT,
    Severity.MED: MED_PROMPT,
    Severity.HIGH: HIGH_PROMPT,
}


def run_recommendation_agent(
    explanation: ExplanationOutput,
    payload: MonitorPayload,
) -> RecommendationOutput:
    cfg = get_config()
    llm = ChatGroq(model=cfg.agents.model, temperature=0)

    system_prompt = _SEVERITY_PROMPTS[payload.severity]
    top_features = [f for f, _ in payload.ranked_features[: cfg.agents.top_k_features]]

    user_message = f"""Batch ID: {payload.batch_id}
Severity: {payload.severity.value}
Top drifted features: {', '.join(top_features)}

Explanation of what happened:
{explanation.explanation}

Provide your recommendation."""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )

    recommendation_text = response.content.strip()

    return RecommendationOutput(
        batch_id=payload.batch_id,
        severity=payload.severity,
        recommendation=recommendation_text,
        features_implicated=top_features,
    )