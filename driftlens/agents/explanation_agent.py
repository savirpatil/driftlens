from __future__ import annotations
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from driftlens.config import get_config
from driftlens.agents.tools import profile_feature, get_shap_comparison
from driftlens.output.schema import (
    ExplanationOutput,
    FeatureProfile,
    SHAPComparison,
    MonitorPayload,
)


SYSTEM_PROMPT = """You are a machine learning monitoring expert. You have access to two tools:
- profile_feature(feature_name): returns current vs reference distribution statistics for a feature
- get_shap_comparison(feature_name): returns how much the feature's influence on model predictions has changed

Your job:
1. Call profile_feature and get_shap_comparison for each of the top drifted features provided
2. Write a concise explanation (3-5 sentences) that connects:
   - Which features drifted and in what direction (increased/decreased)
   - How their SHAP values changed (more/less influential)
   - What this likely means for model performance
3. Be specific — reference actual numbers from the tool outputs
4. Do not speculate beyond what the data shows

Return ONLY the explanation paragraph. No preamble, no bullet points."""


def run_explanation_agent(payload: MonitorPayload) -> ExplanationOutput:
    cfg = get_config()
    llm = ChatGroq(model=cfg.agents.model, temperature=0)
    tools = [profile_feature, get_shap_comparison]
    agent = create_react_agent(llm, tools)

    top_features = [f for f, _ in payload.ranked_features[: cfg.agents.top_k_features]]

    user_message = f"""The following features have drifted in batch {payload.batch_id}:
{', '.join(top_features)}

Severity: {payload.severity.value}

Please call profile_feature and get_shap_comparison for each feature listed, 
then write a grounded explanation of what is happening and why it matters."""

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        }
    )

    explanation_text = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            explanation_text = msg.content.strip()
            break

    feature_profiles: list[FeatureProfile] = []
    shap_comparisons: list[SHAPComparison] = []

    for feature in top_features:
        profile_result = profile_feature.invoke({"feature_name": feature})
        if "error" not in profile_result:
            feature_profiles.append(FeatureProfile(**profile_result))

        shap_result = get_shap_comparison.invoke({"feature_name": feature})
        if "error" not in shap_result:
            shap_comparisons.append(SHAPComparison(**shap_result))

    return ExplanationOutput(
        batch_id=payload.batch_id,
        explanation=explanation_text,
        features_cited=top_features,
        feature_profiles=feature_profiles,
        shap_comparisons=shap_comparisons,
    )