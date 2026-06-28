from __future__ import annotations
import json
import re
from pathlib import Path
from groq import Groq
from driftlens.config import get_config
from driftlens.output.schema import DriftReport, JudgeScore


FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — Strong explanation+recommendation (score ~5):
Explanation: "LIMIT_BAL mean increased from 166k to 592k (PSI=1.47), and its mean absolute SHAP value rose by 34.78%, indicating the model is now placing significantly more weight on credit limit when scoring applicants. AGE also drifted upward (mean +18.9 years, PSI=8.72), though its SHAP delta was smaller (+62.54%), suggesting the model is sensitive to this shift. Together these changes indicate the incoming population skews older and wealthier, which may cause the model to underestimate default risk for this segment."
Recommendation: "Immediate retraining is required. Prioritize LIMIT_BAL and AGE in the retraining window. Escalate to human review before deploying retrained model. Consider rolling back if retraining cannot begin within 24 hours."
Scores: factual_grounding=5, causal_coherence=5, recommendation_appropriateness=5

EXAMPLE 2 — Weak explanation+recommendation (score ~1-2):
Explanation: "Some features have drifted. The model may not perform well."
Recommendation: "Please retrain the model."
Scores: factual_grounding=1, causal_coherence=1, recommendation_appropriateness=2

EXAMPLE 3 — Strong partial (score ~3-4):
Explanation: "PAY_0, PAY_2, and PAY_3 have drifted significantly. Their means increased. SHAP values also changed."
Recommendation: "Retrain the model soon focusing on payment features PAY_0, PAY_2, PAY_3."
Scores: factual_grounding=3, causal_coherence=2, recommendation_appropriateness=4

EXAMPLE 4 — Wrong severity action (score ~1-2):
Explanation: "LIMIT_BAL PSI=1.47, AGE PSI=8.72, BILL_AMT1 PSI=0.60. All three increased in mean. SHAP values rose for all three features indicating higher model reliance."
Recommendation: "Continue monitoring for a few more batches."
Scores: factual_grounding=4, causal_coherence=3, recommendation_appropriateness=1
"""

JUDGE_SYSTEM_PROMPT = f"""You are an expert ML monitoring evaluator. You score drift report explanations and recommendations on three dimensions, each 1-5.

RUBRIC:
Factual Grounding (1-5): Did the explanation correctly identify which features drifted and directionally how?
  1=wrong features or no features cited
  3=correct features but missing magnitude or direction
  5=correct features, correct direction, correct magnitude referenced

Causal Coherence (1-5): Does the explanation logically connect feature shift → SHAP change → performance impact?
  1=no causal chain, just lists numbers
  3=partial chain, missing one link
  5=complete chain clearly stated

Recommendation Appropriateness (1-5): Is the action proportionate to severity and specific to the implicated features?
  1=generic or wrong severity action
  3=correct severity but not feature-specific
  5=correct severity, names specific features, actionable

{FEW_SHOT_EXAMPLES}

Respond ONLY with valid JSON in this exact format:
{{"factual_grounding": <int>, "causal_coherence": <int>, "recommendation_appropriateness": <int>, "rationale": "<one sentence>"}}"""


def score_report(report: DriftReport) -> JudgeScore:
    cfg = get_config()
    client = Groq()

    user_content = f"""Severity: {report.alert.severity.value}
Top drifted features: {report.alert.top_drifted_features}
Max PSI: {report.alert.max_psi:.4f}

Explanation:
{report.explanation.explanation}

Recommendation:
{report.recommendation.recommendation}

Score this explanation and recommendation."""

    response = client.chat.completions.create(
        model=cfg.agents.model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    clean = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(clean)

    avg = round(
        (data["factual_grounding"] + data["causal_coherence"] + data["recommendation_appropriateness"]) / 3,
        2,
    )

    return JudgeScore(
        batch_id=report.batch_id,
        factual_grounding=data["factual_grounding"],
        causal_coherence=data["causal_coherence"],
        recommendation_appropriateness=data["recommendation_appropriateness"],
        average=avg,
        rationale=data["rationale"],
    )


def load_report(path: Path) -> DriftReport:
    return DriftReport(**json.loads(path.read_text()))