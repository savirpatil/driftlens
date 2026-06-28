from __future__ import annotations
import json
import os
from pathlib import Path
import wandb
from dotenv import load_dotenv
from driftlens.config import get_config
from driftlens.eval.judge import load_report, score_report
from driftlens.output.schema import JudgeScore

load_dotenv()


def find_report_for_scenario(reports_dir: Path, scenario: str) -> Path | None:
    candidates = sorted(reports_dir.glob("report_*.json"), reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text())
            triggered = data.get("alert", {}).get("triggered_by", [])
            severity = data.get("alert", {}).get("severity", "")
            explanation = data.get("explanation", {}).get("explanation", "")
            if scenario == "scenario_00_no_drift" and severity == "LOW":
                return path
            if scenario == "scenario_01_feature_drift_mild" and "AGE" in explanation and severity == "HIGH":
                return path
            if scenario == "scenario_02_feature_drift_severe" and "AGE" in explanation and "BILL_AMT1" in explanation:
                return path
            if scenario == "scenario_03_payment_drift" and "PAY_3" in str(triggered):
                return path
            if scenario == "scenario_04_concept_drift" and "PAY_0" in str(triggered) and "LIMIT_BAL" in str(triggered):
                return path
        except Exception:
            continue
    return None


def run_eval() -> None:
    cfg = get_config()
    reports_dir = Path(cfg.output.reports_dir)

    test_cases_path = Path("driftlens/eval/test_cases.json")
    test_cases = json.loads(test_cases_path.read_text())["test_cases"]

    wandb.init(
        project=cfg.output.wandb_project,
        entity=cfg.output.wandb_entity,
        name="eval_run",
        reinit=True,
    )

    results: list[dict] = []
    all_scores: list[float] = []

    print(f"\n{'='*70}")
    print(f"{'SCENARIO':<40} {'SEV':<6} {'FG':<4} {'CC':<4} {'RA':<4} {'AVG':<5}")
    print(f"{'='*70}")

    for tc in test_cases:
        scenario = tc["scenario"]
        expected_severity = tc["expected_severity"]
        expected_min = tc["expected_min_score"]

        report_path = find_report_for_scenario(reports_dir, scenario)
        if report_path is None:
            print(f"{scenario:<40} REPORT NOT FOUND — skipping")
            continue

        report = load_report(report_path)
        score: JudgeScore = score_report(report)

        severity_match = report.alert.severity.value == expected_severity
        score_pass = score.average >= expected_min

        status = "✓" if (severity_match and score_pass) else "✗"

        print(
            f"{status} {scenario:<38} {report.alert.severity.value:<6} "
            f"{score.factual_grounding:<4} {score.causal_coherence:<4} "
            f"{score.recommendation_appropriateness:<4} {score.average:<5}"
        )
        print(f"  Rationale: {score.rationale}")

        wandb.log({
            f"eval/{scenario}/factual_grounding": score.factual_grounding,
            f"eval/{scenario}/causal_coherence": score.causal_coherence,
            f"eval/{scenario}/recommendation_appropriateness": score.recommendation_appropriateness,
            f"eval/{scenario}/average": score.average,
            f"eval/{scenario}/severity_match": int(severity_match),
            f"eval/{scenario}/score_pass": int(score_pass),
        })

        all_scores.append(score.average)
        results.append({
            "scenario": scenario,
            "batch_id": report.batch_id,
            "severity": report.alert.severity.value,
            "expected_severity": expected_severity,
            "severity_match": severity_match,
            "factual_grounding": score.factual_grounding,
            "causal_coherence": score.causal_coherence,
            "recommendation_appropriateness": score.recommendation_appropriateness,
            "average": score.average,
            "expected_min": expected_min,
            "score_pass": score_pass,
            "rationale": score.rationale,
        })

    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
    passed = sum(1 for r in results if r["severity_match"] and r["score_pass"])

    print(f"{'='*70}")
    print(f"Overall average score: {overall_avg} | Passed: {passed}/{len(results)}")

    wandb.log({
        "eval/overall_average": overall_avg,
        "eval/passed": passed,
        "eval/total": len(results),
    })

    out_path = Path("outputs/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    run_eval()