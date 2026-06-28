# DriftLens

A domain-agnostic ML model monitoring system that detects distribution drift, explains why it's happening in natural language, and recommends what to do next.

Inspired by the pattern of Microsoft's internal AuRA (Automated Risk Assessment) system. Grounded in three papers: Leest et al. (2024) on expert-driven monitoring, Klaise et al. (ICML 2020) on production model explainability, and Sipple et al. (Springer 2025) on zero-shot drift detection with LLMs.

## What it does

Any sklearn-compatible model registers via `driftlens.yaml`. The system then:

1. Validates incoming data batches against a typed feature schema
2. Runs two-mode drift detection:
   - **Label-free** (always on): PSI + KL divergence per feature via EvidentlyAI
   - **Label-available** (when ground truth exists): ADWIN online detector via river
   - **SHAP layer** (always on): feature attribution delta vs. reference window
3. Emits a structured `Alert` when drift exceeds threshold
4. Fires a LangGraph multi-agent pipeline:
   - **Monitor Agent**: ranks drifted features, computes severity (LOW/MED/HIGH)
   - **Explanation Agent**: calls tools to fetch real distribution stats and SHAP deltas, generates grounded natural language root cause explanation
   - **Recommendation Agent**: branches on severity to produce specific actionable guidance
5. Streams a `DriftReport` to W&B + FastAPI SSE dashboard
6. Scores explanation quality offline via LLM-as-judge (3-dimension rubric, 1-5)

## Tech stack

| Component | Library |
|---|---|
| Model | XGBoost + scikit-learn |
| Drift detection | EvidentlyAI, river (ADWIN), SHAP |
| Agents | LangGraph + LangChain |
| LLM inference | Groq (llama-3.3-70b-versatile) |
| API | FastAPI + SSE |
| Tracking | Weights & Biases |
| Data validation | Pydantic v2 |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/savirpatil/driftlens
cd driftlens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Set environment variables
cp .env.example .env
# Add GROQ_API_KEY and WANDB_API_KEY to .env

# 3. Train the baseline model
python models/train_baseline.py

# 4. Generate drift scenarios
python scripts/inject_drift.py

# 5. Run the full demo pipeline
python scripts/run_demo.py

# 6. Score outputs with LLM-as-judge
python driftlens/eval/run_eval.py

# 7. Run all tests
pytest tests/ -v

# 8. Start the API + dashboard
uvicorn driftlens.api.main:app --reload
# Open http://localhost:8000/dashboard
```

## Swapping to a new model

Edit `driftlens.yaml` only — no code changes required:

```yaml
model:
  name: your_model_name
  path: models/your_model.joblib

data:
  schema_path: schema/your_features.json
  reference_path: data/your_reference.csv
  target_column: your_target
```

## Eval results

| Scenario | Severity | Factual | Causal | Recommendation | Avg |
|---|---|---|---|---|---|
| No drift | LOW | 5 | 5 | 5 | 5.0 |
| Feature drift mild | HIGH | 5 | 5 | 5 | 5.0 |
| Feature drift severe | HIGH | 5 | 5 | 5 | 5.0 |
| Payment drift | HIGH | 5 | 5 | 5 | 5.0 |
| Concept drift | HIGH | 5 | 5 | 5 | 5.0 |

## Project structure
driftlens/

├── driftlens.yaml              # Config — only file to change per domain

├── schema/                     # Feature schema per model

├── data/                       # Reference window + drift scenarios

├── models/                     # Training script + saved model

├── driftlens/

│   ├── config.py               # Typed config loader

│   ├── ingestion.py            # Schema validation + batch loading

│   ├── detection/              # PSI, KL, SHAP, ADWIN

│   ├── agents/                 # LangGraph pipeline

│   ├── output/                 # Pydantic schemas + sinks (W&B, disk, SSE)

│   ├── eval/                   # LLM-as-judge scorer

│   └── api/                    # FastAPI + dashboard

├── scripts/                    # Demo runner + drift injector

└── tests/                      # 16 tests across detection, agents, eval

## References

1. Leest et al. (2024). *Expert-Driven Monitoring of Operational ML Models*. https://arxiv.org/abs/2401.11993
2. Klaise et al. (2020). *Monitoring and Explainability of Models in Production*. ICML Workshop. https://arxiv.org/abs/2007.06299
3. Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS. https://arxiv.org/abs/2306.05685