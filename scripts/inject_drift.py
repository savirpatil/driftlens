from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COL = "default_payment_next_month"


def load_test() -> pd.DataFrame:
    return pd.read_csv("data/test_set.csv")


def save_scenario(df: pd.DataFrame, name: str) -> None:
    out_dir = Path("data/drift_scenarios")
    out_dir.mkdir(parents=True, exist_ok=True)
    features_only = df[FEATURE_COLS].copy()
    path = out_dir / f"{name}.csv"
    features_only.to_csv(path, index=False)
    print(f"Saved {path} with shape {features_only.shape}")


def scenario_no_drift(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(n=500, random_state=42).reset_index(drop=True)


def scenario_feature_drift_mild(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.sample(n=500, random_state=1).reset_index(drop=True)
    drifted["LIMIT_BAL"] = drifted["LIMIT_BAL"] * np.random.uniform(1.5, 2.0, size=len(drifted))
    drifted["AGE"] = (drifted["AGE"] + np.random.randint(5, 10, size=len(drifted))).clip(upper=100)
    return drifted


def scenario_feature_drift_severe(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.sample(n=500, random_state=2).reset_index(drop=True)
    drifted["LIMIT_BAL"] = drifted["LIMIT_BAL"] * np.random.uniform(4.0, 6.0, size=len(drifted))
    drifted["BILL_AMT1"] = drifted["BILL_AMT1"] * np.random.uniform(3.0, 5.0, size=len(drifted))
    drifted["PAY_AMT1"] = drifted["PAY_AMT1"] * np.random.uniform(3.0, 5.0, size=len(drifted))
    drifted["AGE"] = (drifted["AGE"] + np.random.randint(15, 25, size=len(drifted))).clip(upper=100)
    return drifted


def scenario_payment_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.sample(n=500, random_state=3).reset_index(drop=True)
    drifted["PAY_0"] = drifted["PAY_0"].clip(lower=0) + np.random.randint(1, 4, size=len(drifted))
    drifted["PAY_2"] = drifted["PAY_2"].clip(lower=0) + np.random.randint(1, 3, size=len(drifted))
    drifted["PAY_3"] = drifted["PAY_3"].clip(lower=0) + np.random.randint(1, 3, size=len(drifted))
    return drifted


def scenario_concept_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.sample(n=500, random_state=4).reset_index(drop=True)
    high_limit = drifted["LIMIT_BAL"] > drifted["LIMIT_BAL"].median()
    drifted.loc[high_limit, "LIMIT_BAL"] = (
        drifted.loc[high_limit, "LIMIT_BAL"] * np.random.uniform(0.3, 0.6, size=high_limit.sum())
    ).astype(float)
    drifted["PAY_0"] = np.random.choice([-1, 0, 1, 2], size=len(drifted), p=[0.1, 0.2, 0.3, 0.4])
    return drifted


if __name__ == "__main__":
    df = load_test()
    print(f"Test set loaded: {df.shape}")

    save_scenario(scenario_no_drift(df), "scenario_00_no_drift")
    save_scenario(scenario_feature_drift_mild(df), "scenario_01_feature_drift_mild")
    save_scenario(scenario_feature_drift_severe(df), "scenario_02_feature_drift_severe")
    save_scenario(scenario_payment_drift(df), "scenario_03_payment_drift")
    save_scenario(scenario_concept_drift(df), "scenario_04_concept_drift")

    print("All scenarios generated.")