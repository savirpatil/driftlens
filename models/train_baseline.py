from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COL = "default_payment_next_month"


def load_data() -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    df = pd.read_excel(url, header=1)
    df = df.rename(columns={"default payment next month": TARGET_COL})
    return df


def train(df: pd.DataFrame) -> tuple:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test


def save_artifacts(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    joblib.dump(model, "models/credit_default.joblib")
    print("Model saved to models/credit_default.joblib")

    ref = X_train.iloc[:5000].copy()
    ref.to_csv("data/reference_window.csv", index=False)
    print("Reference window saved to data/reference_window.csv")

    test_data = X_test.copy()
    test_data[TARGET_COL] = y_test.values
    test_data.to_csv("data/test_set.csv", index=False)
    print("Test set saved to data/test_set.csv")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    print("Training model...")
    model, X_train, X_test, y_train, y_test = train(df)

    print("Saving artifacts...")
    save_artifacts(model, X_train, X_test, y_train, y_test)
    print("Done.")