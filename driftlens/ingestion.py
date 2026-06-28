from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from driftlens.config import get_config


class SchemaValidationError(Exception):
    pass


class FeatureSchema:
    def __init__(self, schema_path: str) -> None:
        raw = json.loads(Path(schema_path).read_text())
        self.features: list[dict] = raw["features"]
        self.target: dict = raw["target"]
        self.feature_names: list[str] = [f["name"] for f in self.features]

    def validate(self, df: pd.DataFrame, has_target: bool = False) -> None:
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise SchemaValidationError(f"Missing features: {missing}")

        for feat in self.features:
            name = feat["name"]
            col = df[name]

            if feat["dtype"] == "float":
                if not pd.api.types.is_numeric_dtype(col):
                    raise SchemaValidationError(f"{name} must be numeric")
                if "min" in feat and col.min() < feat["min"]:
                    raise SchemaValidationError(f"{name} below min {feat['min']}")
                if "max" in feat and col.max() > feat["max"]:
                    raise SchemaValidationError(f"{name} above max {feat['max']}")

            elif feat["dtype"] == "int":
                if not pd.api.types.is_numeric_dtype(col):
                    raise SchemaValidationError(f"{name} must be numeric")
                if "allowed_values" in feat:
                    invalid = set(col.unique()) - set(feat["allowed_values"])
                    if invalid:
                        raise SchemaValidationError(
                            f"{name} has invalid values: {invalid}"
                        )
                if "min" in feat and col.min() < feat["min"]:
                    raise SchemaValidationError(f"{name} below min {feat['min']}")
                if "max" in feat and col.max() > feat["max"]:
                    raise SchemaValidationError(f"{name} above max {feat['max']}")

        if has_target:
            target_name = self.target["name"]
            if target_name not in df.columns:
                raise SchemaValidationError(f"Target column {target_name} missing")


class DataIngestion:
    def __init__(self) -> None:
        self.config = get_config()
        self.schema = FeatureSchema(self.config.data.schema_path)
        self.reference: pd.DataFrame = self._load_reference()

    def _load_reference(self) -> pd.DataFrame:
        ref = pd.read_csv(self.config.data.reference_path)
        self.schema.validate(ref, has_target=False)
        return ref[self.schema.feature_names]

    def load_batch(
        self, path: str, has_target: bool = False
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        df = pd.read_csv(path)
        self.schema.validate(df, has_target=has_target)
        target: pd.Series | None = None
        if has_target:
            target = df[self.config.data.target_column]
        return df[self.schema.feature_names], target

    def get_reference(self) -> pd.DataFrame:
        return self.reference

    def get_feature_names(self) -> list[str]:
        return self.schema.feature_names