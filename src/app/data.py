from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    target_name: str = "income"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Convert HF adult dataset dot-style column names to snake_case
    rename_map = {
        "education.num": "education_num",
        "marital.status": "marital_status",
        "capital.gain": "capital_gain",
        "capital.loss": "capital_loss",
        "hours.per.week": "hours_per_week",
        "native.country": "native_country",
    }
    return df.rename(columns=rename_map)


def load_adult_census_hf(
    test_size: float = 0.2, random_state: int = 42
) -> DatasetSplit:
    ds = load_dataset("scikit-learn/adult-census-income")
    df = ds["train"].to_pandas()
    df = _normalize_columns(df)

    target = "income"
    if target not in df.columns:
        raise KeyError(
            f"Expected target column '{target}' not found. Columns: {list(df.columns)}"
        )

    numeric_features = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]

    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    missing = [
        c
        for c in numeric_features + categorical_features + [target]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(
            f"Missing expected columns: {missing}. Available: {list(df.columns)}"
        )

    X = df[numeric_features + categorical_features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_name=target,
    )
