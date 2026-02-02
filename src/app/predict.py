from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.io import load_model


# Map “dotted” Adult dataset keys -> our training schema keys (underscore)
DOT_TO_UNDERSCORE = {
    "education.num": "education_num",
    "marital.status": "marital_status",
    "capital.gain": "capital_gain",
    "capital.loss": "capital_loss",
    "hours.per.week": "hours_per_week",
    "native.country": "native_country",
}

# The exact feature schema your model expects (underscore version)
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    """
    Accepts:
      - a single JSON object
      - a JSON list of objects
    """
    text = path.read_text(encoding="utf-8-sig")
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    payload = json.loads(text)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        return payload
    raise ValueError("JSON must be an object or a list of objects.")


def _normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    """
    Converts dotted keys to underscore keys.
    Leaves underscore keys as-is.
    """
    out: dict[str, Any] = {}
    for k, v in row.items():
        out[DOT_TO_UNDERSCORE.get(k, k)] = v
    return out


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the model-required columns exist, even if input omitted some.
    - numeric missing -> 0
    - categorical missing -> "Unknown"
    """
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

    # Keep only what the model expects (prevents “extra columns” surprises later)
    return df[REQUIRED_COLUMNS]


def predict(
    model_path: Path, input_path: Path, output_path: Path
) -> list[dict[str, Any]]:
    model = load_model(model_path)

    rows_raw = _read_json_rows(input_path)
    rows = [_normalize_row_keys(r) for r in rows_raw]

    df = pd.DataFrame(rows)
    df = _ensure_required_columns(df)

    preds = model.predict(df)

    # Probability of positive class (>50K)
    prob_pos = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if ">50K" in classes:
            pos_idx = classes.index(">50K")
            prob_pos = model.predict_proba(df)[:, pos_idx].tolist()

    output: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        item: dict[str, Any] = {"input": r, "prediction": str(preds[i])}
        if prob_pos is not None:
            item["probability_positive"] = float(prob_pos[i])
        output.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch predict income class using a saved model."
    )
    parser.add_argument("--model", type=Path, default=Path("models") / "model.joblib")
    parser.add_argument("--input", type=Path, default=Path("sample.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports") / "predictions.json"
    )
    args = parser.parse_args()

    out = predict(model_path=args.model, input_path=args.input, output_path=args.output)
    print(f"Wrote {len(out)} predictions to: {args.output}")


if __name__ == "__main__":
    main()
