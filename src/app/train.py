from __future__ import annotations

from pathlib import Path

from app.data import load_adult_census_hf
from app.eval import evaluate
from app.io import save_model
from app.model import build_model


def main() -> None:
    split = load_adult_census_hf(test_size=0.2, random_state=42)

    pipeline = build_model(split.numeric_features, split.categorical_features)
    pipeline.fit(split.X_train, split.y_train)

    preds = pipeline.predict(split.X_test)
    metrics = evaluate(split.y_test, preds)

    model_path = save_model(pipeline, Path("models") / "model.joblib")

    print(f"Saved model to: {model_path}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"F1:       {metrics.f1:.4f}")


if __name__ == "__main__":
    main()
