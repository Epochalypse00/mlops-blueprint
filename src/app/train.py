from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.data import load_adult_census_hf
from app.eval import evaluate
from app.io import save_model
from app.model import build_model


RESULTS_FILE = Path("reports/results.md")


def log_results(metrics, model_path: Path) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat(timespec="seconds")

    line = (
        f"| {timestamp} | adult-census-income | LogisticRegression | "
        f"{metrics.accuracy:.4f} | {metrics.f1:.4f} | {model_path} |\n"
    )

    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "| timestamp | dataset | model | accuracy | f1 | model_path |\n"
            "|-----------|---------|-------|----------|----|------------|\n"
        )

    with RESULTS_FILE.open("a") as f:
        f.write(line)


def main() -> None:
    split = load_adult_census_hf(test_size=0.2, random_state=42)

    pipeline = build_model(split.numeric_features, split.categorical_features)
    pipeline.fit(split.X_train, split.y_train)

    preds = pipeline.predict(split.X_test)
    metrics = evaluate(split.y_test, preds)

    model_path = save_model(pipeline, Path("models") / "model.joblib")

    log_results(metrics, model_path)

    print(f"Saved model to: {model_path}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"F1:       {metrics.f1:.4f}")


if __name__ == "__main__":
    main()
