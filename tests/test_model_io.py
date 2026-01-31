from __future__ import annotations

from pathlib import Path

from app.data import load_adult_census_hf
from app.io import load_model, save_model
from app.model import build_model


def test_save_load_roundtrip(tmp_path: Path):
    split = load_adult_census_hf(test_size=0.2, random_state=42)

    pipeline = build_model(split.numeric_features, split.categorical_features)
    pipeline.fit(split.X_train, split.y_train)

    preds_before = pipeline.predict(split.X_test)

    model_path = tmp_path / "model.joblib"
    save_model(pipeline, model_path)
    loaded = load_model(model_path)

    preds_after = loaded.predict(split.X_test)

    assert (preds_before == preds_after).all()
