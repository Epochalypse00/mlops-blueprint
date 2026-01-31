from __future__ import annotations

import numpy as np

from app.data import load_adult_census_hf
from app.model import build_model


def test_preprocess_produces_features():
    split = load_adult_census_hf(test_size=0.2, random_state=42)

    assert len(split.X_train) > 0
    assert len(split.X_test) > 0
    assert len(split.X_train) > len(split.X_test)

    pipeline = build_model(split.numeric_features, split.categorical_features)
    pipeline.fit(split.X_train, split.y_train)

    X_trans = pipeline.named_steps["preprocess"].transform(split.X_test)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    assert X_trans.shape[0] == len(split.X_test)
    assert X_trans.shape[1] > 0
    assert np.isfinite(X_trans).all()
