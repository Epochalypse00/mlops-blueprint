from __future__ import annotations

import hashlib

import pandas as pd

from app.data import load_adult_census_hf


def _hash_df(df: pd.DataFrame) -> str:
    b = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(b).hexdigest()


def test_split_is_reproducible_with_fixed_seed():
    a = load_adult_census_hf(test_size=0.2, random_state=42)
    b = load_adult_census_hf(test_size=0.2, random_state=42)

    assert _hash_df(a.X_train) == _hash_df(b.X_train)
    assert _hash_df(a.X_test) == _hash_df(b.X_test)
