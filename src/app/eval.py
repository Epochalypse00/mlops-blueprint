from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    f1: float


def evaluate(y_true, y_pred) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
    )
