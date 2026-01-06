from __future__ import annotations
import numpy as np

def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=-1)
    return float((preds == labels).mean())
