from __future__ import annotations
import numpy as np

def fedavg(weights: list[np.ndarray], client_sizes: list[int]) -> np.ndarray:
    """
    Weighted average of client parameter vectors.
    weights: list of 1D arrays of same shape
    client_sizes: number of samples per client (weights for averaging)
    """
    if len(weights) == 0:
        raise ValueError("weights is empty")
    if len(weights) != len(client_sizes):
        raise ValueError("weights and client_sizes must have same length")

    total = sum(client_sizes)
    if total <= 0:
        raise ValueError("total client size must be > 0")

    w0_shape = weights[0].shape
    if any(w.shape != w0_shape for w in weights):
        raise ValueError("all weight arrays must have the same shape")

    acc = np.zeros_like(weights[0], dtype=np.float64)
    for w, n in zip(weights, client_sizes):
        acc += (n / total) * w.astype(np.float64)

    return acc.astype(weights[0].dtype)

