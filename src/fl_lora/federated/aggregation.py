from __future__ import annotations
from typing import Dict, List
import torch

def weighted_average_state(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    if len(states) == 0:
        raise ValueError("No states to aggregate")
    if len(states) != len(weights):
        raise ValueError("states and weights length mismatch")
    total = sum(weights)
    if total <= 0:
        raise ValueError("total weight must be > 0")

    keys = states[0].keys()
    out = {}
    for k in keys:
        acc = None
        for sd, w in zip(states, weights):
            t = sd[k].to(torch.float64)
            acc = t * (w / total) if acc is None else acc + t * (w / total)
        out[k] = acc.to(states[0][k].dtype)
    return out

def apply_state_to_model(model, trainable_state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and name in trainable_state:
                p.copy_(trainable_state[name].to(p.device))


