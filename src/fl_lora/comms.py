from __future__ import annotations
import torch
from typing import Dict

def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    """Return only trainable parameters (LoRA + head)."""
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = p.detach().cpu().clone()
    return out

def num_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bytes_of_state(sd: Dict[str, torch.Tensor]) -> int:
    total = 0
    for t in sd.values():
        total += t.numel() * t.element_size()
    return int(total)

