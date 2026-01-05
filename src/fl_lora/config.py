from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class FederatedConfig:
    num_clients: int
    num_rounds: int
    client_frac: float
    seed: int = 42

def load_config(path: str | Path) -> FederatedConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    return FederatedConfig(**data)

