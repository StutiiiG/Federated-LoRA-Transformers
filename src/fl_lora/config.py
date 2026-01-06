from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class ExperimentConfig:
    # core
    method: str  # centralized_full | fedavg_lora | fedprox_lora | feddual_lora
    model_name: str
    seed: int

    # data
    dataset_name: str  # glue
    dataset_config: str  # sst2
    max_length: int
    dirichlet_alpha: float
    num_clients: int
    tiny: bool
    tiny_train_per_client: int
    tiny_eval_size: int

    # FL
    rounds: int
    client_fraction: float
    local_steps: int
    batch_size: int

    # optimizer
    lr: float
    weight_decay: float

    # FedProx
    prox_mu: float

    # FedDUAL
    dual_beta_mode: str  # fixed | adaptive
    dual_beta_fixed: float
    dual_beta_k: float  # sigmoid sharpness
    dual_agg_mode: str  # uniform | dynamic
    dual_agg_temp: float

    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]

    # output
    out_dir: str

def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    return ExperimentConfig(**data)


