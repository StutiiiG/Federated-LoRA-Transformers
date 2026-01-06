from __future__ import annotations
from peft import LoraConfig, get_peft_model, TaskType

def apply_lora_to_sequence_classifier(model, r: int, alpha: int, dropout: float, target_modules: list[str]):
    # For BLOOMZ sequence classification, attention projections often include "query_key_value" and "dense".
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    return peft_model

