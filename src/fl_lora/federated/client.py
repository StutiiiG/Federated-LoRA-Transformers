from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from transformers import Trainer, TrainingArguments

from fedlora.metrics import accuracy_from_logits
from fedlora.comms import trainable_state_dict
from fedlora.fl.losses import kl_alignment_loss

@dataclass
class ClientResult:
    num_samples: int
    trainable_state: Dict[str, torch.Tensor]
    metrics: Dict[str, Any]

class FedProxTrainer(Trainer):
    def __init__(self, *args, prox_mu: float, global_trainable: Dict[str, torch.Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self.prox_mu = prox_mu
        self.global_trainable = {k: v.to(self.model.device) for k, v in global_trainable.items()}

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss

        # proximal term on trainable params
        prox = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.global_trainable:
                prox = prox + torch.sum((p - self.global_trainable[name]) ** 2)
        loss = loss + 0.5 * self.prox_mu * prox
        return (loss, outputs) if return_outputs else loss

class FedDualTrainer(Trainer):
    def __init__(
        self,
        *args,
        global_model,
        beta_mode: str,
        beta_fixed: float,
        beta_k: float,
        beta_ref_acc: float,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.beta_mode = beta_mode
        self.beta_fixed = beta_fixed
        self.beta_k = beta_k
        self.beta_ref_acc = beta_ref_acc
        self.local_acc_ma = None

    def _beta(self) -> float:
        if self.beta_mode == "fixed":
            return float(self.beta_fixed)
        # adaptive: sigmoid(k*(local_acc_ma - beta_ref_acc))
        if self.local_acc_ma is None:
            return float(self.beta_fixed)
        x = self.beta_k * (self.local_acc_ma - self.beta_ref_acc)
        return float(1.0 / (1.0 + np.exp(-x)))

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits
        ce = outputs.loss

        with torch.no_grad():
            self.global_model.eval()
            teacher_out = self.global_model(**inputs)
            teacher_logits = teacher_out.logits

        # update moving average local accuracy (cheap)
        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)
            acc = (preds == labels).float().mean().item()
            if self.local_acc_ma is None:
                self.local_acc_ma = acc
            else:
                self.local_acc_ma = 0.9 * self.local_acc_ma + 0.1 * acc

        beta = self._beta()
        kl = kl_alignment_loss(student_logits, teacher_logits, temperature=1.0)
        loss = (1 - beta) * ce + beta * kl

        return (loss, outputs) if return_outputs else loss

def quick_accuracy(model, dataset_subset: Subset, batch_size: int) -> float:
    dl = DataLoader(dataset_subset, batch_size=batch_size)
    model.eval()
    logits_list = []
    labels_list = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits_list.append(out.logits.detach().cpu().numpy())
            labels_list.append(batch["labels"].detach().cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return accuracy_from_logits(logits, labels)

def train_one_client(
    method: str,
    client_id: int,
    global_model,
    train_dataset,
    client_indices: np.ndarray,
    eval_subset_size: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    local_steps: int,
    seed: int,
    prox_mu: float,
    dual_beta_mode: str,
    dual_beta_fixed: float,
    dual_beta_k: float,
) -> ClientResult:
    # create a client subset
    client_ds = Subset(train_dataset, client_indices.tolist())

    # small subset to estimate "global acc on client distribution"
    n_eval = min(eval_subset_size, len(client_indices))
    eval_ds = Subset(train_dataset, client_indices[:n_eval].tolist()) if n_eval > 0 else client_ds

    # clone global model weights into a new model instance
    # Note: caller should create the model instance, here we deep-copy by state_dict.
    model = type(global_model).from_pretrained(getattr(global_model, "name_or_path", None))  # not reliable for PEFT
    raise RuntimeError("Do not call train_one_client directly; use server.run which constructs client models safely.")

