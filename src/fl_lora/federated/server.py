from __future__ import annotations
import os
from dataclasses import asdict
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from fedlora.seed import set_seed
from fedlora.data import load_sst2, dirichlet_partition
from fedlora.lora import apply_lora_to_sequence_classifier
from fedlora.comms import trainable_state_dict, bytes_of_state, num_trainable_params
from fedlora.metrics import accuracy_from_logits
from fedlora.fl.aggregation import weighted_average_state, apply_state_to_model
from fedlora.fl.client import FedProxTrainer, FedDualTrainer, quick_accuracy

def select_clients(num_clients: int, frac: float, rng: np.random.Generator) -> List[int]:
    k = max(1, int(round(num_clients * frac)))
    return rng.choice(np.arange(num_clients), size=k, replace=False).tolist()

def build_model(model_name: str) -> torch.nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def make_peft_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: list[str]):
    base = build_model(model_name)
    peft = apply_lora_to_sequence_classifier(base, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_modules=target_modules)
    peft.print_trainable_parameters()
    return peft

def evaluate_accuracy(model, eval_dataset: Dataset, batch_size: int) -> float:
    # Use Trainer for consistency
    args = TrainingArguments(
        output_dir="results/_tmp_eval",
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset)
    preds = trainer.predict(eval_dataset)
    return accuracy_from_logits(preds.predictions, preds.label_ids)

def run_experiment(cfg) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, tok = load_sst2(cfg.model_name, cfg.max_length)
    labels_np = np.array([int(train_ds[i]["labels"]) for i in range(len(train_ds))], dtype=int)

    # partition
    client_splits = dirichlet_partition(labels_np, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed)

    # tiny mode: shrink per client
    if cfg.tiny:
        new_splits = []
        for idxs in client_splits:
            new_splits.append(idxs[: min(cfg.tiny_train_per_client, len(idxs))])
        client_splits = new_splits
        val_eval = Subset(val_ds, list(range(min(cfg.tiny_eval_size, len(val_ds)))))
    else:
        val_eval = val_ds

    rng = np.random.default_rng(cfg.seed + 123)

    # Centralized baseline (full fine-tune)
    if cfg.method == "centralized_full":
        model = build_model(cfg.model_name).to(device)
        args = TrainingArguments(
            output_dir=cfg.out_dir,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            max_steps=cfg.local_steps if cfg.tiny else -1,
            num_train_epochs=1 if cfg.tiny else 1,
            evaluation_strategy="epoch" if not cfg.tiny else "no",
            logging_steps=50,
            report_to=[],
            save_strategy="no",
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_eval)
        trainer.train()
        acc = evaluate_accuracy(model, val_eval, cfg.batch_size)
        return {
            "method": cfg.method,
            "val_accuracy": acc,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_comm_bytes": 0,
        }

    # Federated: LoRA models (trainable: LoRA + head)
    global_model = make_peft_model(cfg.model_name, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.lora_target_modules).to(device)

    # ensure classifier head trainable
    for name, p in global_model.named_parameters():
        if "classifier" in name or "score" in name:
            p.requires_grad = True

    # global trainable state (for comm + prox + aggregation)
    global_trainable = trainable_state_dict(global_model)
    trainable_params = num_trainable_params(global_model)
    down_bytes_per_client = bytes_of_state(global_trainable)  # server -> client
    up_bytes_per_client = bytes_of_state(global_trainable)    # client -> server (trainable only)

    history_rows = []
    total_comm = 0

    for rnd in range(cfg.rounds):
        selected = select_clients(cfg.num_clients, cfg.client_fraction, rng)

        client_states = []
        client_sizes = []
        client_scores = []

        for cid in selected:
            idxs = client_splits[cid]
            if len(idxs) == 0:
                continue

            # fresh client model from global weights
            client_model = make_peft_model(cfg.model_name, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.lora_target_modules).to(device)
            # copy global trainable params into client
            apply_state_to_model(client_model, global_trainable)

            # small subset to estimate global accuracy on this client's distribution (for adaptive beta)
            n_eval = min(cfg.tiny_eval_size if cfg.tiny else 256, len(idxs))
            eval_subset = Subset(train_ds, idxs[:n_eval].tolist())
            beta_ref_acc = quick_accuracy(global_model, eval_subset, batch_size=cfg.batch_size)

            # training args: local_steps controls compute
            args = TrainingArguments(
                output_dir=os.path.join(cfg.out_dir, f"client_{cid}_round_{rnd}"),
                per_device_train_batch_size=cfg.batch_size,
                learning_rate=cfg.lr,
                weight_decay=cfg.weight_decay,
                max_steps=cfg.local_steps,
                logging_steps=50,
                report_to=[],
                save_strategy="no",
                disable_tqdm=True,
            )

            # local train dataset
            local_train = Subset(train_ds, idxs.tolist())

            if cfg.method == "fedprox_lora":
                trainer = FedProxTrainer(
                    model=client_model,
                    args=args,
                    train_dataset=local_train,
                    prox_mu=cfg.prox_mu,
                    global_trainable=global_trainable,
                )
            elif cfg.method == "feddual_lora":
                trainer = FedDualTrainer(
                    model=client_model,
                    args=args,
                    train_dataset=local_train,
                    global_model=global_model,
                    beta_mode=cfg.dual_beta_mode,
                    beta_fixed=cfg.dual_beta_fixed,
                    beta_k=cfg.dual_beta_k,
                    beta_ref_acc=beta_ref_acc,
                )
            else:
                trainer = Trainer(model=client_model, args=args, train_dataset=local_train)

            trainer.train()

            # evaluate client quickly (optional)
            client_acc = quick_accuracy(client_model, eval_subset, batch_size=cfg.batch_size)

            sd = trainable_state_dict(client_model)
            client_states.append(sd)
            client_sizes.append(len(idxs))
            client_scores.append(client_acc)

            # communication: download + upload trainable params
            total_comm += down_bytes_per_client + up_bytes_per_client

        if len(client_states) == 0:
            continue

        # aggregation
        if cfg.method == "feddual_lora" and cfg.dual_agg_mode == "dynamic":
            # dynamic weighting: softmax over client scores (temperature)
            scores = np.array(client_scores, dtype=np.float64)
            temp = max(1e-6, float(cfg.dual_agg_temp))
            w = np.exp(scores / temp)
            w = w / (w.sum() + 1e-12)
            # convert to integer-like weights by scaling with client_sizes
            dyn_weights = [max(1, int(round(wi * 1000))) for wi in w]
            agg = weighted_average_state(client_states, dyn_weights)
        else:
            agg = weighted_average_state(client_states, client_sizes)

        apply_state_to_model(global_model, agg)
        global_trainable = trainable_state_dict(global_model)

        # global evaluation
        val_acc = evaluate_accuracy(global_model, val_eval, cfg.batch_size)

        history_rows.append({
            "round": rnd,
            "val_accuracy": val_acc,
            "selected_clients": len(client_states),
            "trainable_params": trainable_params,
            "bytes_down_per_client": down_bytes_per_client,
            "bytes_up_per_client": up_bytes_per_client,
            "total_comm_bytes_so_far": total_comm,
        })

    df = pd.DataFrame(history_rows)
    df.to_csv(os.path.join(cfg.out_dir, "metrics.csv"), index=False)

    summary = {
        "method": cfg.method,
        "final_val_accuracy": float(df["val_accuracy"].iloc[-1]) if len(df) else None,
        "trainable_params": int(trainable_params),
        "bytes_down_per_client": int(down_bytes_per_client),
        "bytes_up_per_client": int(up_bytes_per_client),
        "total_comm_bytes": int(total_comm),
        "rounds_completed": int(df["round"].max() + 1) if len(df) else 0,
    }
    return summary

