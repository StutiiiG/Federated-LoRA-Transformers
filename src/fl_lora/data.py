from __future__ import annotations
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def load_sst2(model_name: str, max_length: int):
    ds = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tok(batch["sentence"], truncation=True, padding="max_length", max_length=max_length)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["sentence", "idx"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch")
    return ds["train"], ds["validation"], tok

def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int):
    """
    Non-IID partition by sampling per-class proportions from Dirichlet(alpha).
    Returns list of index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    # indices per class
    class_indices = {c: np.where(labels == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idxs = class_indices[c]
        proportions = rng.dirichlet([alpha] * num_clients)
        # split sizes
        splits = (proportions * len(idxs)).astype(int)
        # fix rounding to sum exactly
        diff = len(idxs) - splits.sum()
        if diff > 0:
            splits[:diff] += 1
        elif diff < 0:
            # subtract where possible
            for i in range(num_clients):
                if diff == 0:
                    break
                if splits[i] > 0:
                    splits[i] -= 1
                    diff += 1

        start = 0
        for i in range(num_clients):
            end = start + splits[i]
            client_indices[i].extend(idxs[start:end].tolist())
            start = end

    # shuffle within each client
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i], dtype=int)

    return client_indices

