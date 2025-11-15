
# Federated LoRA on Transformer Models

> Federated fine-tuning of a Transformer model using LoRA adapters, with multiple simulated clients and aggregation of adapter weights only.

This repo contains a single Jupyter notebook that implements a **federated learning setup for LoRA**:
- A base Transformer model (from Hugging Face)
- Multiple **simulated clients** with their own local data
- Each client fine-tunes only **LoRA parameters**
- A central server that **aggregates LoRA weights** (no raw data is shared)

---

## 🧩 Project Overview

The goal of this project is to explore:

- How **parameter-efficient fine-tuning (LoRA)** behaves in a **federated, non-IID** setting  
- The trade-offs between **LoRA rank vs. accuracy vs. communication cost**  
- How to run LLM-style experiments on a **single GPU / limited compute** by only training small adapter modules

All the code lives in a single notebook:

- `FL_Lora.ipynb` – complete experiment pipeline (data loading, client splits, LoRA setup, federated rounds, evaluation)

---

## 🏗 Model & Training Architecture

At a high level, training looks like this:

1. **Server** initializes the base Transformer and a LoRA adapter.
2. Data is **split across clients** (you can control the number of clients and how skewed the data is).
3. For each federated round:
   - The server **broadcasts** the current LoRA weights to each client.
   - Each client:
     - Loads the base model (kept **frozen**) + LoRA adapter
     - Trains on its **local dataset** (only LoRA parameters update)
     - Sends updated LoRA weights back
   - The server **aggregates** the LoRA weights (FedAvg-style).
4. The server periodically evaluates the global adapter on a validation/test set.

### Diagram (Federated LoRA)

```mermaid
flowchart LR
    subgraph Server
        S_Base[Base Transformer (frozen)]
        S_LoRA[Global LoRA Weights]
    end

    subgraph Client1
        C1_Base[Base Transformer (frozen)]
        C1_LoRA[Local LoRA Weights]
        C1_Data[(Local Data D1)]
    end

    subgraph Client2
        C2_Base[Base Transformer (frozen)]
        C2_LoRA[Local LoRA Weights]
        C2_Data[(Local Data D2)]
    end

    subgraph ClientN
        CN_Base[Base Transformer (frozen)]
        CN_LoRA[Local LoRA Weights]
        CN_Data[(Local Data DN)]
    end

    S_LoRA -->|broadcast LoRA| C1_LoRA
    S_LoRA -->|broadcast LoRA| C2_LoRA
    S_LoRA -->|broadcast LoRA| CN_LoRA

    C1_LoRA -->|train on D1| C1_LoRA
    C2_LoRA -->|train on D2| C2_LoRA
    CN_LoRA -->|train on DN| CN_LoRA

    C1_LoRA -->|send updates| S_LoRA
    C2_LoRA -->|send updates| S_LoRA
    CN_LoRA -->|send updates| S_LoRA
