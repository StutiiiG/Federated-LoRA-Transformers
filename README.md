
# Federated LoRA on Transformer Models

> Federated fine-tuning of a Transformer model using LoRA adapters, with multiple simulated clients and aggregation of adapter weights only.

This repo contains a **federated learning setup for LoRA**:
- A base Transformer model (from Hugging Face)
- Multiple **simulated clients** with their own local data
- Each client fine-tunes only **LoRA parameters**
- A central server that **aggregates LoRA weights** (no raw data is shared)


##  Project Overview

The goal of this project is to explore:

- How **parameter-efficient fine-tuning (LoRA)** behaves in a **federated, non-IID** setting  
- The trade-offs between **LoRA rank vs. accuracy vs. communication cost**  
- How to run LLM-style experiments on a **single GPU / limited compute** by only training small adapter modules

All the code lives in a single notebook:

- `FL_Lora.ipynb` – complete experiment pipeline (data loading, client splits, LoRA setup, federated rounds, evaluation)


##  Model & Training Architecture

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

## Federated LoRA Architecture 

```mermaid
flowchart LR

subgraph SERVER
    SBase["Base Transformer (frozen)"]
    SLoRA["Global LoRA Weights"]
end

subgraph CLIENT1
    C1Base["Base Transformer (frozen)"]
    C1LoRA["Local LoRA Weights"]
    C1Data["Local Data D1"]
end

subgraph CLIENT2
    C2Base["Base Transformer (frozen)"]
    C2LoRA["Local LoRA Weights"]
    C2Data["Local Data D2"]
end

SLoRA -->|broadcast| C1LoRA
SLoRA -->|broadcast| C2LoRA

C1LoRA -->|local train| C1LoRA
C2LoRA -->|local train| C2LoRA

C1LoRA -->|update| SLoRA
C2LoRA -->|update| SLoRA



