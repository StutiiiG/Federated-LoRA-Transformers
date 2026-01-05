from fl_lora.config import load_config

def main():
    cfg = load_config("configs/federated_small.yaml")
    print("Loaded config:", cfg)
    # TODO: wire into your real training loop
    # This script is a stable entrypoint for reproducibility.

if __name__ == "__main__":
    main()

