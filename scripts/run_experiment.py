import json
import os
import argparse
from fedlora.config import load_config
from fedlora.fl.server import run_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    summary = run_experiment(cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

