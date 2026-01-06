import argparse
import glob
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.results_root, "**/summary.json"), recursive=True):
        try:
            df = pd.read_json(path, typ="series")
            rows.append(df.to_dict())
        except Exception:
            continue

    if not rows:
        print("No summaries found.")
        return

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.results_root, "summary.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.sort_values(by="final_val_accuracy", ascending=False).head(10))

if __name__ == "__main__":
    main()

