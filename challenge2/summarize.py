"""
summarize.py  -  builds the model comparison table from results/*.json. Run:  python summarize.py
(The final notebook uses the same function, so the table always matches the saved results.)
"""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def comparison_table():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        b, t = r["baseline"], r["tuned"]
        rows.append({
            "Model": r["model"],
            "Baseline F1": b["f1_mean"],
            "Tuned F1": t["f1_mean"],
            "Tuned F1 std": t["f1_std"],
            "Tuned F1 + best threshold": t["f1_at_best_threshold_oof"],
            "Precision": t["precision_mean"],
            "Recall": t["recall_mean"],
            "ROC AUC": t["roc_auc_mean"],
            "Train F1": t["train_f1_mean"],
            "Train-CV gap": t["train_f1_mean"] - t["f1_mean"],
            "Seconds": r["seconds"],
            "Best params": r["best_params"],
        })
    return pd.DataFrame(rows).sort_values("Tuned F1 + best threshold", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
    print(comparison_table().round(4).to_string(index=False))
