"""
regenerate_oof.py  -  re-creates results/oof/<model>_tuned.npy (out-of-fold scores) for the models whose
file is missing, using the best parameters saved in results/<model>.json and the SAME 5 folds as everything else.
Run:  python challenge2/regenerate_oof.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from common import OOF_DIR, RANDOM_STATE, RESULTS_DIR, cv_evaluate, load_train, make_pipeline

MODELS = {"decision_tree": DecisionTreeClassifier, "random_forest": RandomForestClassifier}

if __name__ == "__main__":
    X, y = load_train()
    OOF_DIR.mkdir(exist_ok=True)
    for slug, cls in MODELS.items():
        out_file = OOF_DIR / f"{slug}_tuned.npy"
        if out_file.exists():
            print(slug, "already has OOF scores, skipping"); continue
        saved = json.loads((RESULTS_DIR / f"{slug}.json").read_text())
        pipe = make_pipeline(cls(random_state=RANDOM_STATE, **saved["best_params"]), saved["preprocessing"])
        res, oof = cv_evaluate(pipe, X, y)
        np.save(out_file, oof)
        print(f"{slug}: F1 {res['f1_mean']:.4f} (saved JSON says {saved['tuned']['f1_mean']:.4f}), "
              f"best threshold {res['best_threshold']:.3f} -> F1 {res['f1_at_best_threshold_oof']:.4f} "
              f"(saved JSON says {saved['tuned']['f1_at_best_threshold_oof']:.4f})")
