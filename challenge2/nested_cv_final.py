"""
nested_cv_final.py  -  honest performance estimate for the FINAL model (Random Forest).

Why: the Grid Search scores in results/*.json were picked as the BEST of many settings on the same folds,
so they are slightly optimistic. Here the whole procedure is repeated inside each outer fold:
    1. Grid Search on the outer-training part only (inner 3-fold CV)
    2. choose the decision threshold on inner out-of-fold probabilities (outer-training part only)
    3. fit on the outer-training part, score on the outer fold that was never touched
The mean over the 5 outer folds is our best estimate of the F1 we should get on new data.
The grid is a smaller neighbourhood around the best setting found in results/random_forest.json
(the full grid was too slow to repeat 5 times), n_estimators is fixed to the final 500.

Run:  python challenge2/nested_cv_final.py     (about 10-15 minutes)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from common import RANDOM_STATE, RESULTS_DIR, best_threshold, get_cv, load_train, make_pipeline

PARAM_GRID = {"model__max_depth": [10, None], "model__min_samples_leaf": [3, 5, 10]}

if __name__ == "__main__":
    X, y = load_train()
    pipe = make_pipeline(RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                random_state=RANDOM_STATE), kind="tree")
    inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    rows, t0 = [], time.time()
    for k, (tr, va) in enumerate(get_cv().split(X, y), 1):
        gs = GridSearchCV(pipe, PARAM_GRID, scoring="f1", cv=inner, n_jobs=2).fit(X.iloc[tr], y.iloc[tr])
        inner_oof = cross_val_predict(gs.best_estimator_, X.iloc[tr], y.iloc[tr], cv=inner,
                                      method="predict_proba", n_jobs=2)[:, 1]
        thr, _ = best_threshold(y.iloc[tr].values, inner_oof)
        proba = gs.best_estimator_.predict_proba(X.iloc[va])[:, 1]
        pred05, pred_thr = proba >= 0.5, proba >= thr
        row = {"fold": k, "best_params": {p.replace("model__", ""): v for p, v in gs.best_params_.items()},
               "threshold": thr, "f1_at_0.5": f1_score(y.iloc[va], pred05),
               "f1_with_threshold": f1_score(y.iloc[va], pred_thr),
               "precision": precision_score(y.iloc[va], pred_thr), "recall": recall_score(y.iloc[va], pred_thr)}
        rows.append(row)
        print(f"outer fold {k}: F1@0.5 = {row['f1_at_0.5']:.4f}   F1 with threshold {thr:.3f} = "
              f"{row['f1_with_threshold']:.4f}   {gs.best_params_}   ({time.time() - t0:.0f}s)", flush=True)

    def ms(key):
        v = np.array([r[key] for r in rows]); return float(v.mean()), float(v.std(ddof=1))

    out = {"model": "Random Forest", "grid": {k.replace("model__", ""): v for k, v in PARAM_GRID.items()},
           "folds": rows, "seconds": round(time.time() - t0)}
    for key in ["f1_at_0.5", "f1_with_threshold", "precision", "recall"]:
        out[key + "_mean"], out[key + "_std"] = ms(key)
    (RESULTS_DIR / "final_model_nested_cv.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nNESTED CV  F1@0.5 = {out['f1_at_0.5_mean']:.4f} +/- {out['f1_at_0.5_std']:.4f}   "
          f"F1 with tuned threshold = {out['f1_with_threshold_mean']:.4f} +/- {out['f1_with_threshold_std']:.4f}")
