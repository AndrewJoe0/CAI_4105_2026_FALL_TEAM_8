"""
common.py  -  shared code for Challenge 2 (predict Revenue = purchase / no purchase).

EVERY model script uses this file, so all models are compared in exactly the same way:
  * same data loading
  * same feature engineering + preprocessing
  * same 5-fold stratified cross-validation (same folds for every model)
  * same metrics, with F1 of the positive class (Revenue=True) as the main one

Nothing in here uses the test set except load_test().
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OOF_DIR = RESULTS_DIR / "oof"

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "Revenue"

# --------------------------------------------------------------------------------------
# 1. Loading data
# --------------------------------------------------------------------------------------
def load_train():
    """Return X (features, ID removed) and y (0/1 target) from the training file."""
    df = pd.read_csv(DATA_DIR / "challenge2_train.csv")
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, "ID"])          # ID must NOT be used as a predictor
    return X, y


def load_test():
    """Return test features X_test and the ID column (needed for predictions.csv)."""
    df = pd.read_csv(DATA_DIR / "challenge2_test.csv")
    return df.drop(columns=["ID"]), df["ID"]


# --------------------------------------------------------------------------------------
# 2. Feature engineering (row-by-row only, so it cannot leak information between rows)
# --------------------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                           + df["ProductRelated_Duration"])
    # average seconds spent per page (max(.,1) avoids dividing by zero)
    df["AvgTime_Product"] = df["ProductRelated_Duration"] / df["ProductRelated"].clip(lower=1)
    df["AvgTime_Admin"] = df["Administrative_Duration"] / df["Administrative"].clip(lower=1)
    df["AvgTime_Info"] = df["Informational_Duration"] / df["Informational"].clip(lower=1)
    df["BounceExitDiff"] = df["ExitRates"] - df["BounceRates"]
    # PageValues is by far the strongest signal: ~82% of buyers have PageValues > 0
    df["HasPageValue"] = (df["PageValues"] > 0).astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    return df


# columns that are heavily right-skewed (long tail) -> log1p then scale
SKEWED = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
          "ProductRelated", "ProductRelated_Duration", "PageValues", "TotalPages",
          "TotalDuration", "AvgTime_Product", "AvgTime_Admin", "AvgTime_Info"]
OTHER_NUMERIC = ["BounceRates", "ExitRates", "SpecialDay", "BounceExitDiff",
                 "HasPageValue", "Weekend"]
# numbers that are really category codes (e.g. Browser 1..13) -> one-hot, not numeric
CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]


def build_preprocessor(kind="scaled"):
    """
    kind = "scaled": for Logistic Regression, Naive Bayes, KNN, SVM
           (log1p on skewed columns, then StandardScaler, one-hot categoricals)
    kind = "tree"  : for Decision Tree, Random Forest, Gradient Boosting
           (no scaling or log needed, trees only care about the order of values)
    """
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                           sparse_output=False)
    if kind == "scaled":
        skewed = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("scale", StandardScaler())])
        numeric_other = StandardScaler()
    elif kind == "tree":
        skewed = "passthrough"
        numeric_other = "passthrough"
    else:
        raise ValueError("kind must be 'scaled' or 'tree'")

    columns = ColumnTransformer([
        ("skewed", skewed, SKEWED),
        ("numeric", numeric_other, OTHER_NUMERIC),
        ("categorical", onehot, CATEGORICAL),
    ])
    return Pipeline([("features", FunctionTransformer(add_features)), ("columns", columns)])


def make_pipeline(estimator, kind="scaled"):
    return Pipeline([("prep", build_preprocessor(kind)), ("model", estimator)])


# --------------------------------------------------------------------------------------
# 3. Validation strategy: 5-fold STRATIFIED cross-validation, identical for every model
# --------------------------------------------------------------------------------------
def get_cv():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def get_scores(model, X):
    """Probability of the positive class (or a decision score for models without probabilities)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def best_threshold(y, scores):
    """The cut-off on `scores` that gives the highest F1 for the positive class."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.30, 0.995, 300)))
    f1s = [f1_score(y, scores >= t) for t in candidates]
    i = int(np.argmax(f1s))
    return float(candidates[i]), float(f1s[i])


def cv_evaluate(pipeline, X, y, cv=None):
    """
    Fit `pipeline` on each training fold and score it on the held-out fold.
    Returns a dict of metrics (mean and std over folds) plus out-of-fold scores.
    'train_f1' next to 'f1' shows overfitting (big gap = overfitting).
    """
    cv = cv or get_cv()
    rows, oof = [], np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        s_val = get_scores(model, X.iloc[val_idx])
        oof[val_idx] = s_val
        pred_val = model.predict(X.iloc[val_idx])
        pred_tr = model.predict(X.iloc[train_idx])
        rows.append({
            "f1": f1_score(y.iloc[val_idx], pred_val),
            "precision": precision_score(y.iloc[val_idx], pred_val, zero_division=0),
            "recall": recall_score(y.iloc[val_idx], pred_val),
            "roc_auc": roc_auc_score(y.iloc[val_idx], s_val),
            "avg_precision": average_precision_score(y.iloc[val_idx], s_val),
            "train_f1": f1_score(y.iloc[train_idx], pred_tr),
        })
    folds = pd.DataFrame(rows)
    out = {f"{m}_mean": float(folds[m].mean()) for m in folds}
    out.update({f"{m}_std": float(folds[m].std()) for m in folds})
    thr, f1_thr = best_threshold(y.values, oof)
    out["best_threshold"], out["f1_at_best_threshold_oof"] = thr, f1_thr
    return out, oof


def nested_cv_f1(pipeline, param_grid, X, y, inner_splits=3):
    """
    UNBIASED estimate of 'tune with Grid Search, then predict new data'.
    Outer loop = our usual 5 folds, inner loop = Grid Search on the outer-training part only.
    Slow, so use it for the final model (or the top 2-3 models), not for every model.
    """
    outer = get_cv()
    inner = StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in outer.split(X, y):
        gs = GridSearchCV(clone(pipeline), param_grid, scoring="f1", cv=inner, n_jobs=2)
        gs.fit(X.iloc[tr], y.iloc[tr])
        scores.append(f1_score(y.iloc[va], gs.predict(X.iloc[va])))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


# --------------------------------------------------------------------------------------
# 4. One call per model: baseline -> Grid Search -> tuned model -> save results
# --------------------------------------------------------------------------------------
def run_experiment(name, estimator, param_grid, preprocessing="scaled", n_jobs=2, note=""):
    """
    name          short label, e.g. "Logistic Regression"
    estimator     an UNFITTED sklearn model with default settings (the baseline)
    param_grid    dict for Grid Search, keys WITHOUT the 'model__' prefix, e.g. {"C": [0.1, 1]}
    preprocessing "scaled" or "tree" (see build_preprocessor)
    Saves results/<name>.json and the out-of-fold scores in results/oof/.
    """
    X, y = load_train()
    pipe = make_pipeline(estimator, preprocessing)
    cv = get_cv()
    slug = name.lower().replace(" ", "_")

    t0 = time.time()
    base, base_oof = cv_evaluate(pipe, X, y, cv)
    print(f"[{name}] baseline   F1 = {base['f1_mean']:.4f} +/- {base['f1_std']:.4f}"
          f"   (train F1 {base['train_f1_mean']:.4f})")

    grid = {f"model__{k}": v for k, v in param_grid.items()}
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    print(f"[{name}] best params = {best_params}")

    tuned, tuned_oof = cv_evaluate(gs.best_estimator_, X, y, cv)
    print(f"[{name}] tuned      F1 = {tuned['f1_mean']:.4f} +/- {tuned['f1_std']:.4f}"
          f"   (train F1 {tuned['train_f1_mean']:.4f})")
    print(f"[{name}] tuned + best threshold {tuned['best_threshold']:.3f}: "
          f"F1 = {tuned['f1_at_best_threshold_oof']:.4f}   ({time.time() - t0:.0f}s)")

    result = {"model": name, "preprocessing": preprocessing, "best_params": best_params,
              "n_grid_combinations": int(len(gs.cv_results_["params"])),
              "baseline": base, "tuned": tuned, "note": note,
              "seconds": round(time.time() - t0, 1)}
    RESULTS_DIR.mkdir(exist_ok=True)
    OOF_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{slug}.json").write_text(json.dumps(result, indent=2, default=str))
    np.save(OOF_DIR / f"{slug}_tuned.npy", tuned_oof)
    return result
