"""Gradient Boosting  -  run with:  python challenge2/models/gradient_boosting.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import GradientBoostingClassifier
from common import RANDOM_STATE, run_experiment

if __name__ == "__main__":
    run_experiment(
        name="Gradient Boosting",
        estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [200, 400],           # number of small trees added one after another
            "learning_rate": [0.01, 0.02, 0.03, 0.05],  # how much each new tree is allowed to correct
            "max_depth": [4, 5, 6],               # size of each tree (small trees = less overfitting)
            "subsample": [0.6, 0.8],              # share of rows each tree sees (<1 adds randomness)
        },
        preprocessing="tree",
        note="No class_weight option in sklearn's GradientBoostingClassifier; the F1-optimal threshold handles the imbalance.",
    )
