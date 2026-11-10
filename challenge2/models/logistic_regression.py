"""Logistic Regression  -  run with:  python models/logistic_regression.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from common import RANDOM_STATE, run_experiment

if __name__ == "__main__":
    run_experiment(
        name="Logistic Regression",
        estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        param_grid={
            "C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],            # strength of regularization (small = strong)
            "class_weight": [None, "balanced"],       # 'balanced' gives the rare class more weight
        },
        preprocessing="scaled",
    )
