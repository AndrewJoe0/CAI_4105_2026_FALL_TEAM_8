"""Naive Bayes  -  run with:  python models/naive_bayes.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.naive_bayes import GaussianNB
from common import run_experiment

if __name__ == "__main__":
    run_experiment(
        name="Naive Bayes",
        estimator=GaussianNB(),
        param_grid={
            "var_smoothing": [1e-9, 1e-5, 1e-3, 1e-2, 1e-1, 0.3, 1, 3, 10],
            "priors": [None, [0.5, 0.5], [0.7, 0.3]],   # prior P(no purchase), P(purchase)
        },
        preprocessing="scaled",
        note="GaussianNB assumes bell-shaped, independent features; one-hot columns break that assumption.",
    )
