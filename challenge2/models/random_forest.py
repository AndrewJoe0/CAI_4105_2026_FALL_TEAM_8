
"""
Random Forest Classifier - Challenge 2
Uses shared experiment code for baseline evaluation,
hyperparameter tuning, and saving results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier
from common import RANDOM_STATE, run_experiment

if __name__ == "__main__":
    run_experiment(
        name="Random Forest",
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        param_grid={
            "n_estimators": [200, 500],
            "max_depth": [6, 10, None],
            "min_samples_leaf": [1, 5, 10],
            "class_weight": [None, "balanced_subsample"],
        },
        preprocessing="tree",
    )
