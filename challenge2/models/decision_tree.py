
"""
Decision Tree Classifier - Challenge 2
Uses shared experiment code for baseline evaluation,
hyperparameter tuning, and saving results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.tree import DecisionTreeClassifier
from common import RANDOM_STATE, run_experiment

if __name__ == "__main__":
    run_experiment(
        name="Decision Tree",
        estimator=DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        param_grid={
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_leaf": [1, 5, 20, 50],
            "class_weight": [None, "balanced"],
        },
        preprocessing="tree",
    )
