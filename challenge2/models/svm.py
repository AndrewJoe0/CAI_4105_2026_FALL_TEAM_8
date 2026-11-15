
"""
Support Vector Machine (SVM) for Challenge 2.
Uses the shared preprocessing and validation workflow.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.svm import SVC
from common import RANDOM_STATE, run_experiment

if __name__ == "__main__":
    run_experiment(
        name="Support Vector Machine",
        estimator=SVC(),
    param_grid={
        "C": [0.01, 0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1, 1.0],
        "class_weight": [None, "balanced"],
    },
        preprocessing="scaled",
    )
