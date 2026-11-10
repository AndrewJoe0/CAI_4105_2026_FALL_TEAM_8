"""
TEMPLATE  -  copy this file to models/<your_model>.py and fill in the 4 marked lines.
Run it from the project folder with:  python models/<your_model>.py
It saves results/<model name>.json, which the final notebook reads automatically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.neighbors import KNeighborsClassifier      # <- 1. import YOUR model
from common import RANDOM_STATE, run_experiment

if __name__ == "__main__":
    run_experiment(
        name="K-Nearest Neighbors",                        # <- 2. label shown in the results table
        estimator=KNeighborsClassifier(),                  # <- 3. the model with DEFAULT settings (baseline)
        param_grid={                                       # <- 4. Grid Search values (no "model__" prefix)
            "n_neighbors": [5, 15, 31, 61],
            "weights": ["uniform", "distance"],
        },
        preprocessing="scaled",                            # "scaled" for LR/NB/KNN/SVM, "tree" for trees
    )
