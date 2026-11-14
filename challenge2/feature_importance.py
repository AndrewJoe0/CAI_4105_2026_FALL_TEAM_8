
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common import load_train, make_pipeline, RANDOM_STATE

ROOT = Path(__file__).resolve().parent

# Load the saved best parameters
results_path = ROOT / "results" / "random_forest.json"
results = json.loads(results_path.read_text())
params = results["best_params"]

# Load training data only
X, y = load_train()

# Build the same preprocessing pipeline
forest = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=2,
    **params
)

pipeline = make_pipeline(forest, kind="tree")

# Fit one final model for feature importance
pipeline.fit(X, y)

# Extract feature names and importance values
preprocessor = pipeline.named_steps["prep"]
columns = preprocessor.named_steps["columns"]
feature_names = columns.get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

# Select the 15 most important features
top_indices = np.argsort(importances)[-15:]

plt.figure(figsize=(11, 7))
plt.barh(
    feature_names[top_indices],
    importances[top_indices]
)

plt.xlabel("Feature Importance")
plt.title("Random Forest - Top 15 Important Features")
plt.tight_layout()

# Save the figure
output = ROOT / "results" / "random_forest_importance.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print(f"Feature importance plot saved to: {output}")
