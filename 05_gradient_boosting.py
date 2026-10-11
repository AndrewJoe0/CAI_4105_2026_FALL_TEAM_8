
# CAP 4105 - Machine Learning Challenge 1
# Team 8
# Gradient Boosting Regression and Grid Search
#
# Goal: Predict monthly apartment rental prices.
# Evaluation metric: Root Mean Squared Error (RMSE)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from preprocessing import (
    load_splits,
    build_preprocessor,
    rmse,
    FEATURES,
    TARGET
)


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading training and validation data...")

train_split, val_split = load_splits()

X_train = train_split[FEATURES]
y_train = train_split[TARGET]

X_val = val_split[FEATURES]
y_val = val_split[TARGET]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================
# 2. BASELINE GRADIENT BOOSTING
# ==========================================

print("\nTraining baseline Gradient Boosting...")

baseline_model = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42
    ))
])

baseline_model.fit(X_train, y_train)

baseline_train_pred = baseline_model.predict(X_train)
baseline_val_pred = baseline_model.predict(X_val)

baseline_train_rmse = rmse(
    y_train, baseline_train_pred
)

baseline_val_rmse = rmse(
    y_val, baseline_val_pred
)

print("\nBaseline Results:")
print("Training RMSE:", round(baseline_train_rmse, 2))
print("Validation RMSE:", round(baseline_val_rmse, 2))


# ==========================================
# 3. GRID SEARCH HYPERPARAMETER TUNING
# ==========================================

print("\nStarting Grid Search...")

param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_leaf": [1, 5]
}

tuning_pipeline = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", GradientBoostingRegressor(
        random_state=42
    ))
])

grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation RMSE:")
print(round(-grid_search.best_score_, 2))


# ==========================================
# 4. EVALUATE TUNED MODEL
# ==========================================

best_model = grid_search.best_estimator_

tuned_train_pred = best_model.predict(X_train)
tuned_val_pred = best_model.predict(X_val)

tuned_train_rmse = rmse(
    y_train, tuned_train_pred
)

tuned_val_rmse = rmse(
    y_val, tuned_val_pred
)

print("\nTuned Gradient Boosting Results:")
print("Training RMSE:", round(tuned_train_rmse, 2))
print("Validation RMSE:", round(tuned_val_rmse, 2))


# ==========================================
# 5. COMPARE BASELINE AND TUNED MODELS
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Gradient Boosting",
        "Tuned Gradient Boosting"
    ],
    "Training RMSE": [
        baseline_train_rmse,
        tuned_train_rmse
    ],
    "Validation RMSE": [
        baseline_val_rmse,
        tuned_val_rmse
    ]
})

results = results.sort_values(
    by="Validation RMSE"
)

print("\nMODEL COMPARISON")
print(results.round(2).to_string(index=False))


# ==========================================
# 6. CHECK FOR OVERFITTING
# ==========================================

print("\nOverfitting Analysis:")

gap = tuned_val_rmse - tuned_train_rmse

print("RMSE Gap:", round(gap, 2))

if gap > 0:
    print(
        "Validation error is higher than training error."
    )
    print(
        "This may indicate overfitting."
    )
else:
    print(
        "No positive training-validation RMSE gap."
    )

if tuned_val_rmse < baseline_val_rmse:
    print("Grid Search improved validation RMSE.")
else:
    print(
        "Grid Search did not improve validation RMSE."
    )


# ==========================================
# 7. ACTUAL VS PREDICTED VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_val,
    tuned_val_pred,
    alpha=0.4,
    label="Predictions"
)

limits = [
    min(y_val.min(), tuned_val_pred.min()),
    max(y_val.max(), tuned_val_pred.max())
]

plt.plot(
    limits,
    limits,
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Rental Price ($)")
plt.ylabel("Predicted Rental Price ($)")
plt.title("Gradient Boosting: Actual vs Predicted")
plt.legend()
plt.tight_layout()

plt.savefig(
    "gradient_boosting_results.png",
    dpi=150
)

plt.show()


# ==========================================
# 8. SAVE MODEL COMPARISON RESULTS
# ==========================================

results.to_csv(
    "gradient_boosting_comparison.csv",
    index=False
)

print("\nResults saved successfully.")


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print(
    "Baseline Validation RMSE:",
    round(baseline_val_rmse, 2)
)

print(
    "Tuned Validation RMSE:",
    round(tuned_val_rmse, 2)
)

print("Best Hyperparameters:")
print(grid_search.best_params_)

print("\nGradient Boosting analysis completed.")
print(
    "Compare these results with Linear Regression,"
)
print(
    "Decision Tree, and Random Forest before"
)
print("selecting the final model.")
