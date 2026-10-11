
# CAP 4105 - Team 8 - Challenge 1
# Complete rental price prediction workflow

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import GridSearchCV

from preprocessing import (
    load_splits,
    load_test,
    build_preprocessor,
    rmse,
    FEATURES,
    TARGET
)

RANDOM_STATE = 42


def make_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model)
    ])


# 1. Load shared training and validation data

train, val = load_splits()

X_train = train[FEATURES]
y_train = train[TARGET]
X_val = val[FEATURES]
y_val = val[TARGET]

assert len(train) > 0 and len(val) > 0
assert not y_train.isna().any()
assert not y_val.isna().any()

print("Training rows:", len(train))
print("Validation rows:", len(val))


# 2. Define four required regression models

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )
}

results = []
candidates = {}


def evaluate(name, pipeline):
    train_predictions = pipeline.predict(X_train)
    val_predictions = pipeline.predict(X_val)

    train_score = rmse(y_train, train_predictions)
    val_score = rmse(y_val, val_predictions)

    results.append({
        "Model": name,
        "Training RMSE": train_score,
        "Validation RMSE": val_score,
        "RMSE Gap": val_score - train_score
    })

    candidates[name] = pipeline

    print(
        name,
        "| Training RMSE:", round(train_score, 2),
        "| Validation RMSE:", round(val_score, 2)
    )


# 3. Train and evaluate baseline models

print("\nBASELINE MODEL RESULTS")

for name, model in models.items():
    pipeline = make_pipeline(model)
    pipeline.fit(X_train, y_train)
    evaluate(name, pipeline)


# 4. Grid Search: Random Forest

print("\nTUNING RANDOM FOREST")

rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, None],
    "model__min_samples_leaf": [1, 3]
}

rf_search = GridSearchCV(
    estimator=make_pipeline(
        RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1
        )
    ),
    param_grid=rf_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    refit=True
)

rf_search.fit(X_train, y_train)

print("Best RF parameters:", rf_search.best_params_)
print("RF CV RMSE:", round(-rf_search.best_score_, 2))

evaluate("Tuned Random Forest", rf_search.best_estimator_)


# 5. Grid Search: Gradient Boosting

print("\nTUNING GRADIENT BOOSTING")

gb_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
    "model__min_samples_leaf": [1, 5]
}

gb_search = GridSearchCV(
    estimator=make_pipeline(
        GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    ),
    param_grid=gb_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    refit=True
)

gb_search.fit(X_train, y_train)

print("Best GB parameters:", gb_search.best_params_)
print("GB CV RMSE:", round(-gb_search.best_score_, 2))

evaluate(
    "Tuned Gradient Boosting",
    gb_search.best_estimator_
)


# 6. Compare all models

comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    "Validation RMSE"
).reset_index(drop=True)

comparison.to_csv("model_comparison.csv", index=False)

print("\nCOMPLETE MODEL COMPARISON")
print(comparison.round(2).to_string(index=False))

print("\nOVERFITTING ANALYSIS")
for _, row in comparison.iterrows():
    print(
        row["Model"],
        "| Validation - Training RMSE:",
        round(row["RMSE Gap"], 2)
    )

print(
    "\nA large positive gap may indicate overfitting."
)


# 7. Select best model by validation RMSE

best_name = comparison.iloc[0]["Model"]
best_pipeline = candidates[best_name]

print("\nSELECTED FINAL MODEL:", best_name)
print(
    "Validation RMSE:",
    round(comparison.iloc[0]["Validation RMSE"], 2)
)

print(
    "Selected because it achieved the lowest "
    "validation RMSE among evaluated candidates."
)


# 8. Plot actual vs predicted prices

val_predictions = best_pipeline.predict(X_val)

plt.figure(figsize=(8, 6))
plt.scatter(y_val, val_predictions, alpha=0.35)

limits = [
    min(y_val.min(), val_predictions.min()),
    max(y_val.max(), val_predictions.max())
]

plt.plot(limits, limits, "r--")
plt.xlabel("Actual Rental Price ($)")
plt.ylabel("Predicted Rental Price ($)")
plt.title("Final Model: Actual vs Predicted")
plt.tight_layout()
plt.savefig("final_model_plot.png", dpi=150)
plt.show()


# 9. Train final model on all available training data

full_train_path = "data/challenge1_train.csv"

if os.path.exists(full_train_path):
    full_train = pd.read_csv(full_train_path)
else:
    # Use the team's shared training and validation
    # partitions if the original CSV is unavailable.
    full_train = pd.concat(
        [train, val],
        ignore_index=True
    )

if "ID" in full_train.columns:
    assert full_train["ID"].is_unique

X_full = full_train[FEATURES]
y_full = full_train[TARGET]

assert not y_full.isna().any()

print("\nRetraining selected model on", len(full_train), "rows")

# Refit selected pipeline using all labeled training data.
best_pipeline.fit(X_full, y_full)


# 10. Generate competition predictions

test = load_test()

assert len(test) == 2467, (
    "Expected 2467 test observations."
)
assert "ID" in test.columns
assert test["ID"].is_unique
assert TARGET not in test.columns

X_test = test[FEATURES]

predicted_prices = best_pipeline.predict(X_test)

assert len(predicted_prices) == len(test)
assert np.isfinite(predicted_prices).all()

predictions = pd.DataFrame({
    "ID": test["ID"].values,
    "price": predicted_prices
})

assert predictions["ID"].equals(
    test["ID"].reset_index(drop=True)
)

predictions.to_csv("predictions.csv", index=False)

print("\nPREDICTIONS CREATED")
print(predictions.head())
print("Prediction rows:", len(predictions))


# 11. Final report

print("\n========== FINAL REPORT ==========")
print("Selected model:", best_name)
print(
    "Validation RMSE:",
    round(comparison.iloc[0]["Validation RMSE"], 2)
)
print("Full training rows:", len(full_train))
print("Test predictions:", len(predictions))
print("Files generated:")
print("- model_comparison.csv")
print("- predictions.csv")
print("- final_model_plot.png")
print("\nChallenge modeling workflow completed.")
