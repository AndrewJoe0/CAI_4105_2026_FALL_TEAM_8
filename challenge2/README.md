# Challenge 2: Predicting Online Shopping Purchases
## CAI 4105, Team 8

**Task:** predict whether an online shopping session ends in a purchase (`Revenue = True`). Only about 15.5% of the 9,247 training sessions are purchases, so the primary metric is the **F1 score of the purchase class** (precision and recall are reported too).

**Final model: Random Forest** (500 trees, `class_weight="balanced_subsample"`, `min_samples_leaf=5`, no depth limit), default 0.5 threshold.
**Validation F1 (5-fold stratified CV): 0.691 ± 0.010.** **Honest estimate for new data (nested CV): about 0.69** (0.695 ± 0.010).

## Deliverables

| File | What it is |
|---|---|
| `challenge2_final_notebook.ipynb` | Final executed notebook: data, preprocessing, validation, all 7 models, final model, nested CV, predictions |
| `predictions.csv` | `ID` and predicted `Revenue` (True/False) for all 3,083 test sessions |
| `README.md` | This summary |
| `environment.yml` | Conda environment with pinned package versions |
| `common.py` | Shared code: data loading, feature engineering, preprocessing, 5-fold CV, metrics, `run_experiment` |
| `models/*.py` | One script per model (Grid Search); each saves `results/<model>.json` |
| `nested_cv_final.py`, `regenerate_oof.py`, `eda.py`, `summarize.py`, `feature_importance.py` | Nested CV for the final model, out-of-fold score helper, EDA, results table, importance plot |

## Approach

1. **Exploration:** no missing values; the target is imbalanced (15.5% purchases); `PageValues` is by far the strongest signal (82% of buyers have a value above 0 vs 11% of non-buyers); purchase rates differ strongly by month and visitor type. `ID` is not used as a predictor.
2. **Preprocessing and feature engineering** (inside a scikit-learn Pipeline, so it is fitted only on training folds): totals and per-page averages of pages and time, `ExitRates - BounceRates`, a `PageValues > 0` flag; log + standardization for the models that need scaling; one-hot encoding of the category codes (`Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`).
3. **Validation strategy:** one **5-fold stratified cross-validation** with a fixed seed, used for *every* model, for Grid Search, and for the final comparison. We report mean ± standard deviation of F1, precision, recall, ROC AUC, and the training-fold F1 (to detect overfitting).
4. **Models:** all seven required models, each with a baseline and a Grid Search (scoring = F1).
5. **Final model estimate:** nested cross-validation repeats the whole tuning procedure inside each outer fold, removing the optimism of "best of many settings".

## Models evaluated (5-fold CV, training data only)

| Model | Baseline F1 | Tuned F1 | Tuned F1 + best threshold* | Precision | Recall | ROC AUC | Train − CV F1 |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.663 | 0.691 ± 0.010 | 0.702 | 0.616 | 0.787 | 0.933 | +0.114 |
| Gradient Boosting | 0.662 | 0.669 ± 0.023 | 0.696 | 0.721 | 0.625 | 0.933 | +0.099 |
| Decision Tree | 0.577 | 0.671 ± 0.016 | 0.682 | 0.710 | 0.637 | 0.918 | +0.026 |
| Logistic Regression | 0.649 | 0.672 ± 0.015 | 0.681 | 0.571 | 0.817 | 0.918 | -0.000 |
| K-Nearest Neighbors | 0.617 | 0.643 ± 0.017 | 0.675 | 0.695 | 0.599 | 0.916 | +0.029 |
| Support Vector Machine | 0.663 | 0.673 ± 0.014 | 0.674 | 0.571 | 0.821 | 0.910 | +0.004 |
| Naive Bayes | 0.464 | 0.624 ± 0.006 | 0.627 | 0.504 | 0.820 | 0.876 | +0.000 |

\* Cut-off chosen on out-of-fold predictions, so slightly optimistic. Precision, recall, ROC AUC and the gap are for the tuned model at the default cut-off.

## Final model and justification

We selected the **Random Forest**:

- **Performance:** highest tuned F1 (0.691) and the smallest fold-to-fold variation among the top models; ROC AUC 0.933 (tied with Gradient Boosting).
- **Overfitting:** its training F1 is higher than its validation F1 (gap 0.11, larger than Logistic Regression, which underfits instead), but its validation score is still the best, and the nested CV shows the score is not an artifact of tuning (0.695 vs 0.691).
- **Complexity and compute:** many trees but few hyperparameters; the Grid Search was slow, a single fit takes seconds.
- **Interpretability:** less than Logistic Regression, but feature importance is clear (`PageValues` dominates, followed by exit rate, session duration and product pages).
- **Why not the others:** Gradient Boosting is practically tied (0.696 with a tuned threshold vs 0.702) but needs more tuning and the threshold adjustment; Logistic Regression is simpler and interpretable but about 0.02 lower in F1.
- **Threshold:** the nested CV shows that a custom threshold does *not* help the Random Forest (F1 0.693 with it, 0.695 without), because balanced class weights already handle the imbalance. We therefore use 0.5.

**Differences among the top models are small relative to the fold-to-fold standard deviation (0.01 to 0.02), so the ranking should not be over-interpreted.** The expected F1 on the hidden test set is about 0.69.

## How to reproduce

```
conda env create -f challenge2/environment.yml
conda activate cai4105-challenge2
python challenge2/models/random_forest.py        # or any script in challenge2/models/ (results/*.json)
python challenge2/nested_cv_final.py             # nested CV for the final model (~5 to 15 min)
jupyter nbconvert --to notebook --execute --inplace challenge2/challenge2_final_notebook.ipynb
```

The notebook reads the saved model results in `results/`, re-computes the final model's validation, and writes `challenge2/predictions.csv`. Set `RERUN_ALL_MODELS = True` in its second code cell to re-run all seven searches (about 30 to 40 minutes).

## Team contributions

- **Andres Salazar:** shared pipeline (`common.py`: preprocessing, feature engineering, validation, `run_experiment`), EDA, environment, Logistic Regression, Naive Bayes, Gradient Boosting, nested CV, final notebook, README, predictions.
- **Sarah Singh:** K-Nearest Neighbors and Support Vector Machine.
- **Lucas (GitHub: lucasg14fernandez):** Decision Tree and Random Forest (the final model), and the Random Forest feature-importance plot.

See the commit history and pull requests for the individual contributions.
