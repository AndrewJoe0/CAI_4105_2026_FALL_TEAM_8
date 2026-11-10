# Challenge 2: team plan

Goal: predict `Revenue` (purchase or not). Primary metric: **F1 of the positive class**.
Deadline: **Nov 15**. Everyone also submits their own peer-review form.

## Rules we agreed to follow (so every model is comparable)
1. All models use `common.py`: same preprocessing, same 5-fold stratified CV, same metrics.
   **Do not write your own split or scoring.**
2. Model scripts live in `models/` and are run with `python models/<name>.py`. Each saves a
   `results/<name>.json`, so nobody edits the same file (no merge conflicts).
3. Use the test set only in the very last step (`make_predictions.py`).
4. Commit often, with a meaningful message ("Add SVM grid search"), and use a branch + pull request
   so reviews show up in GitHub.

## Who does what

| Person | Models (script in `models/`) | Also owns |
|---|---|---|
| **Andres** | Logistic Regression, Naive Bayes (done) | `common.py` (shared code), `eda.py`, `environment.yml`, README draft, final notebook assembly |
| **Teammate 2** | K-Nearest Neighbors, Support Vector Machine | Compare distance/kernel based models; a short note on why SVM is slow and how we kept it manageable |
| **Teammate 3** | Decision Tree, Random Forest, Gradient Boosting | Feature importance plot for the best tree model; try one extra model if time allows (e.g. HistGradientBoosting) |

### Steps for Teammates 2 and 3
1. `git pull`, create the environment (`conda env create -f environment.yml`).
2. Copy `models/_template.py` to `models/<your_model>.py`, fill in the 4 marked lines.
3. Run it. Look at the printed baseline vs. tuned F1 and the train F1 (large gap = overfitting).
4. **If the best parameter is at the edge of your grid, widen the grid and rerun.**
5. Commit the script **and** the `results/*.json`, open a pull request.
6. Write 3 to 4 sentences for the README about your models (what worked, overfitting, run time).

## Shared steps at the end (everyone, ~Nov 13 to 14)
- Pick the final model (top F1 in the results table, then weigh overfitting, complexity, run time, interpretability).
- Estimate its performance without bias with `nested_cv_f1` in `common.py` (slow, final model only).
- Choose the decision threshold from the out-of-fold scores (it improved F1 for every model we tried).
- Train on all training data, predict the test set, write `predictions.csv` (`ID`, `Revenue` as True/False).
- Final notebook + README + each person submits the peer-review form.
