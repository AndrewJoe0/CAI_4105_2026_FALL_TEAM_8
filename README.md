# CAI 4105 - Machine Learning Challenge 1
## Team 8 - Apartment Rental Price Prediction

### Project Objective

The objective of this project is to predict monthly
apartment rental prices based on property characteristics
and geographic location.

The target variable is `price`.

### Dataset

Training data: 7,398 observations.

Test data: 2,467 observations.

The ID column is used to identify observations but
is excluded from model training.

### Data Preparation

The project uses a shared preprocessing module.

Numerical features:
- bathrooms
- bedrooms
- square_feet
- latitude
- longitude

Categorical features:
- has_photo
- state

Numerical variables are standardized using StandardScaler.

Categorical variables are encoded using OneHotEncoder
with handle_unknown="ignore".

Preprocessing is fitted on training data to avoid
validation data leakage.

### Validation Strategy

The team uses a consistent training/validation split
for comparing regression models.

Root Mean Squared Error (RMSE) is the primary metric.

Lower RMSE indicates better predictive performance.

Grid Search uses cross-validation within the
training partition.

Training and validation RMSE are compared to assess
possible overfitting.

### Regression Models

Four required models are evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

### Hyperparameter Tuning

GridSearchCV is applied to Random Forest and
Gradient Boosting.

Parameters investigated include:

- Number of estimators
- Maximum tree depth
- Minimum leaf size
- Learning rate (Gradient Boosting)

The searches use RMSE-based scoring.

### Model Comparison

All baseline and tuned models are evaluated
using the shared validation set.

The results are exported to:

`model_comparison.csv`

The final model is selected based on the lowest
validation RMSE among evaluated candidates.

The selected model and its measured RMSE are printed
when the final notebook is executed.

### Final Predictions

After model selection, the selected model is
retrained on all labeled training observations.

The final model predicts prices for the 2,467
observations in the test dataset.

Predictions are exported to:

`predictions.csv`

Required columns:
- ID
- price

### Reproducibility

The project uses Python 3.11 and the packages
listed in environment.yml.

The final notebook is:

`final_challenge.ipynb`

The notebook runs the complete modeling workflow
through `final_solution.py`.

To reproduce locally:

1. Create the Conda environment from environment.yml.
2. Activate the environment.
3. Open final_challenge.ipynb in Jupyter.
4. Run the notebook from top to bottom.

The GitHub Actions workflow can also execute the
notebook and generate the prediction files.

### Project Files

- preprocessing.py: Shared preprocessing functions
- 01_data_prep.ipynb: Data preparation
- 02_linear_regression.ipynb: Linear Regression
- 03_decision_tree.ipynb: Decision Tree
- 04_random_forest.ipynb: Random Forest
- 05_gradient_boosting.py: Gradient Boosting analysis
- final_solution.py: Complete modeling workflow
- make_final_notebook.py: Notebook generator
- final_challenge.ipynb: Final notebook
- model_comparison.csv: Model performance results
- predictions.csv: Final competition predictions
- environment.yml: Conda environment

### Team Collaboration

Team members contributed to data preparation,
model development, evaluation, and documentation.

Individual contributions can be reviewed through
GitHub commits and pull requests.
