"""
Shared preprocessing for Challenge 1 (Rental Price Prediction).
Centralizes data loading and preprocessing so every model is trained/
evaluated on identical data and transformations, making
RMSE comparable across our four models.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

TARGET = 'price'

# 'ID' excluded (not a predictor). 'category' excluded (99.96% one value,
# no signal).
FEATURES = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'has_photo', 'state']
NUMERIC_FEATURES = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']
CATEGORICAL_FEATURES = ['has_photo', 'state']


def load_splits():
    """
    Loads the team's shared train/validation split.
    IMPORTANT: use this, don't create your own train_test_split() --
    everyone needs the same validation set for RMSE to be comparable.
    """
    train_split = pd.read_csv('data/train_split.csv')
    val_split = pd.read_csv('data/val_split.csv')
    return train_split, val_split


def load_test():
    """Official test set -- no price column, used only for final predictions.csv."""
    return pd.read_csv('data/challenge1_test.csv')


def build_preprocessor():
    """
    Fresh, unfitted pipeline. Fit only on train_split, then .transform()
    val_split/test. handle_unknown='ignore' is required: 'state' has
    categories that appear in train but not test (or vice versa).
    """
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])


def rmse(y_true, y_pred):
    """Root Mean Squared Error, in dollars -- lower is better."""
    return np.sqrt(mean_squared_error(y_true, y_pred))