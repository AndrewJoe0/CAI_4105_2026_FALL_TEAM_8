
# Creates the final Jupyter notebook
# from the complete Python solution.

import json
from pathlib import Path

source_file = Path("final_solution.py")
notebook_file = Path("final_challenge.ipynb")

assert source_file.exists(), "Missing final_solution.py"

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CAP 4105 - Challenge 1\n",
                "## Team 8: Apartment Rental Price Prediction\n",
                "\n",
                "This notebook reproduces the complete "
                "machine learning workflow.\n",
                "\n",
                "Models: Linear Regression, Decision Tree, "
                "Random Forest, and Gradient Boosting.\n",
                "\n",
                "Evaluation: shared validation split, "
                "RMSE, and Grid Search cross-validation.\n",
                "\n",
                "The final selected model is retrained "
                "on all labeled training data and used "
                "to create predictions.csv.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%run final_solution.py"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

notebook_file.write_text(
    json.dumps(notebook, indent=2),
    encoding="utf-8"
)

print("Created:", notebook_file)
