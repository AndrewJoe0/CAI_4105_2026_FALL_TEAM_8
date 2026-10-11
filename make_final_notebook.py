# Creates the final Jupyter notebook from the complete Python solution.
#
# final_solution.py is the single source of truth. Each numbered
# section comment ("# 1. ...", "# 2. ...") becomes a markdown heading
# followed by a code cell, so the notebook shows the full workflow
# from loading the data to generating predictions.csv.

import json
import re
from pathlib import Path

source_file = Path("final_solution.py")
notebook_file = Path("final_challenge.ipynb")

assert source_file.exists(), "Missing final_solution.py"

lines = source_file.read_text(encoding="utf-8").splitlines()
section_pattern = re.compile(r"^# (\d+)\. (.+)$")

section_starts = [
    i for i, line in enumerate(lines) if section_pattern.match(line)
]
assert section_starts, "No numbered sections found in final_solution.py"


def to_source(code_lines):
    """Trim blank lines at both ends and format for a notebook cell."""
    code_lines = list(code_lines)
    while code_lines and not code_lines[0].strip():
        code_lines.pop(0)
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()
    return [line + "\n" for line in code_lines[:-1]] + [code_lines[-1]]


def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(code_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(code_lines),
    }


cells = [
    markdown_cell(
        "# CAI 4105 - Challenge 1\n"
        "## Team 8: Apartment Rental Price Prediction\n"
        "\n"
        "This notebook shows the complete machine learning workflow, "
        "from loading the data to generating `predictions.csv`.\n"
        "\n"
        "- **Models:** Linear Regression, Decision Tree, Random Forest, "
        "and Gradient Boosting\n"
        "- **Validation:** shared train/validation split, RMSE, and "
        "Grid Search with cross-validation\n"
        "- **Final model:** the model with the lowest validation RMSE, "
        "retrained on all labeled training data\n"
        "\n"
        "Preprocessing is shared through `preprocessing.py` so every "
        "model uses identical data and transformations.\n"
    ),
    # Show plots inside the notebook when it is executed.
    code_cell(["%matplotlib inline"]),
    markdown_cell("## Setup: imports and helper function\n"),
    code_cell(lines[: section_starts[0]]),
]

for index, start in enumerate(section_starts):
    end = (
        section_starts[index + 1]
        if index + 1 < len(section_starts)
        else len(lines)
    )
    number, title = section_pattern.match(lines[start]).groups()
    cells.append(markdown_cell(f"## {number}. {title}\n"))
    cells.append(code_cell(lines[start + 1 : end]))

for number, cell in enumerate(cells, start=1):
    cell["id"] = f"cell-{number:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebook_file.write_text(json.dumps(notebook, indent=2), encoding="utf-8")

print("Created:", notebook_file, "with", len(cells), "cells")
