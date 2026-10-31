"""
Integration Test for Exploratory Data Analysis Notebook
-------------------------------------------------------
Verifies that all 14 code cells in `notebooks/exploratory_data_analysis.ipynb`
execute sequentially without any runtime exceptions or missing dependencies.
"""

import os
import json
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_eda_notebook_exists():
    """Verify notebook file presence and structure."""
    nb_path = os.path.join("notebooks", "exploratory_data_analysis.ipynb")
    assert os.path.exists(nb_path), f"Notebook missing at: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 20
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 14


def test_eda_notebook_full_execution():
    """Execute every code cell in the notebook sequentially."""
    nb_path = os.path.join("notebooks", "exploratory_data_analysis.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Disable blocking show
    plt.show = lambda *args, **kwargs: None

    ctx = {}
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    for idx, cell in enumerate(code_cells):
        code_str = "".join(cell["source"])
        try:
            exec(code_str, ctx)
        except Exception as exc:
            pytest.fail(f"Notebook code cell {idx + 1} raised {type(exc).__name__}: {exc}\nCode snippet:\n{code_str[:200]}")

    assert "df_raw" in ctx
    assert len(ctx["df_raw"]) == 35064
    assert "AQI_EPA" in ctx["df_raw"].columns
