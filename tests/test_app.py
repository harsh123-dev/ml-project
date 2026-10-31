"""
Integration Tests for AIRWISE Streamlit Web Application
-------------------------------------------------------
Verifies that the main application and all 6 individual pages
load and execute with zero unhandled exceptions.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("page_file", [
    "app.py",
    "pages/1_Dashboard.py",
    "pages/2_AQI_Prediction.py",
    "pages/3_Model_Performance.py",
    "pages/4_SHAP_Explainability.py",
    "pages/5_Data_Explorer.py",
    "pages/6_About_Project.py"
])
def test_streamlit_pages_load_cleanly(page_file):
    """Verify that every Streamlit page executes without any runtime exceptions."""
    assert os.path.exists(page_file), f"Page file does not exist: {page_file}"
    at = AppTest.from_file(page_file, default_timeout=30).run()
    assert not at.exception, f"Page '{page_file}' raised runtime exception: {at.exception[0].value if at.exception else ''}"


def test_prediction_page_interactive_forecast():
    """Verify end-to-end interactive prediction form submission and SHAP visualization on Page 2."""
    at = AppTest.from_file("pages/2_AQI_Prediction.py", default_timeout=30).run()
    assert not at.exception

    # Click first preset (Latest Station Observation)
    if at.button:
        at.button[0].click().run()
        assert not at.exception

    # Submit the forecast form
    at.button[-1].click().run()
    assert not at.exception
    # Ensure health advisory or markdown output exists
    assert len(at.markdown) > 0


def test_dashboard_page_multiselect():
    """Verify changing pollutant multiselect works without error on Dashboard."""
    at = AppTest.from_file("pages/1_Dashboard.py", default_timeout=30).run()
    assert not at.exception
    if at.multiselect:
        at.multiselect[0].select("CO_lag1").run()
        assert not at.exception


def test_performance_page_metrics_switch():
    """Verify switching metric dropdown and inspecting models works on Model Performance."""
    at = AppTest.from_file("pages/3_Model_Performance.py", default_timeout=30).run()
    assert not at.exception
    if at.selectbox:
        at.selectbox[0].select("RMSE").run()
        assert not at.exception

