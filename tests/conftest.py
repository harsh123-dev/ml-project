"""
Pytest configuration for AIRWISE test suite.
Ensures LightGBM C++ library runtime is initialized before any other numerical library.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Initialize LightGBM first
import src.lgb_wrapper
