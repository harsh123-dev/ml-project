"""
AIRWISE LightGBM Model Wrapper
------------------------------
Standard Scikit-learn compliant estimator wrapper around LightGBM C++ Booster
supporting cross-platform pickle/joblib serialization and isolated native C++ runtime.
"""

import lightgbm as lgb
import ctypes
import numpy as np
from typing import Optional, Dict, Any

# Windows x64 ctypes ABI fix: explicitly declare 64-bit argument signatures
# to prevent stack corruption on 64-bit Windows architectures
try:
    lgb.basic._LIB.LGBM_BoosterPredictForMat.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
    ]
    lgb.basic._LIB.LGBM_DatasetCreateFromMat.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lgb.basic._LIB.LGBM_DatasetSetField.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
except Exception:
    pass


class LightGBMBoosterWrapper:
    """
    Robust serializable LightGBM wrapper storing model definition
    as an internal string for deterministic unpickling across processes.
    """
    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.08,
        num_leaves: int = 31,
        max_depth: int = 6,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.random_state = random_state
        self.booster = None

    def fit(self, X, y):
        X_arr = np.array(X, dtype=np.float32, order="C", copy=True)
        y_arr = np.array(y, dtype=np.float32, order="C", copy=True)
        ds = lgb.Dataset(X_arr, label=y_arr, free_raw_data=False)
        params = {
            "objective": "regression",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "verbosity": -1,
            "random_state": self.random_state,
            "force_col_wise": True
        }
        self.booster = lgb.train(params, ds, num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        X_arr = np.array(X, dtype=np.float32, order="C", copy=True)
        return self.booster.predict(X_arr)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.booster is not None:
            state["booster_str"] = self.booster.model_to_string()
        else:
            state["booster_str"] = None
        state["booster"] = None
        return state

    def __setstate__(self, state):
        booster_str = state.pop("booster_str", None)
        self.__dict__.update(state)
        if booster_str:
            self.booster = lgb.Booster(model_str=booster_str)
        else:
            self.booster = None
