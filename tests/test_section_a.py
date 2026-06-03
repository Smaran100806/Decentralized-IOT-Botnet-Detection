"""
test_section_a.py
=================
Unit tests for Section A modules:
  - utils_hyperopt.py   (get_param_grids, run_hyperopt)
  - train_binary_baseline.py  (evaluate helper)
  - train_family_baseline.py  (evaluate helper)

Run with:
    pytest tests/test_section_a.py -v
"""

import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_hyperopt import get_param_grids, run_hyperopt


# ═══════════════════════════════════════════════════════════════════════════════
# utils_hyperopt — get_param_grids()
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetParamGrids:
    """Tests for the get_param_grids() helper."""

    def test_returns_dict(self):
        assert isinstance(get_param_grids(), dict)

    def test_has_all_model_keys(self):
        grids = get_param_grids()
        for key in ("lr", "rf", "xgb", "lgbm"):
            assert key in grids, f"Missing model key: {key}"

    def test_lr_grid_has_required_keys(self):
        lr = get_param_grids()["lr"]
        for key in ("C", "max_iter", "solver", "class_weight"):
            assert key in lr, f"LR grid missing key: {key}"

    def test_rf_grid_has_required_keys(self):
        rf = get_param_grids()["rf"]
        for key in ("n_estimators", "max_depth", "min_samples_split",
                    "min_samples_leaf", "max_features", "class_weight"):
            assert key in rf, f"RF grid missing key: {key}"

    def test_xgb_grid_has_required_keys(self):
        xgb = get_param_grids()["xgb"]
        for key in ("n_estimators", "max_depth", "learning_rate",
                    "subsample", "colsample_bytree"):
            assert key in xgb, f"XGB grid missing key: {key}"

    def test_lgbm_grid_has_required_keys(self):
        lgbm = get_param_grids()["lgbm"]
        for key in ("n_estimators", "num_leaves", "learning_rate",
                    "subsample", "colsample_bytree", "class_weight"):
            assert key in lgbm, f"LGBM grid missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# utils_hyperopt — run_hyperopt()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return X, y


class TestRunHyperopt:
    """Tests for the run_hyperopt() wrapper."""

    def test_returns_three_values(self, binary_data):
        X, y = binary_data
        best_model, best_params, cv_results = run_hyperopt(
            RandomForestClassifier(random_state=42),
            X, y,
            param_grid={"n_estimators": [5, 10], "max_depth": [2, 3]},
            n_iter=2, cv=2, scoring="f1", verbose=0,
        )
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert hasattr(cv_results, "shape")  # pandas DataFrame

    def test_best_model_is_fitted(self, binary_data):
        X, y = binary_data
        best_model, _, _ = run_hyperopt(
            LogisticRegression(max_iter=200, random_state=42),
            X, y,
            param_grid={"C": [0.1, 1.0]},
            n_iter=2, cv=2, scoring="f1", verbose=0,
        )
        preds = best_model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_best_params_keys_in_grid(self, binary_data):
        X, y = binary_data
        grid = {"n_estimators": [5, 10], "max_depth": [2, 3]}
        _, best_params, _ = run_hyperopt(
            RandomForestClassifier(random_state=42),
            X, y,
            param_grid=grid,
            n_iter=2, cv=2, scoring="f1", verbose=0,
        )
        for key in best_params:
            assert key in grid, f"Unexpected key in best_params: {key}"

    def test_cv_results_sorted_by_rank(self, binary_data):
        X, y = binary_data
        _, _, cv_results = run_hyperopt(
            RandomForestClassifier(random_state=42),
            X, y,
            param_grid={"n_estimators": [5, 10], "max_depth": [2, 3]},
            n_iter=2, cv=2, scoring="f1", verbose=0,
        )
        ranks = cv_results["rank_test_score"].tolist()
        assert ranks == sorted(ranks), "cv_results not sorted by rank_test_score"

    def test_custom_scoring_macro_f1(self, binary_data):
        """run_hyperopt should accept 'f1_macro' scoring without error."""
        X, y = binary_data
        best_model, _, _ = run_hyperopt(
            RandomForestClassifier(random_state=42),
            X, y,
            param_grid={"n_estimators": [5, 10]},
            n_iter=2, cv=2, scoring="f1_macro", verbose=0,
        )
        assert best_model is not None


# ═══════════════════════════════════════════════════════════════════════════════
# train_binary_baseline — evaluate()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def fitted_binary_rf():
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=5,
        n_redundant=2, random_state=0,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X[:300], y[:300])
    return model, X[300:], y[300:]


class TestBinaryEvaluate:
    """Tests for train_binary_baseline.evaluate()."""

    def test_returns_dict(self, fitted_binary_rf):
        from train_binary_baseline import evaluate
        model, X_val, y_val = fitted_binary_rf
        result = evaluate("RF", model, X_val, y_val)
        assert isinstance(result, dict)

    def test_has_required_keys(self, fitted_binary_rf):
        from train_binary_baseline import evaluate
        model, X_val, y_val = fitted_binary_rf
        result = evaluate("RF", model, X_val, y_val)
        for key in ("model", "accuracy", "precision", "recall", "f1"):
            assert key in result, f"Missing key: {key}"

    def test_metrics_in_valid_range(self, fitted_binary_rf):
        from train_binary_baseline import evaluate
        model, X_val, y_val = fitted_binary_rf
        result = evaluate("RF", model, X_val, y_val)
        for metric in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= result[metric] <= 1.0, (
                f"{metric} out of [0,1]: {result[metric]}"
            )

    def test_model_name_stored_correctly(self, fitted_binary_rf):
        from train_binary_baseline import evaluate
        model, X_val, y_val = fitted_binary_rf
        result = evaluate("MY_MODEL", model, X_val, y_val)
        assert result["model"] == "MY_MODEL"

    def test_predictions_match_label_set(self, fitted_binary_rf):
        from train_binary_baseline import evaluate
        model, X_val, y_val = fitted_binary_rf
        evaluate("RF", model, X_val, y_val)   # should not raise
        preds = model.predict(X_val)
        assert set(preds).issubset({0, 1})


# ═══════════════════════════════════════════════════════════════════════════════
# train_family_baseline — evaluate()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def fitted_family_rf():
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=6,
        n_redundant=2, n_classes=4, n_clusters_per_class=1,
        random_state=7,
    )
    target_names = ["Benign", "DDoS", "DoS", "Mirai"]
    model = RandomForestClassifier(n_estimators=10, random_state=7)
    model.fit(X[:480], y[:480])
    return model, X[480:], y[480:], target_names


class TestFamilyEvaluate:
    """Tests for train_family_baseline.evaluate()."""

    def test_returns_dict(self, fitted_family_rf):
        from train_family_baseline import evaluate
        model, X_val, y_val, names = fitted_family_rf
        result = evaluate("RF", model, X_val, y_val, names)
        assert isinstance(result, dict)

    def test_has_required_keys(self, fitted_family_rf):
        from train_family_baseline import evaluate
        model, X_val, y_val, names = fitted_family_rf
        result = evaluate("RF", model, X_val, y_val, names)
        for key in ("model", "accuracy", "macro_f1", "weighted_f1"):
            assert key in result, f"Missing key: {key}"

    def test_metrics_in_valid_range(self, fitted_family_rf):
        from train_family_baseline import evaluate
        model, X_val, y_val, names = fitted_family_rf
        result = evaluate("RF", model, X_val, y_val, names)
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            assert 0.0 <= result[metric] <= 1.0, (
                f"{metric} out of [0,1]: {result[metric]}"
            )

    def test_model_name_stored_correctly(self, fitted_family_rf):
        from train_family_baseline import evaluate
        model, X_val, y_val, names = fitted_family_rf
        result = evaluate("LGBM", model, X_val, y_val, names)
        assert result["model"] == "LGBM"

    def test_macro_f1_leq_weighted_f1_on_balanced(self, fitted_family_rf):
        """On reasonably balanced data macro and weighted F1 should be close."""
        from train_family_baseline import evaluate
        model, X_val, y_val, names = fitted_family_rf
        result = evaluate("RF", model, X_val, y_val, names)
        # They should be within 0.3 of each other on synthetic balanced data
        assert abs(result["macro_f1"] - result["weighted_f1"]) < 0.30
