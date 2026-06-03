"""
tests/test_section_e.py
========================
Unit tests for Section E modules:
  - matrix_features.py  (E1)
  - train_matrix_rf.py  (E2)
  - fl_client.py        (E3)
  - fl_server.py        (E3)
  - secure_aggregation.py (E4)

Run with:
    pytest tests/test_section_e.py -v
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.federated.matrix_features import (
    MATRIX_FEATURE_NAMES,
    augment_with_matrix_features,
    extract_matrix_features_row,
)
from src.federated.secure_aggregation import SecureAggregator, _generate_mask
from src.federated.fl_client import FederatedClient, load_client_data
from src.federated.fl_server import rf_to_bytes, bytes_to_rf, aggregate_forests


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def synthetic_flow_df():
    """Synthetic DataFrame that mirrors CICIoT23 feature columns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "AVG":          np.random.uniform(100, 1500, n),
        "Std":          np.random.uniform(0, 500, n),
        "Min":          np.random.uniform(0, 100, n),
        "Max":          np.random.uniform(1000, 2000, n),
        "IAT":          np.random.uniform(0, 1, n),
        "Rate":         np.random.uniform(1, 1000, n),
        "Srate":        np.random.uniform(0.5, 500, n),
        "Drate":        np.random.uniform(0.5, 500, n),
        "Tot_sum":      np.random.uniform(1000, 50000, n),
        "Duration":     np.random.uniform(0.001, 10, n),
        "label_binary": np.random.randint(0, 2, n),
    })
    return df


@pytest.fixture(scope="module")
def fitted_rf():
    """Simple fitted RandomForestClassifier on synthetic data."""
    X, y = make_classification(n_samples=300, n_features=10,
                                n_informative=5, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf, X, y


# ===========================================================================
# E1 — matrix_features.py
# ===========================================================================

class TestMatrixFeatureNames:
    def test_names_list_is_nonempty(self):
        assert len(MATRIX_FEATURE_NAMES) > 0

    def test_expected_count(self):
        # 8 packet + 8 iat + 1 cross = 17
        assert len(MATRIX_FEATURE_NAMES) == 17

    def test_all_mat_prefix(self):
        for name in MATRIX_FEATURE_NAMES:
            assert name.startswith("mat_"), f"Unexpected name: {name}"

    def test_contains_key_features(self):
        for feat in ("mat_pkt_mean", "mat_pkt_entropy", "mat_iat_std",
                     "mat_cross_correlation"):
            assert feat in MATRIX_FEATURE_NAMES


class TestExtractMatrixFeaturesRow:
    def test_returns_dict(self, synthetic_flow_df):
        row = synthetic_flow_df.iloc[0]
        result = extract_matrix_features_row(row)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self, synthetic_flow_df):
        row = synthetic_flow_df.iloc[0]
        result = extract_matrix_features_row(row)
        for key in MATRIX_FEATURE_NAMES:
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_finite_floats(self, synthetic_flow_df):
        row = synthetic_flow_df.iloc[0]
        result = extract_matrix_features_row(row)
        for k, v in result.items():
            assert np.isfinite(v), f"Non-finite value for {k}: {v}"

    def test_zero_row_does_not_crash(self):
        zero_row = pd.Series({
            "AVG": 0.0, "Std": 0.0, "Min": 0.0, "Max": 0.0,
            "IAT": 0.0, "Rate": 0.0, "Srate": 0.0, "Drate": 0.0,
        })
        result = extract_matrix_features_row(zero_row)
        assert isinstance(result, dict)
        assert len(result) == len(MATRIX_FEATURE_NAMES)


class TestAugmentWithMatrixFeatures:
    def test_adds_correct_number_of_columns(self, synthetic_flow_df):
        original_cols = len(synthetic_flow_df.columns)
        aug = augment_with_matrix_features(synthetic_flow_df)
        assert len(aug.columns) == original_cols + len(MATRIX_FEATURE_NAMES)

    def test_original_columns_preserved(self, synthetic_flow_df):
        aug = augment_with_matrix_features(synthetic_flow_df)
        for col in synthetic_flow_df.columns:
            assert col in aug.columns

    def test_no_nan_in_matrix_columns(self, synthetic_flow_df):
        aug = augment_with_matrix_features(synthetic_flow_df)
        for col in MATRIX_FEATURE_NAMES:
            assert not aug[col].isnull().any(), f"NaN found in column {col}"

    def test_row_count_unchanged(self, synthetic_flow_df):
        aug = augment_with_matrix_features(synthetic_flow_df)
        assert len(aug) == len(synthetic_flow_df)

    def test_pkt_mean_equals_avg(self, synthetic_flow_df):
        """mat_pkt_mean should equal the AVG column (same statistic)."""
        aug = augment_with_matrix_features(synthetic_flow_df)
        assert np.allclose(aug["mat_pkt_mean"].values,
                           aug["AVG"].values, atol=1e-6)

    def test_pkt_std_equals_std(self, synthetic_flow_df):
        """mat_pkt_std should equal the Std column."""
        aug = augment_with_matrix_features(synthetic_flow_df)
        assert np.allclose(aug["mat_pkt_std"].values,
                           aug["Std"].values, atol=1e-6)


# ===========================================================================
# E4 — secure_aggregation.py
# ===========================================================================

class TestGenerateMask:
    def test_correct_length(self):
        for length in (0, 1, 16, 100, 10000):
            mask = _generate_mask(length, b"test-seed")
            assert len(mask) == length

    def test_deterministic(self):
        m1 = _generate_mask(64, b"seed")
        m2 = _generate_mask(64, b"seed")
        assert np.array_equal(m1, m2)

    def test_different_seeds_differ(self):
        m1 = _generate_mask(64, b"seed-a")
        m2 = _generate_mask(64, b"seed-b")
        assert not np.array_equal(m1, m2)


class TestSecureAggregatorDisabled:
    def test_mask_is_identity_when_disabled(self):
        agg = SecureAggregator(enabled=False)
        data = b"hello world"
        assert agg.mask(data) == data

    def test_unmask_is_identity_when_disabled(self):
        agg = SecureAggregator(enabled=False)
        data = b"hello world"
        assert agg.unmask(data) == data


class TestSecureAggregatorEnabled:
    @pytest.fixture
    def agg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.federated.secure_aggregation.SecureAggregator.SEED_FILE",
            tmp_path / "seed.bin"
        )
        return SecureAggregator(enabled=True, shared_seed=b"test-shared-seed-32bytes!!!!!!!!")

    def test_masking_changes_data(self, agg):
        original = b"sensitive model bytes " * 10
        masked = agg.mask(original)
        assert masked != original

    def test_round_trip_bytes(self, agg):
        original = b"model update data " * 100
        assert agg.unmask(agg.mask(original)) == original

    def test_round_trip_empty(self, agg):
        assert agg.unmask(agg.mask(b"")) == b""

    def test_round_trip_array(self, agg):
        arr = np.random.rand(50).astype(np.float32)
        recovered = agg.unmask_array(agg.mask_array(arr))
        assert np.allclose(arr, recovered, atol=1e-6)

    def test_mask_length_preserved(self, agg):
        data = os.urandom(1024)
        assert len(agg.mask(data)) == len(data)


# ===========================================================================
# E3 — fl_server.py (serialisation & aggregation)
# ===========================================================================

class TestRFSerialisation:
    def test_rf_to_bytes_and_back(self, fitted_rf):
        clf, X, y = fitted_rf
        data = rf_to_bytes(clf)
        assert isinstance(data, bytes)
        assert len(data) > 0
        recovered = bytes_to_rf(data)
        assert recovered.n_features_in_ == clf.n_features_in_
        assert recovered.n_classes_ == clf.n_classes_

    def test_predictions_match_after_serialisation(self, fitted_rf):
        clf, X, y = fitted_rf
        data = rf_to_bytes(clf)
        recovered = bytes_to_rf(data)
        # Predictions should be identical since trees are preserved
        preds_orig = clf.predict(X[:20])
        preds_recv = recovered.predict(X[:20])
        assert np.array_equal(preds_orig, preds_recv)


class TestAggregateForests:
    def test_merged_forest_has_all_trees(self):
        X, y = make_classification(n_samples=200, n_features=10, random_state=0)
        clf1 = RandomForestClassifier(n_estimators=5, random_state=0)
        clf2 = RandomForestClassifier(n_estimators=7, random_state=1)
        clf1.fit(X[:100], y[:100])
        clf2.fit(X[100:], y[100:])
        merged = aggregate_forests([(clf1, 100), (clf2, 100)])
        assert merged.n_estimators == 12

    def test_merged_forest_can_predict(self):
        X, y = make_classification(n_samples=200, n_features=10, random_state=0)
        clf1 = RandomForestClassifier(n_estimators=5, random_state=0)
        clf2 = RandomForestClassifier(n_estimators=5, random_state=1)
        clf1.fit(X[:100], y[:100])
        clf2.fit(X[100:], y[100:])
        merged = aggregate_forests([(clf1, 100), (clf2, 100)])
        preds = merged.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_single_client_aggregation(self):
        X, y = make_classification(n_samples=100, n_features=8, random_state=7)
        clf = RandomForestClassifier(n_estimators=5, random_state=7)
        clf.fit(X, y)
        merged = aggregate_forests([(clf, 100)])
        assert merged.n_estimators == 5


# ===========================================================================
# E3 — FederatedClient (unit tests without real data files)
# ===========================================================================

class TestFederatedClientPredict:
    """Test FederatedClient.predict_single() independently of data files."""

    def test_predict_single_binary_output(self, fitted_rf):
        clf, X, y = fitted_rf
        client = FederatedClient(client_id=0)
        pred = client.predict_single(X[0].astype(np.float32), clf)
        assert pred in (0, 1)

    def test_predict_single_feature_truncation(self, fitted_rf):
        """Client should handle fewer features than model expects."""
        clf, X, y = fitted_rf
        client = FederatedClient(client_id=0)
        # Provide more features than model expects
        extra = np.hstack([X[0], np.zeros(5)]).astype(np.float32)
        pred = client.predict_single(extra, clf)
        assert pred in (0, 1)

    def test_predict_single_feature_padding(self, fitted_rf):
        """Client should handle more features than model expects (padding)."""
        clf, X, y = fitted_rf
        client = FederatedClient(client_id=0)
        # Provide fewer features — client should zero-pad
        fewer = X[0, :5].astype(np.float32)
        pred = client.predict_single(fewer, clf)
        assert pred in (0, 1)

    def test_evaluate_alignment(self, fitted_rf):
        """evaluate() should work even when feature counts differ."""
        clf, X, y = fitted_rf
        client = FederatedClient(client_id=0)
        client.X = X.astype(np.float32)
        client.y = y.astype(np.int32)
        client.feature_names = [f"f{i}" for i in range(X.shape[1])]
        metrics = client.evaluate(clf)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_train_produces_valid_rf(self, fitted_rf):
        """train() should return a fitted RF and valid metrics."""
        clf_global, X, y = fitted_rf
        client = FederatedClient(client_id=0, n_estimators=5)
        client.X = X.astype(np.float32)
        client.y = y.astype(np.int32)
        client.feature_names = [f"f{i}" for i in range(X.shape[1])]
        local_clf, n_samples, metrics = client.train(clf_global)
        assert isinstance(local_clf, RandomForestClassifier)
        assert n_samples == len(y)
        assert 0.0 <= metrics["f1"] <= 1.0
        assert metrics["n_samples"] == n_samples
