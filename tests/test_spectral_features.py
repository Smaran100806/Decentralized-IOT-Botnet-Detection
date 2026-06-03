"""
tests/test_spectral_features.py  —  Unit tests for spectral_features.py
========================================================================
Tests: Laplacian construction, spectral decomposition, feature augmentation,
eigenvector properties, and pickle persistence.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.spectral_features import (
    build_adjacency_matrix,
    build_normalized_laplacian,
    build_unnormalized_laplacian,
    compute_spectral_decomposition,
    extract_spectral_features_for_partition,
    augment_dataframe_with_spectral_features,
)


# -- Helpers --------------------------------------------------------------------

FEATURES = [
    "Header_Length", "Duration", "syn_flag_number", "ack_flag_number",
    "syn_count", "urg_count", "rst_count", "HTTPS", "UDP", "ICMP",
    "Tot sum", "Min", "Max", "AVG", "Tot size", "Covariance", "Variance",
]


def make_graph_with_attrs(n: int = 6, seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    for i in range(n):
        fc = int(rng.integers(20, 100))
        ac = int(rng.integers(0, fc))
        G.add_node(i, flow_count=fc, attack_count=ac,
                   benign_count=fc - ac, attack_ratio=ac / fc)
    for i in range(n - 1):
        G.add_edge(i, i + 1, weight=float(rng.uniform(1, 5)))
    return G


def make_complete_weighted(n: int = 4) -> tuple[nx.Graph, np.ndarray]:
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    A = np.ones((n, n)) - np.eye(n)
    return G, A


def make_augmented_df(n_rows=60, k=4, ws=15, seed=42):
    """Create a DataFrame that already has node_id and window_id columns."""
    rng = np.random.default_rng(seed)
    data = {f: rng.standard_normal(n_rows).astype(np.float32) for f in FEATURES}
    data["label_binary"] = rng.integers(0, 2, n_rows)
    data["node_id"] = rng.integers(0, k, n_rows)
    data["window_id"] = np.arange(n_rows) // ws
    return pd.DataFrame(data)


# -- build_adjacency_matrix (spectral_features version) ------------------------

class TestBuildAdjacencyMatrixSF:
    def test_shape_and_symmetry(self):
        G = make_graph_with_attrs(5)
        nodes = list(G.nodes())
        A = build_adjacency_matrix(G, nodes)
        assert A.shape == (5, 5)
        np.testing.assert_array_almost_equal(A, A.T)

    def test_non_negative(self):
        G = make_graph_with_attrs(5)
        A = build_adjacency_matrix(G, list(G.nodes()))
        assert (A >= 0).all()


# -- build_normalized_laplacian ------------------------------------------------

class TestNormalizedLaplacian:
    def test_shape(self):
        _, A = make_complete_weighted(4)
        L = build_normalized_laplacian(A)
        assert L.shape == (4, 4)

    def test_symmetric(self):
        _, A = make_complete_weighted(4)
        L = build_normalized_laplacian(A)
        np.testing.assert_array_almost_equal(L, L.T)

    def test_eigenvalues_in_0_2(self):
        """Normalised Laplacian eigenvalues are in [0, 2]."""
        _, A = make_complete_weighted(5)
        L = build_normalized_laplacian(A)
        eigenvalues = np.linalg.eigvalsh(L)
        assert (eigenvalues >= -1e-9).all(), "Eigenvalues must be >= 0"
        assert (eigenvalues <= 2.0 + 1e-9).all(), "Normalised Laplacian eigenvalues <= 2"

    def test_smallest_eigenvalue_near_zero(self):
        """Connected graph → λ₀ ≈ 0."""
        _, A = make_complete_weighted(4)
        L = build_normalized_laplacian(A)
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        assert eigenvalues[0] == pytest.approx(0.0, abs=1e-8)

    def test_isolated_node_handled(self):
        """Zero-degree node (row of zeros in A) must not cause division errors."""
        A = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]])
        L = build_normalized_laplacian(A)
        assert np.isfinite(L).all()

    def test_identity_minus_normalized_A(self):
        """For a path graph, L = I - D^{-1/2} A D^{-1/2} must hold."""
        A = np.array([[0., 1., 0.],
                      [1., 0., 1.],
                      [0., 1., 0.]], dtype=float)
        degree = A.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        expected = np.eye(3) - D_inv_sqrt @ A @ D_inv_sqrt
        L = build_normalized_laplacian(A)
        np.testing.assert_array_almost_equal(L, expected)


# -- build_unnormalized_laplacian ----------------------------------------------

class TestUnnormalizedLaplacian:
    def test_shape(self):
        A = np.array([[0., 1.], [1., 0.]])
        L = build_unnormalized_laplacian(A)
        assert L.shape == (2, 2)

    def test_row_sums_zero(self):
        """Each row of the unnormalized Laplacian must sum to zero."""
        _, A = make_complete_weighted(4)
        L = build_unnormalized_laplacian(A)
        np.testing.assert_array_almost_equal(L.sum(axis=1), np.zeros(4))

    def test_positive_semidefinite(self):
        """L = D - A is PSD for undirected graphs."""
        _, A = make_complete_weighted(4)
        L = build_unnormalized_laplacian(A)
        eigenvalues = np.linalg.eigvalsh(L)
        assert (eigenvalues >= -1e-9).all()

    def test_formula(self):
        A = np.array([[0., 2., 0.],
                      [2., 0., 3.],
                      [0., 3., 0.]], dtype=float)
        D = np.diag(A.sum(axis=1))
        expected = D - A
        L = build_unnormalized_laplacian(A)
        np.testing.assert_array_almost_equal(L, expected)


# -- compute_spectral_decomposition --------------------------------------------

class TestComputeSpectralDecomposition:
    def _path_laplacian(self, n):
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
        return build_normalized_laplacian(A)

    def test_returns_k_eigenvalues(self):
        L = self._path_laplacian(6)
        vals, vecs = compute_spectral_decomposition(L, top_k=3)
        assert vals.shape == (3,)

    def test_returns_k_eigenvectors(self):
        L = self._path_laplacian(6)
        vals, vecs = compute_spectral_decomposition(L, top_k=3)
        assert vecs.shape == (6, 3)

    def test_eigenvalues_ascending(self):
        L = self._path_laplacian(6)
        vals, _ = compute_spectral_decomposition(L, top_k=4)
        assert (np.diff(vals) >= -1e-10).all()

    def test_top_k_larger_than_n_capped(self):
        """Requesting more eigenpairs than matrix size should not raise."""
        L = self._path_laplacian(3)
        vals, vecs = compute_spectral_decomposition(L, top_k=100)
        assert vals.shape[0] <= 3

    def test_eigenvectors_orthonormal(self):
        """Columns of eigenvector matrix should be orthonormal."""
        L = self._path_laplacian(6)
        _, vecs = compute_spectral_decomposition(L, top_k=4)
        dot = vecs.T @ vecs
        np.testing.assert_array_almost_equal(dot, np.eye(vecs.shape[1]), decimal=10)

    def test_real_eigenvalues(self):
        L = self._path_laplacian(5)
        vals, _ = compute_spectral_decomposition(L, top_k=3)
        assert np.isrealobj(vals)


# -- extract_spectral_features_for_partition -----------------------------------

class TestExtractSpectralFeaturesForPartition:
    def test_empty_node_list(self):
        G = make_graph_with_attrs(4)
        result = extract_spectral_features_for_partition(G, [], top_k=4)
        assert result["eigenvalues"].size == 0
        assert result["fiedler_value"] == pytest.approx(0.0)

    def test_returns_required_keys(self):
        G = make_graph_with_attrs(4)
        result = extract_spectral_features_for_partition(G, list(G.nodes()), top_k=3)
        for key in ["eigenvalues", "eigenvectors", "node_list", "fiedler_value"]:
            assert key in result

    def test_eigenvalue_count(self):
        G = make_graph_with_attrs(5)
        result = extract_spectral_features_for_partition(G, list(G.nodes()), top_k=3)
        assert len(result["eigenvalues"]) == 3

    def test_fiedler_value_non_negative(self):
        G = make_graph_with_attrs(5)
        result = extract_spectral_features_for_partition(G, list(G.nodes()), top_k=4)
        assert result["fiedler_value"] >= 0.0

    def test_single_node(self):
        G = make_graph_with_attrs(1)
        result = extract_spectral_features_for_partition(G, [0], top_k=4)
        assert result["fiedler_value"] == pytest.approx(0.0)


# -- augment_dataframe_with_spectral_features ----------------------------------

class TestAugmentDataframe:
    def _setup(self, k=4):
        G = make_graph_with_attrs(k)
        df = make_augmented_df(n_rows=80, k=k)
        # Simple partition: even → 0, odd → 1
        partition_map = {i: i % 2 for i in range(k)}
        return G, df, partition_map

    def test_new_columns_added(self):
        G, df, pm = self._setup(k=4)
        aug, _ = augment_dataframe_with_spectral_features(df, G, pm, top_k=3)
        for col in ["partition_id", "fiedler_value",
                    "spectral_eigen_0", "spectral_proj_0"]:
            assert col in aug.columns

    def test_row_count_unchanged(self):
        G, df, pm = self._setup(k=4)
        aug, _ = augment_dataframe_with_spectral_features(df, G, pm, top_k=3)
        assert len(aug) == len(df)

    def test_partition_id_assigned(self):
        G, df, pm = self._setup(k=4)
        aug, _ = augment_dataframe_with_spectral_features(df, G, pm, top_k=3)
        valid_parts = set(pm.values()) | {-1}
        assert aug["partition_id"].isin(valid_parts).all()

    def test_fiedler_value_non_negative(self):
        G, df, pm = self._setup(k=4)
        aug, _ = augment_dataframe_with_spectral_features(df, G, pm, top_k=3)
        assert (aug["fiedler_value"] >= 0.0).all()

    def test_spectral_store_returned(self):
        G, df, pm = self._setup(k=4)
        _, store = augment_dataframe_with_spectral_features(df, G, pm, top_k=3)
        assert isinstance(store, dict)
        for part_id in set(pm.values()):
            assert part_id in store

    def test_spectral_store_picklable(self, tmp_path):
        G, df, pm = self._setup(k=4)
        _, store = augment_dataframe_with_spectral_features(df, G, pm, top_k=3)
        path = tmp_path / "spectral_features.pkl"
        with path.open("wb") as fh:
            pickle.dump(store, fh)
        with path.open("rb") as fh:
            store2 = pickle.load(fh)
        assert set(store2.keys()) == set(store.keys())

    def test_unnormalized_laplacian_option(self):
        G, df, pm = self._setup(k=4)
        aug, _ = augment_dataframe_with_spectral_features(
            df, G, pm, top_k=3, normalize=False
        )
        assert "spectral_eigen_0" in aug.columns

    def test_no_nan_in_spectral_columns(self):
        G, df, pm = self._setup(k=4)
        aug, _ = augment_dataframe_with_spectral_features(df, G, pm, top_k=3)
        spec_cols = [c for c in aug.columns if c.startswith("spectral_") or c == "fiedler_value"]
        assert not aug[spec_cols].isna().any().any()
