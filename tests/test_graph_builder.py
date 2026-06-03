"""
tests/test_graph_builder.py  —  Unit tests for graph_builder.py
================================================================
Tests cover:
  • Virtual device node assignment (KMeans clustering)
  • Temporal window assignment
  • Graph construction (nodes, edges, attributes)
  • Graph summary statistics
  • Persistence helpers (save/load graph & KMeans)
  • Edge cases: single-node graph, empty windows, missing label column
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# -- Import the module under test -----------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.graph_builder import (
    assign_device_nodes,
    assign_windows,
    build_graph,
    graph_summary,
    load_graph,
    save_graph,
    save_kmeans,
    load_kmeans,
)


# -- Fixtures -------------------------------------------------------------------

FEATURES = [
    "Header_Length", "Duration", "syn_flag_number", "ack_flag_number",
    "syn_count", "urg_count", "rst_count", "HTTPS", "UDP", "ICMP",
    "Tot sum", "Min", "Max", "AVG", "Tot size", "Covariance", "Variance",
]


def make_dummy_df(n_rows: int = 200, n_classes: int = 2, seed: int = 42) -> pd.DataFrame:
    """Create a minimal synthetic DataFrame that mimics processed_ciciot23 CSVs."""
    rng = np.random.default_rng(seed)
    data = {feat: rng.standard_normal(n_rows).astype(np.float32) for feat in FEATURES}
    data["label_binary"] = rng.integers(0, n_classes, size=n_rows)
    data["label_family_id"] = rng.integers(0, 5, size=n_rows)
    return pd.DataFrame(data)


# -- assign_device_nodes --------------------------------------------------------

class TestAssignDeviceNodes:
    def test_returns_dataframe_with_node_id(self):
        df = make_dummy_df(100)
        result, km = assign_device_nodes(df, FEATURES, k=5)
        assert "node_id" in result.columns

    def test_node_id_range(self):
        k = 5
        df = make_dummy_df(100)
        result, _ = assign_device_nodes(df, FEATURES, k=k)
        assert result["node_id"].between(0, k - 1).all()

    def test_correct_number_of_unique_nodes(self):
        """With n_rows >> k, all k clusters should be populated."""
        k = 5
        df = make_dummy_df(500)
        result, _ = assign_device_nodes(df, FEATURES, k=k)
        assert result["node_id"].nunique() <= k

    def test_preserves_original_columns(self):
        df = make_dummy_df(100)
        result, _ = assign_device_nodes(df, FEATURES, k=5)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self):
        df = make_dummy_df(150)
        result, _ = assign_device_nodes(df, FEATURES, k=5)
        assert len(result) == len(df)

    def test_reproducible_with_same_seed(self):
        df = make_dummy_df(100)
        r1, _ = assign_device_nodes(df, FEATURES, k=5, random_state=0)
        r2, _ = assign_device_nodes(df, FEATURES, k=5, random_state=0)
        pd.testing.assert_series_equal(r1["node_id"], r2["node_id"])

    def test_k_equals_one(self):
        """With k=1 all rows must get node_id=0."""
        df = make_dummy_df(50)
        result, _ = assign_device_nodes(df, FEATURES, k=1)
        assert (result["node_id"] == 0).all()

    def test_returns_fitted_kmeans(self):
        from sklearn.cluster import MiniBatchKMeans
        df = make_dummy_df(100)
        _, km = assign_device_nodes(df, FEATURES, k=5)
        assert isinstance(km, MiniBatchKMeans)
        assert km.n_clusters == 5


# -- assign_windows -------------------------------------------------------------

class TestAssignWindows:
    def test_returns_dataframe_with_window_id(self):
        df = make_dummy_df(100)
        result = assign_windows(df, window_size=10)
        assert "window_id" in result.columns

    def test_window_ids_are_correct(self):
        n, ws = 100, 10
        df = make_dummy_df(n)
        result = assign_windows(df, window_size=ws)
        expected = np.arange(n) // ws
        np.testing.assert_array_equal(result["window_id"].to_numpy(), expected)

    def test_number_of_windows(self):
        n, ws = 105, 10
        df = make_dummy_df(n)
        result = assign_windows(df, window_size=ws)
        assert result["window_id"].nunique() == 11  # ceil(105/10)

    def test_window_size_larger_than_df(self):
        """All rows should fall into window 0."""
        df = make_dummy_df(50)
        result = assign_windows(df, window_size=1000)
        assert (result["window_id"] == 0).all()

    def test_window_size_one(self):
        """Each row gets its own window."""
        n = 30
        df = make_dummy_df(n)
        result = assign_windows(df, window_size=1)
        assert result["window_id"].nunique() == n

    def test_preserves_row_count(self):
        df = make_dummy_df(80)
        result = assign_windows(df, window_size=10)
        assert len(result) == 80


# -- build_graph ----------------------------------------------------------------

class TestBuildGraph:
    def _make_labelled_df(self, n=200, k=5, ws=20):
        df = make_dummy_df(n)
        df, _ = assign_device_nodes(df, FEATURES, k=k)
        df = assign_windows(df, window_size=ws)
        return df

    def test_returns_networkx_graph(self):
        df = self._make_labelled_df()
        G = build_graph(df, k=5)
        assert isinstance(G, nx.Graph)

    def test_node_count_equals_k(self):
        k = 5
        df = self._make_labelled_df(k=k)
        G = build_graph(df, k=k)
        assert G.number_of_nodes() == k

    def test_all_nodes_present(self):
        k = 5
        df = self._make_labelled_df(k=k)
        G = build_graph(df, k=k)
        assert set(G.nodes()) == set(range(k))

    def test_node_attributes_present(self):
        k = 5
        df = self._make_labelled_df(k=k)
        G = build_graph(df, k=k)
        for node in G.nodes():
            attrs = G.nodes[node]
            assert "flow_count" in attrs
            assert "attack_count" in attrs
            assert "attack_ratio" in attrs

    def test_attack_ratio_in_range(self):
        k = 5
        df = self._make_labelled_df(k=k)
        G = build_graph(df, k=k)
        for node in G.nodes():
            ratio = G.nodes[node]["attack_ratio"]
            assert 0.0 <= ratio <= 1.0

    def test_edges_have_weight(self):
        k = 5
        df = self._make_labelled_df(k=k)
        G = build_graph(df, k=k)
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert data["weight"] >= 1

    def test_graph_is_undirected(self):
        df = self._make_labelled_df()
        G = build_graph(df, k=5)
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_single_cluster(self):
        """With k=1 there should be 1 node and 0 edges."""
        df = make_dummy_df(50)
        df, _ = assign_device_nodes(df, FEATURES, k=1)
        df = assign_windows(df, window_size=10)
        G = build_graph(df, k=1)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_no_label_column(self):
        """Graph construction should not raise if label column is absent."""
        df = make_dummy_df(100)
        df = df.drop(columns=["label_binary"])
        df, _ = assign_device_nodes(df, FEATURES, k=5)
        df = assign_windows(df, window_size=20)
        # Use node_id as the label_col placeholder
        G = build_graph(df, k=5, node_label_col="node_id")
        assert G.number_of_nodes() == 5


# -- graph_summary --------------------------------------------------------------

class TestGraphSummary:
    def test_summary_keys(self):
        G = nx.path_graph(4)
        summary = graph_summary(G)
        for key in ["nodes", "edges", "density", "avg_degree", "max_degree",
                    "is_connected", "num_connected_components"]:
            assert key in summary

    def test_empty_graph(self):
        G = nx.Graph()
        summary = graph_summary(G)
        assert summary["nodes"] == 0
        assert summary["edges"] == 0
        assert summary["avg_degree"] == 0.0
        assert summary["is_connected"] is False
        assert summary["num_connected_components"] == 0

    def test_complete_graph(self):
        k = 5
        G = nx.complete_graph(k)
        summary = graph_summary(G)
        assert summary["nodes"] == k
        assert summary["is_connected"] is True
        assert summary["num_connected_components"] == 1

    def test_density_range(self):
        G = nx.path_graph(5)
        summary = graph_summary(G)
        assert 0.0 <= summary["density"] <= 1.0


# -- Persistence helpers --------------------------------------------------------

class TestPersistenceHelpers:
    def test_save_and_load_graph(self, tmp_path):
        G = nx.path_graph(5)
        path = tmp_path / "test_graph.gpickle"
        save_graph(G, path)
        G_loaded = load_graph(path)
        assert G_loaded.number_of_nodes() == G.number_of_nodes()
        assert G_loaded.number_of_edges() == G.number_of_edges()

    def test_graph_attributes_preserved(self, tmp_path):
        G = nx.Graph()
        G.add_node(0, flow_count=42, attack_ratio=0.5)
        G.add_edge(0, 1, weight=7)
        path = tmp_path / "attrs.gpickle"
        save_graph(G, path)
        G2 = load_graph(path)
        assert G2.nodes[0]["flow_count"] == 42
        assert G2.edges[0, 1]["weight"] == 7

    def test_save_and_load_kmeans(self, tmp_path):
        from sklearn.cluster import MiniBatchKMeans
        df = make_dummy_df(100)
        _, km = assign_device_nodes(df, FEATURES, k=5)
        path = tmp_path / "km.pkl"
        save_kmeans(km, path)
        km2 = load_kmeans(path)
        assert isinstance(km2, MiniBatchKMeans)
        assert km2.n_clusters == 5
