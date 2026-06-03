"""
tests/test_graph_partition.py  —  Unit tests for graph_partition.py
====================================================================
Tests: adjacency matrix, spectral partitioning, partition summary, persistence.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.graph_partition import (
    partition_spectral,
    compute_partition_summary,
    load_graph,
)
from src.graph.spectral_features import build_adjacency_matrix


# -- Helpers --------------------------------------------------------------------

def make_weighted_graph(n_nodes: int = 8, seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        fc = int(rng.integers(10, 100))
        ac = int(rng.integers(0, fc))
        G.nodes[i]["flow_count"] = fc
        G.nodes[i]["attack_count"] = ac
        G.nodes[i]["benign_count"] = fc - ac
        G.nodes[i]["attack_ratio"] = ac / fc
    for i in range(n_nodes - 1):
        G.add_edge(i, i + 1, weight=float(rng.integers(1, 10)))
    if n_nodes > 2:
        G.add_edge(0, n_nodes - 1, weight=float(rng.integers(1, 10)))
    return G


def make_two_clique_graph() -> nx.Graph:
    """Two dense cliques joined by a weak bridge — ideal test for 2-partition."""
    G = nx.Graph()
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j, weight=10.0)
    for i in range(4, 8):
        for j in range(i + 1, 8):
            G.add_edge(i, j, weight=10.0)
    G.add_edge(3, 4, weight=1.0)
    for node in G.nodes():
        G.nodes[node].update({"flow_count": 50, "attack_count": 5,
                               "benign_count": 45, "attack_ratio": 0.1})
    return G


# -- build_adjacency_matrix -----------------------------------------------------

class TestBuildAdjacencyMatrix:
    def test_shape(self):
        G = make_weighted_graph(6)
        A = build_adjacency_matrix(G, list(G.nodes()))
        assert A.shape == (6, 6)

    def test_symmetric(self):
        G = make_weighted_graph(6)
        A = build_adjacency_matrix(G, list(G.nodes()))
        np.testing.assert_array_almost_equal(A, A.T)

    def test_zero_diagonal(self):
        G = make_weighted_graph(5)
        A = build_adjacency_matrix(G, list(G.nodes()))
        np.testing.assert_array_equal(np.diag(A), np.zeros(5))

    def test_weights_transferred(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=7.5)
        A = build_adjacency_matrix(G, [0, 1])
        assert A[0, 1] == pytest.approx(7.5)
        assert A[1, 0] == pytest.approx(7.5)

    def test_default_weight_is_one(self):
        G = nx.Graph()
        G.add_edge(0, 1)  # no weight attribute
        A = build_adjacency_matrix(G, [0, 1])
        assert A[0, 1] == pytest.approx(1.0)

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        A = build_adjacency_matrix(G, [0])
        assert A.shape == (1, 1)
        assert A[0, 0] == pytest.approx(0.0)


# -- partition_spectral ---------------------------------------------------------

class TestPartitionSpectral:
    def test_returns_dict(self):
        G = make_weighted_graph(8)
        result = partition_spectral(G, n_partitions=2)
        assert isinstance(result, dict)

    def test_all_nodes_assigned(self):
        G = make_weighted_graph(8)
        result = partition_spectral(G, n_partitions=2)
        assert set(result.keys()) == set(G.nodes())

    def test_partition_ids_in_range(self):
        n_parts = 3
        G = make_weighted_graph(9)
        result = partition_spectral(G, n_partitions=n_parts)
        for pid in result.values():
            assert 0 <= pid < n_parts

    def test_reproducible(self):
        G = make_weighted_graph(8)
        r1 = partition_spectral(G, n_partitions=2, random_state=42)
        r2 = partition_spectral(G, n_partitions=2, random_state=42)
        assert r1 == r2

    def test_n_partitions_exceeds_nodes_warns(self):
        G = make_weighted_graph(3)
        with pytest.warns(UserWarning):
            result = partition_spectral(G, n_partitions=10)
        assert len(result) == 3

    def test_two_clique_structure(self):
        """Spectral clustering should keep each clique together."""
        G = make_two_clique_graph()
        result = partition_spectral(G, n_partitions=2, random_state=42)
        clique_a_parts = {result[n] for n in range(4)}
        clique_b_parts = {result[n] for n in range(4, 8)}
        assert len(clique_a_parts) == 1
        assert len(clique_b_parts) == 1
        assert clique_a_parts != clique_b_parts


# -- compute_partition_summary --------------------------------------------------

class TestComputePartitionSummary:
    def _round_robin_map(self, G, n=2):
        return {node: node % n for node in G.nodes()}

    def test_required_keys(self):
        G = make_weighted_graph(6)
        summary = compute_partition_summary(G, self._round_robin_map(G))
        for key in ["n_partitions", "cross_partition_edges", "per_partition"]:
            assert key in summary

    def test_n_partitions(self):
        G = make_weighted_graph(6)
        pm = self._round_robin_map(G, 3)
        summary = compute_partition_summary(G, pm)
        assert summary["n_partitions"] == 3

    def test_attack_ratio_in_range(self):
        G = make_weighted_graph(8)
        summary = compute_partition_summary(G, self._round_robin_map(G))
        for stats in summary["per_partition"].values():
            assert 0.0 <= stats["attack_ratio"] <= 1.0

    def test_cross_edges_non_negative(self):
        G = make_weighted_graph(6)
        summary = compute_partition_summary(G, self._round_robin_map(G))
        assert summary["cross_partition_edges"] >= 0

    def test_all_same_partition_zero_cross_edges(self):
        G = make_weighted_graph(5)
        pm = {n: 0 for n in G.nodes()}
        summary = compute_partition_summary(G, pm)
        assert summary["cross_partition_edges"] == 0


# -- load_graph -----------------------------------------------------------------

class TestLoadGraph:
    def test_roundtrip(self, tmp_path):
        G = make_weighted_graph(5)
        path = tmp_path / "g.gpickle"
        with path.open("wb") as fh:
            pickle.dump(G, fh)
        G2 = load_graph(path)
        assert G2.number_of_nodes() == G.number_of_nodes()
        assert G2.number_of_edges() == G.number_of_edges()
