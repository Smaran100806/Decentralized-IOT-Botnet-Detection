"""
graph_partition.py  --  Task C2  (Team Member 2)
================================================
Partition the device-flow graph into K sub-graphs, simulating independent
edge clusters (e.g. separate network segments, rooms, or physical zones).

Partitioning Strategy
---------------------
We use **Spectral Clustering** (scikit-learn) as the primary algorithm
because:
  1. It works on Windows without native-library compilation (unlike pymetis).
  2. It is well-suited for graphs with community structure.
  3. It operates on the graph's Laplacian, making it mathematically
     consistent with the spectral analysis in spectral_features.py.

A METIS-based fallback path is provided for Linux/macOS environments where
pymetis is installed.

Complexity
----------
  Adjacency matrix build : O(V^2)  -- V = number of nodes
  Eigen-decomposition    : O(V^3)  -- for full spectral clustering
  K-Means on embeddings  : O(V * K * I)
  Total                  : O(V^3)  -- dominated by eigen-decomp; acceptable
                                     because V = K_clusters <= 50 (small graph)

Outputs
-------
- partition_map.json   -- {str(node_id): partition_id} mapping
- partition_summary.json -- per-partition statistics

Usage (CLI)
-----------
    python graph_partition.py                    # 4 partitions, validation split
    python graph_partition.py --n-partitions 6 --split train
    python graph_partition.py --n-partitions 4 --use-metis   # Linux/macOS only
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

# -- Paths ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = ROOT / "graph_artifacts"
OUTPUT_DIR = ROOT / "graph_artifacts"


# -- Graph loading -------------------------------------------------------------

def load_graph(path: Path) -> nx.Graph:
    """Load a serialised networkx graph."""
    with path.open("rb") as fh:
        return pickle.load(fh)


# -- Partition algorithms -------------------------------------------------------

def partition_spectral(
    G: nx.Graph,
    n_partitions: int,
    random_state: int = 42,
) -> dict[int, int]:
    """Partition graph using Spectral Clustering on the adjacency matrix.

    Constructs a weighted adjacency matrix (V×V), then applies sklearn's
    SpectralClustering which internally computes the graph Laplacian and
    extracts its leading eigenvectors.

    Args:
        G:             NetworkX graph (nodes must be integers 0..V-1).
        n_partitions:  Number of output clusters.
        random_state:  Seed for reproducibility.

    Returns:
        ``{node_id: partition_id}`` mapping.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)

    if n < n_partitions:
        warnings.warn(
            f"Graph has only {n} nodes but {n_partitions} partitions requested. "
            "Reducing n_partitions to match node count.",
            stacklevel=2,
        )
        n_partitions = n

    # Build weighted adjacency matrix — O(V²) allocation + O(E) fill
    A = np.zeros((n, n), dtype=np.float64)
    node_index = {node: i for i, node in enumerate(nodes)}

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        i, j = node_index[u], node_index[v]
        A[i, j] = weight
        A[j, i] = weight  # undirected

    # Spectral Clustering — internally computes normalised Laplacian
    # Complexity: O(V³) eigen-decomp + O(V·K·I) K-Means on embeddings
    sc = SpectralClustering(
        n_clusters=n_partitions,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
        n_init=10,
    )
    labels = sc.fit_predict(A)

    return {nodes[i]: int(labels[i]) for i in range(n)}


def partition_metis(G: nx.Graph, n_partitions: int) -> dict[int, int]:
    """Partition graph using METIS via the ``pymetis`` binding.

    METIS provides near-linear time partitioning: O(V log V).  It is
    preferred on Linux/macOS for large graphs.  Requires ``pip install pymetis``
    and a working METIS native library.

    Args:
        G:             NetworkX graph.
        n_partitions:  Number of output partitions.

    Returns:
        ``{node_id: partition_id}`` mapping.

    Raises:
        ImportError: if ``pymetis`` is not installed.
    """
    try:
        import pymetis  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pymetis is not installed. Run: pip install pymetis\n"
            "On Windows, use the spectral fallback: omit --use-metis"
        ) from exc

    nodes = sorted(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}

    # Build adjacency list required by pymetis
    adjacency: list[list[int]] = [[] for _ in range(len(nodes))]
    for u, v in G.edges():
        adjacency[node_index[u]].append(node_index[v])
        adjacency[node_index[v]].append(node_index[u])

    _, membership = pymetis.part_graph(n_partitions, adjacency=adjacency)
    return {nodes[i]: int(membership[i]) for i in range(len(nodes))}


# -- Partition statistics -------------------------------------------------------

def compute_partition_summary(
    G: nx.Graph,
    partition_map: dict[int, int],
) -> dict[str, object]:
    """Compute per-partition and overall statistics.

    Args:
        G:              The full graph.
        partition_map:  ``{node_id: partition_id}`` mapping.

    Returns:
        Dictionary with per-partition node lists, edge counts, and
        cross-partition edge counts.
    """
    partitions: dict[int, list[int]] = {}
    for node, part in partition_map.items():
        partitions.setdefault(part, []).append(node)

    per_partition = {}
    for part_id, node_list in sorted(partitions.items()):
        subgraph = G.subgraph(node_list)
        node_data = {
            n: G.nodes[n] for n in node_list
        }
        total_flows = sum(
            d.get("flow_count", 0) for d in node_data.values()
        )
        attack_flows = sum(
            d.get("attack_count", 0) for d in node_data.values()
        )
        per_partition[str(part_id)] = {
            "node_count": len(node_list),
            "nodes": node_list,
            "internal_edges": subgraph.number_of_edges(),
            "total_flows": total_flows,
            "attack_flows": attack_flows,
            "attack_ratio": attack_flows / total_flows if total_flows > 0 else 0.0,
        }

    # Cross-partition edges
    cross_edges = sum(
        1 for u, v in G.edges()
        if partition_map.get(u) != partition_map.get(v)
    )

    return {
        "n_partitions": len(partitions),
        "cross_partition_edges": cross_edges,
        "per_partition": per_partition,
    }


# -- Main pipeline -------------------------------------------------------------

def partition_graph_pipeline(
    split: str = "validation",
    n_partitions: int = 4,
    use_metis: bool = False,
    random_state: int = 42,
    graph_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> tuple[dict[int, int], dict[str, object]]:
    """Full partitioning pipeline.

    Args:
        split:         Dataset split (used to find the graph file).
        n_partitions:  Number of partitions to create.
        use_metis:     Use METIS instead of spectral clustering.
        random_state:  Seed for reproducibility.
        graph_dir:     Directory containing the graph pickle.
        output_dir:    Where to write partition artefacts.

    Returns:
        (partition_map, summary)
    """
    g_dir = graph_dir or GRAPH_DIR
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    graph_path = g_dir / f"graph_{split}.gpickle"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_path}\n"
            f"Run graph_builder.py --split {split} first."
        )

    print(f"[graph_partition] Loading graph from {graph_path} …")
    G = load_graph(graph_path)
    print(
        f"  Graph loaded — {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges."
    )

    # -- Partition --------------------------------------------------------------
    algorithm = "METIS" if use_metis else "Spectral Clustering"
    print(f"  Partitioning into {n_partitions} clusters using {algorithm} …")

    if use_metis:
        partition_map = partition_metis(G, n_partitions)
    else:
        partition_map = partition_spectral(G, n_partitions, random_state)

    # -- Summary ----------------------------------------------------------------
    summary = compute_partition_summary(G, partition_map)
    print(f"  Partitioning complete.")
    for part_id, stats in summary["per_partition"].items():
        print(
            f"    Partition {part_id}: {stats['node_count']} nodes, "
            f"{stats['internal_edges']} internal edges, "
            f"{stats['total_flows']:,} flows, "
            f"attack_ratio={stats['attack_ratio']:.3f}"
        )
    print(f"  Cross-partition edges: {summary['cross_partition_edges']}")

    # -- Save artefacts ---------------------------------------------------------
    # Convert keys to strings for JSON serialisation
    partition_map_str = {str(k): v for k, v in partition_map.items()}

    partition_map_path = out / "partition_map.json"
    with partition_map_path.open("w", encoding="utf-8") as fh:
        json.dump(partition_map_str, fh, indent=2)
    print(f"  Partition map saved → {partition_map_path}")

    summary_path = out / "partition_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Partition summary saved → {summary_path}")

    print("[graph_partition] Done.")
    return partition_map, summary


# -- CLI entry-point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition the device-flow graph into K edge clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split whose graph to partition (default: validation).",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=4,
        help="Number of partitions / edge clusters (default: 4).",
    )
    parser.add_argument(
        "--use-metis",
        action="store_true",
        help="Use METIS partitioning (Linux/macOS, requires pymetis).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for spectral clustering (default: 42).",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Directory containing the graph pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for partition artefacts.",
    )
    args = parser.parse_args()

    partition_graph_pipeline(
        split=args.split,
        n_partitions=args.n_partitions,
        use_metis=args.use_metis,
        random_state=args.random_state,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
