"""
graph_builder.py  --  Task C1 + D2  (Team Member 2)
====================================================
Build a temporal device-flow graph from the pre-processed CICIoT23 CSVs
and optionally augment each flow row with spectral Laplacian features.

Graph Construction Strategy
----------------------------
The pre-processed CSVs do **not** contain raw IP addresses (dropped during
feature selection). We therefore derive *virtual device nodes* using
MiniBatchKMeans clustering over the 17 selected flow-level features.
Each cluster represents a distinct IoT behavioural profile
(e.g. a camera flooding UDP, a sensor generating ICMP keep-alives, etc.).

Temporal Windowing
------------------
Flows are grouped into non-overlapping windows of W consecutive rows,
simulating a sliding-time-window over the capture stream.  Within each
window an undirected edge is added between every pair of node-clusters
whose representative flows co-occur; edge weight tracks co-occurrence
count across all windows.

Complexity
----------
  Clustering   : O(N * K * I)   -- N rows, K clusters, I iterations
  Graph build  : O(N)           -- single pass to accumulate edges
  Total        : O(N * K * I + E)  where E <= K^2

Outputs
-------
- ``networkx.Graph``  returned / saved to ``<output_dir>/graph.gpickle``
- Augmented DataFrame with columns:
    - ``node_id``    : cluster assignment (int)
    - ``window_id``  : temporal window index (int)
  Saved to ``<output_dir>/graph_augmented_{split}.csv``

Usage (CLI)
-----------
    python graph_builder.py                        # uses validation split
    python graph_builder.py --split train --k 30
    python graph_builder.py --split validation --window-size 500 --k 20
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# -- Paths ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "processed_ciciot23"
OUTPUT_DIR = ROOT / "graph_artifacts"

SELECTED_FEATURES_PATH = PROCESSED_DIR / "selected_features.json"


# -- Feature loading ------------------------------------------------------------

def load_selected_features() -> list[str]:
    """Load the canonical feature list produced by the preprocessing pipeline.

    Using this list guarantees consistent feature ordering across all
    downstream scripts (see ``selected_features.json``).
    """
    with SELECTED_FEATURES_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["selected_features"]


# -- Virtual device node assignment --------------------------------------------

def assign_device_nodes(
    df: pd.DataFrame,
    features: list[str],
    k: int = 20,
    random_state: int = 42,
    batch_size: int = 10_000,
) -> tuple[pd.DataFrame, MiniBatchKMeans]:
    """Cluster flows into K virtual device nodes via MiniBatchKMeans.

    MiniBatchKMeans is used instead of standard KMeans because the dataset
    can contain millions of rows.  Time complexity is O(N · K · I) with a
    small constant factor due to mini-batching.

    Args:
        df:           Input DataFrame (pre-scaled features).
        features:     Column names to use for clustering.
        k:            Number of virtual device nodes (clusters).
        random_state: Reproducibility seed.
        batch_size:   Mini-batch size for KMeans.

    Returns:
        Augmented DataFrame with ``node_id`` column, fitted KMeans model.
    """
    X = df[features].to_numpy(dtype=np.float32)

    # MiniBatchKMeans — O(N·K·I), memory-safe for large datasets
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(batch_size, len(X)),
        n_init=3,
        max_iter=100,
    )
    km.fit(X)
    labels = km.predict(X)  # O(N·K) prediction pass

    result = df.copy()
    result["node_id"] = labels
    return result, km


# -- Temporal windowing --------------------------------------------------------

def assign_windows(df: pd.DataFrame, window_size: int = 500) -> pd.DataFrame:
    """Assign each flow row to a non-overlapping temporal window.

    Row order serves as a proxy for time (flows are stored chronologically
    in the CICIOT23 dataset).  Window ID is computed in O(N) via integer
    division.

    Args:
        df:          DataFrame with ``node_id`` column.
        window_size: Number of consecutive rows per window.

    Returns:
        DataFrame with added ``window_id`` column.
    """
    result = df.copy()
    # O(N) assignment — vectorised integer division
    result["window_id"] = np.arange(len(result)) // window_size
    return result


# -- Graph construction --------------------------------------------------------

def build_graph(
    df: pd.DataFrame,
    k: int,
    node_label_col: str = "label_binary",
) -> nx.Graph:
    """Build a weighted, undirected device-flow graph.

    Nodes
    -----
    Each of the K virtual device clusters is a node.  Node attributes:
      - flow_count    : total flows assigned to this node
      - attack_count  : flows labelled as attack (label_binary == 1)
      - benign_count  : flows labelled as benign
      - attack_ratio  : fraction of malicious flows

    Edges
    -----
    An edge (u, v) is added when nodes u and v co-occur in the same
    temporal window (i.e. they exchange traffic in that window).
    Edge weight = number of windows where both appeared.

    Complexity: O(W * n_w^2) where W = number of windows,
    n_w = average distinct nodes per window.  In practice n_w << K.

    Args:
        df:             DataFrame with node_id, window_id, and a
                        binary label column.
        k:              Total number of node clusters.
        node_label_col: Column used to compute per-node attack ratio.

    Returns:
        Populated networkx.Graph.
    """
    G = nx.Graph()

    # -- Add nodes --------------------------------------------------------------
    for node_id in range(k):
        subset = df[df["node_id"] == node_id]
        if len(subset) == 0:
            G.add_node(node_id, flow_count=0, attack_count=0,
                       benign_count=0, attack_ratio=0.0)
            continue

        has_label = node_label_col in subset.columns
        attack_count = int(subset[node_label_col].sum()) if has_label else 0
        total_count = len(subset)
        benign_count = total_count - attack_count

        G.add_node(
            node_id,
            flow_count=total_count,
            attack_count=attack_count,
            benign_count=benign_count,
            attack_ratio=attack_count / total_count if total_count > 0 else 0.0,
        )

    # -- Add edges via co-occurrence in windows ---------------------------------
    # Vectorised groupby approach: O(N) groupby + O(W · n²_w) edge accumulation
    edge_weights: dict[tuple[int, int], int] = {}

    for _window_id, window_df in df.groupby("window_id"):
        nodes_in_window = window_df["node_id"].unique()
        if len(nodes_in_window) < 2:
            continue
        # All pairs in the window share an edge
        sorted_nodes = sorted(nodes_in_window)
        for i, u in enumerate(sorted_nodes):
            for v in sorted_nodes[i + 1 :]:
                key = (int(u), int(v))
                edge_weights[key] = edge_weights.get(key, 0) + 1

    for (u, v), weight in edge_weights.items():
        G.add_edge(u, v, weight=weight)

    return G


# -- Persistence helpers --------------------------------------------------------

def save_graph(G: nx.Graph, path: Path) -> None:
    """Serialise graph to a pickle file (cross-version portable)."""
    with path.open("wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Graph saved → {path}")


def load_graph(path: Path) -> nx.Graph:
    """Load a previously serialised graph."""
    with path.open("rb") as fh:
        return pickle.load(fh)


def save_kmeans(km: MiniBatchKMeans, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(km, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  KMeans model saved → {path}")


def load_kmeans(path: Path) -> MiniBatchKMeans:
    with path.open("rb") as fh:
        return pickle.load(fh)


# -- Summary helpers ------------------------------------------------------------

def graph_summary(G: nx.Graph) -> dict:
    """Return a dictionary of key graph statistics."""
    degrees = [d for _, d in G.degree()]
    # nx.is_connected raises NetworkXPointlessConcept for the null graph
    if G.number_of_nodes() == 0:
        is_conn = False
        n_components = 0
    else:
        is_conn = nx.is_connected(G)
        n_components = nx.number_connected_components(G)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "is_connected": is_conn,
        "num_connected_components": n_components,
    }


# -- Main pipeline -------------------------------------------------------------

def build_graph_pipeline(
    split: str = "validation",
    k: int = 20,
    window_size: int = 500,
    max_rows: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> tuple[nx.Graph, pd.DataFrame, MiniBatchKMeans]:
    """End-to-end graph construction pipeline.

    Args:
        split:       One of ``"train"``, ``"validation"``, ``"test"``.
        k:           Number of virtual device nodes.
        window_size: Rows per temporal window.
        max_rows:    Row cap for quick testing (``None`` = full split).
        output_dir:  Where to write artefacts.  Defaults to ``graph_artifacts/``.

    Returns:
        (G, augmented_df, kmeans_model)
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED_DIR / f"{split}_clean.csv"
    print(f"[graph_builder] Loading '{split}' split from {csv_path} …")
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    features = load_selected_features()
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # -- C1: Cluster flows into virtual device nodes ----------------------------
    print(f"  Assigning {k} virtual device nodes via MiniBatchKMeans …")
    df, km = assign_device_nodes(df, features, k=k)

    # -- C1: Assign temporal windows --------------------------------------------
    print(f"  Assigning temporal windows (window_size={window_size}) …")
    df = assign_windows(df, window_size=window_size)

    num_windows = df["window_id"].nunique()
    print(f"  Windows created: {num_windows:,}")

    # -- C1: Build graph --------------------------------------------------------
    print("  Building device-flow graph …")
    label_col = "label_binary" if "label_binary" in df.columns else None
    G = build_graph(df, k=k, node_label_col=label_col or "node_id")

    summary = graph_summary(G)
    print(
        f"  Graph built — Nodes: {summary['nodes']}, "
        f"Edges: {summary['edges']}, "
        f"Density: {summary['density']:.4f}, "
        f"Connected: {summary['is_connected']}"
    )

    # -- Save artefacts ---------------------------------------------------------
    save_graph(G, out / f"graph_{split}.gpickle")
    save_kmeans(km, out / "kmeans_model.pkl")

    df.to_csv(out / f"graph_augmented_{split}.csv", index=False)
    print(f"  Augmented CSV saved → {out / f'graph_augmented_{split}.csv'}")

    with (out / f"graph_summary_{split}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("[graph_builder] Done.")
    return G, df, km


# -- CLI entry-point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a temporal device-flow graph from processed CICIoT23 data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split to process (default: validation).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of virtual device node clusters (default: 20).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=500,
        help="Rows per temporal window (default: 500).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Row cap for quick testing (default: full split).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artefacts (default: graph_artifacts/).",
    )
    args = parser.parse_args()

    build_graph_pipeline(
        split=args.split,
        k=args.k,
        window_size=args.window_size,
        max_rows=args.max_rows,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
