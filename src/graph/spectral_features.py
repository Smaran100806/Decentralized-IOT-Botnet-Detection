"""
spectral_features.py  --  Tasks D1 + D2  (Team Member 2)
=========================================================
Compute Graph Laplacian spectral features for each graph partition and
augment the per-flow feature vectors with the resulting eigenvalue/
eighenvector projections.

Mathematical Background (D1)
-----------------------------
For a graph partition with adjacency matrix A and degree matrix D:

    Laplacian          L  = D - A
    Normalized Lapl.   Ln = D^{-1/2} L D^{-1/2}  =  I - D^{-1/2} A D^{-1/2}

Spectral decomposition: Ln = U Lambda U^T  (eigen-decomp via numpy.linalg.eigh,
which exploits symmetry for O(V^3/3) speed instead of O(V^3)).

The k smallest non-trivial eigenvalues (excluding lambda_0 ~= 0) and their
corresponding eigenvectors carry the graph's spectral fingerprint:
  - Small eigenvalues -> tightly connected, hard-to-cut subgraphs.
  - Large eigenvalues -> sparse, easy-to-separate regions.
  - The Fiedler value (lambda_1) quantifies algebraic connectivity.

Feature Augmentation (D2)
--------------------------
Each flow row is enriched with the following spectral columns:
  - spectral_eigen_{i}   : the i-th smallest non-trivial eigenvalue
                           of its partition's normalised Laplacian.
  - spectral_proj_{i}    : projection of the flow's node onto the
                           i-th eigenvector (spectral coordinate).
  - fiedler_value        : lambda_1 of the partition (algebraic connectivity).
  - partition_id         : which partition this flow belongs to.

This augmented feature set is then fed to the Random Forest in
train_spectral_rf.py (Task D3).

Complexity
----------
  Per partition (V_p nodes):
    Laplacian build   : O(V_p^2)
    Eigen-decomp      : O(V_p^3)   -- acceptable because V_p <= K <= 50
  Overall             : O(P * V_p^3 + N)  where P = partitions, N = flow rows

Outputs
-------
- spectral_features.pkl   -- dict mapping partition_id to eigenvalues/eigenvectors
- Augmented CSV with spectral columns appended

Usage (CLI)
-----------
    python spectral_features.py                        # validation split, k=8
    python spectral_features.py --split train --top-k 5
    python spectral_features.py --split validation --top-k 8 --normalize
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
import pandas as pd

# -- Paths ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = ROOT / "graph_artifacts"
OUTPUT_DIR = ROOT / "graph_artifacts"


# -- Graph loading -------------------------------------------------------------

def load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_partition_map(path: Path) -> dict[int, int]:
    """Load partition map, converting string keys back to ints."""
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): int(v) for k, v in raw.items()}


# -- Laplacian construction ----------------------------------------------------

def build_adjacency_matrix(
    G: nx.Graph,
    node_list: list[int],
) -> np.ndarray:
    """Build a weighted adjacency matrix for a subgraph node set.

    Uses vectorised NumPy assignment — O(V²) alloc + O(E) fill.

    Args:
        G:         Full graph (edges are looked up by subgraph membership).
        node_list: Ordered list of node IDs forming the subgraph.

    Returns:
        Square float64 adjacency matrix of shape (|node_list|, |node_list|).
    """
    n = len(node_list)
    A = np.zeros((n, n), dtype=np.float64)
    idx = {node: i for i, node in enumerate(node_list)}

    for u in node_list:
        for v in G.neighbors(u):
            if v in idx:
                weight = G[u][v].get("weight", 1.0)
                A[idx[u]][idx[v]] = weight

    return A


def build_normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Compute the symmetric normalized Laplacian Lₙ = I − D^{-½} A D^{-½}.

    Implements the standard spectral graph theory definition:
      • D = diag(row sums of A)
      • Lₙ = I − D^{-½} A D^{-½}

    Nodes with zero degree receive D^{-½} = 0 (no contribution).

    Complexity: O(V²) — dominated by matrix multiplication.

    Args:
        A: Weighted adjacency matrix.

    Returns:
        Normalized Laplacian matrix of the same shape.
    """
    degree = A.sum(axis=1)

    # D^{-1/2}: replace 0 entries with 0 to avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)

    D_inv_sqrt = np.diag(d_inv_sqrt)  # O(V²) alloc

    # Lₙ = I − D^{-½} A D^{-½}
    L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    return L_norm


def build_unnormalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Compute the unnormalized Laplacian L = D − A.

    Args:
        A: Weighted adjacency matrix.

    Returns:
        Unnormalized Laplacian matrix.
    """
    D = np.diag(A.sum(axis=1))
    return D - A


# -- Spectral decomposition ----------------------------------------------------

def compute_spectral_decomposition(
    L: np.ndarray,
    top_k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the top-k smallest eigenvalues and their eigenvectors.

    Uses ``numpy.linalg.eigh`` which exploits symmetry (real symmetric
    matrix) for efficient computation — O(V³/3) vs O(V³) for general
    ``eig``.  Returns eigenvalues in ascending order (smallest first).

    The smallest eigenvalue (≈ 0 for a connected graph) is the trivial
    Fiedler companion.  We return all k values including λ₀.

    Args:
        L:     Symmetric Laplacian matrix.
        top_k: Number of leading small eigenvalues/vectors to return.

    Returns:
        (eigenvalues, eigenvectors) — shapes (k,) and (V, k).
    """
    n = L.shape[0]
    k = min(top_k, n)  # cannot request more than V eigenpairs

    # eigh returns eigenvalues in ascending order — already sorted
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    return eigenvalues[:k], eigenvectors[:, :k]


# -- Per-partition spectral feature extraction ---------------------------------

def extract_spectral_features_for_partition(
    G: nx.Graph,
    node_list: list[int],
    top_k: int = 8,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Compute spectral features for a single partition.

    Args:
        G:         Full graph.
        node_list: Nodes belonging to this partition.
        top_k:     Number of eigenvalues/vectors to extract.
        normalize: Use normalized (True) or unnormalized (False) Laplacian.

    Returns:
        Dictionary with:
          • ``eigenvalues``  : shape (k,)
          • ``eigenvectors`` : shape (V_p, k)   — V_p = len(node_list)
          • ``node_list``    : the ordered node list
          • ``fiedler_value``: λ₁ (second smallest eigenvalue)
    """
    if len(node_list) == 0:
        return {
            "eigenvalues": np.array([]),
            "eigenvectors": np.array([]).reshape(0, 0),
            "node_list": [],
            "fiedler_value": 0.0,
        }

    # Build Laplacian for this partition's subgraph
    A = build_adjacency_matrix(G, node_list)

    L = build_normalized_laplacian(A) if normalize else build_unnormalized_laplacian(A)

    eigenvalues, eigenvectors = compute_spectral_decomposition(L, top_k=top_k)

    # Fiedler value: second-smallest eigenvalue (index 1), or 0 if only 1 node
    fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "node_list": node_list,
        "fiedler_value": fiedler,
    }


# -- Feature augmentation ------------------------------------------------------

def augment_dataframe_with_spectral_features(
    df: pd.DataFrame,
    G: nx.Graph,
    partition_map: dict[int, int],
    top_k: int = 8,
    normalize: bool = True,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Augment flow DataFrame with per-partition spectral features (D2).

    For each flow row:
      1. Look up its ``node_id`` → ``partition_id``.
      2. Get the partition's eigenvalues and eigenvectors.
      3. Append the following new columns:
           - ``partition_id``
           - ``fiedler_value``
           - ``spectral_eigen_0`` … ``spectral_eigen_{k-1}``
           - ``spectral_proj_0``  … ``spectral_proj_{k-1}``
             (projection of the flow's node onto the i-th eigenvector)

    Complexity: O(P · V_p³) for spectral decomp + O(N) for augmentation.

    Args:
        df:            DataFrame with ``node_id`` and ``window_id`` columns.
        G:             Full device-flow graph.
        partition_map: ``{node_id: partition_id}`` mapping.
        top_k:         Number of spectral features per flow.
        normalize:     Use normalized Laplacian.

    Returns:
        (augmented_df, spectral_store) where spectral_store maps
        partition_id → spectral feature dict.
    """
    # Group nodes by partition
    partitions: dict[int, list[int]] = {}
    for node_id, part_id in partition_map.items():
        partitions.setdefault(part_id, []).append(node_id)

    # Compute spectral decomp per partition
    spectral_store: dict[int, dict] = {}
    for part_id, node_list in sorted(partitions.items()):
        print(f"    Computing spectral features for partition {part_id} "
              f"({len(node_list)} nodes) …")
        spectral_store[part_id] = extract_spectral_features_for_partition(
            G, sorted(node_list), top_k=top_k, normalize=normalize
        )

    # -- Build augmentation columns in O(N) ------------------------------------
    result = df.copy()

    # Map node_id → partition_id (vectorised)
    node_to_part = pd.Series(partition_map)  # type: ignore[arg-type]
    result["partition_id"] = result["node_id"].map(node_to_part).fillna(-1).astype(int)

    # Initialise spectral columns with zeros
    eigen_cols = [f"spectral_eigen_{i}" for i in range(top_k)]
    proj_cols = [f"spectral_proj_{i}" for i in range(top_k)]
    for col in eigen_cols + proj_cols + ["fiedler_value"]:
        result[col] = 0.0

    # Fill per-partition values
    for part_id, spec in spectral_store.items():
        if len(spec["eigenvalues"]) == 0:
            continue

        mask = result["partition_id"] == part_id
        eigenvalues = spec["eigenvalues"]
        eigenvectors = spec["eigenvectors"]  # shape (V_p, k)
        node_list = spec["node_list"]
        fiedler = spec["fiedler_value"]

        node_to_eigrow = {node: i for i, node in enumerate(node_list)}

        # Fill eigenvalue columns (same for all flows in this partition)
        for i, val in enumerate(eigenvalues):
            if i < top_k:
                result.loc[mask, f"spectral_eigen_{i}"] = float(val)

        result.loc[mask, "fiedler_value"] = fiedler

        # Fill eigenvector projection columns (node-specific within partition)
        # O(N_partition) loop over rows — acceptable
        for idx in result.index[mask]:
            node_id = int(result.at[idx, "node_id"])
            row_i = node_to_eigrow.get(node_id, None)
            if row_i is None:
                continue
            for j in range(min(top_k, eigenvectors.shape[1])):
                result.at[idx, f"spectral_proj_{j}"] = float(eigenvectors[row_i, j])

    return result, spectral_store


# -- Main pipeline -------------------------------------------------------------

def spectral_features_pipeline(
    split: str = "validation",
    top_k: int = 8,
    normalize: bool = True,
    graph_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Full spectral feature extraction pipeline.

    Requires that ``graph_builder.py`` and ``graph_partition.py`` have
    already been run for the specified split.

    Args:
        split:      Dataset split to process.
        top_k:      Number of spectral features to extract.
        normalize:  Use normalized Laplacian (recommended).
        graph_dir:  Directory with graph artefacts.
        output_dir: Where to save augmented CSV and spectral pickle.

    Returns:
        (augmented_df, spectral_store)
    """
    g_dir = graph_dir or GRAPH_DIR
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # -- Load artefacts ---------------------------------------------------------
    graph_path = g_dir / f"graph_{split}.gpickle"
    aug_csv_path = g_dir / f"graph_augmented_{split}.csv"
    partition_path = g_dir / "partition_map.json"

    for p in [graph_path, aug_csv_path, partition_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                f"Run graph_builder.py and graph_partition.py first."
            )

    print(f"[spectral_features] Loading graph from {graph_path} …")
    G = load_graph(graph_path)

    print(f"[spectral_features] Loading augmented CSV from {aug_csv_path} …")
    df = pd.read_csv(aug_csv_path)
    print(f"  Loaded {len(df):,} rows.")

    print(f"[spectral_features] Loading partition map from {partition_path} …")
    partition_map = load_partition_map(partition_path)

    # -- Compute spectral features ----------------------------------------------
    laplacian_type = "normalized" if normalize else "unnormalized"
    print(
        f"[spectral_features] Computing top-{top_k} spectral features "
        f"using {laplacian_type} Laplacian …"
    )
    df_augmented, spectral_store = augment_dataframe_with_spectral_features(
        df, G, partition_map, top_k=top_k, normalize=normalize
    )

    new_cols = (
        [f"spectral_eigen_{i}" for i in range(top_k)]
        + [f"spectral_proj_{i}" for i in range(top_k)]
        + ["fiedler_value", "partition_id"]
    )
    print(f"  Added {len(new_cols)} spectral columns to DataFrame.")

    # -- Save artefacts ---------------------------------------------------------
    # Spectral store (per-partition eigenvalues/vectors)
    spectral_pkl_path = out / "spectral_features.pkl"
    with spectral_pkl_path.open("wb") as fh:
        pickle.dump(spectral_store, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Spectral store saved → {spectral_pkl_path}")

    # Augmented CSV with spectral columns
    out_csv = out / f"spectral_augmented_{split}.csv"
    df_augmented.to_csv(out_csv, index=False)
    print(f"  Spectral-augmented CSV saved → {out_csv}")

    # Summary JSON
    summary = {
        "split": split,
        "top_k": top_k,
        "laplacian_type": laplacian_type,
        "new_columns": new_cols,
        "partitions_processed": len(spectral_store),
        "per_partition_summary": {
            str(part_id): {
                "node_count": len(spec["node_list"]),
                "eigenvalues": spec["eigenvalues"].tolist(),
                "fiedler_value": spec["fiedler_value"],
            }
            for part_id, spec in spectral_store.items()
        },
    }
    with (out / "spectral_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Spectral summary saved → {out / 'spectral_summary.json'}")

    print("[spectral_features] Done.")
    return df_augmented, spectral_store


# -- CLI entry-point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute graph Laplacian spectral features and augment "
            "the per-flow feature vectors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split to process (default: validation).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of smallest eigenvalues/vectors to extract (default: 8).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use unnormalized Laplacian (default: normalized).",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Directory containing graph artefacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for spectral artefacts.",
    )
    args = parser.parse_args()

    spectral_features_pipeline(
        split=args.split,
        top_k=args.top_k,
        normalize=not args.no_normalize,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
