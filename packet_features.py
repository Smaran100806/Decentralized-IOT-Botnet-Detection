"""
packet_features.py
==================
Maps a FlowRecord to the exact numpy feature vector expected by the
trained scaler and model (FEATURE_COLS order from preprocess_ciciot23.py).

Usage
-----
    from packet_ingest import ingest_csv
    from packet_features import extract_features

    for record in ingest_csv("test.csv", n_flows=10):
        features = extract_features(record)  # shape: (46,)
"""

import json
import os
import numpy as np
from packet_ingest import FlowRecord

FEATURES_PATH = os.path.join("processed_ciciot23", "selected_features.json")
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        f"Selected features not found at {FEATURES_PATH}.\n"
        "Please run preprocess_ciciot23.py first to generate this file."
    )

with open(FEATURES_PATH, "r") as f:
    ORDERED_FIELDS = json.load(f)["selected_features"]

N_FEATURES = len(ORDERED_FIELDS)


def extract_features(record: FlowRecord) -> np.ndarray:
    """
    Convert a FlowRecord → 1D numpy array of shape (46,) with dtype float32.

    The ordering matches the FEATURE_COLS list used during training so the
    array can be passed directly to scaler.transform([[...]])[0].
    """
    return np.array(
        [getattr(record, field, 0.0) for field in ORDERED_FIELDS],
        dtype=np.float32,
    )


def batch_extract(records) -> np.ndarray:
    """
    Convert an iterable of FlowRecords to a 2D array of shape (n, 46).
    Useful for benchmarking or bulk inference.
    """
    return np.vstack([extract_features(r) for r in records])


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from packet_ingest import ingest_csv

    path = sys.argv[1] if len(sys.argv) > 1 else r"Dataset\CICIOT23\test\test.csv"
    records = list(ingest_csv(path, n_flows=5))
    arr = batch_extract(records)
    print(f"Feature matrix shape: {arr.shape}")
    print(f"Feature names ({N_FEATURES}): {ORDERED_FIELDS[:5]} … {ORDERED_FIELDS[-3:]}")
    print(f"First row: {arr[0]}")
