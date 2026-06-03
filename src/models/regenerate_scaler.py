"""regenerate_scaler.py
Regenerate scaler.pkl using the current NumPy / sklearn versions.

This is needed when the saved scaler.pkl becomes incompatible due to a
NumPy major-version upgrade (e.g. 1.x → 2.x changes the internal pickle
path from numpy.core to numpy._core).

The script re-derives the scaler parameters incrementally (via
StandardScaler.partial_fit) so that the full 1.6 GB train.csv is never
loaded into memory all at once.

Steps performed:
  1. Read train.csv in chunks.
  2. For each chunk: cast numeric columns to float, replace ±inf with
     train-set fill values (from numeric_fill_stats.json), fill NaN with
     median, then keep only the 17 selected features.
  3. Call scaler.partial_fit(chunk[selected_features]).
  4. Save the fitted scaler as both scaler.pkl (pickle) and
     scaler.joblib (joblib — more robust across NumPy versions).

Usage:
    python regenerate_scaler.py
    python regenerate_scaler.py --chunk-size 200000   # adjust for RAM
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "processed_ciciot23"
TRAIN_CSV = ROOT / "CICIOT23" / "train" / "train.csv"

CHUNK_SIZE = 500_000  # rows per chunk — tune down if RAM is limited


def main() -> None:
    # ── Load supporting artefacts ──────────────────────────────────────────
    meta = json.loads((PROCESSED / "preprocess_metadata.json").read_text())
    fill_stats: dict[str, dict[str, float]] = json.loads(
        (PROCESSED / "numeric_fill_stats.json").read_text()
    )
    selected_features: list[str] = meta["features"]

    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Reading {TRAIN_CSV} in chunks of {CHUNK_SIZE:,} rows …\n")

    scaler = StandardScaler()
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(TRAIN_CSV, chunksize=CHUNK_SIZE)):
        # Keep only the columns we need
        present = [c for c in selected_features if c in chunk.columns]

        for col in present:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            if col in fill_stats:
                fmax = fill_stats[col]["max"]
                fmin = fill_stats[col]["min"]
                fmed = fill_stats[col]["median"]
            else:
                finite = chunk[col].replace([np.inf, -np.inf], np.nan).dropna()
                fmax = float(finite.max()) if not finite.empty else 0.0
                fmin = float(finite.min()) if not finite.empty else 0.0
                fmed = float(finite.median()) if not finite.empty else 0.0

            chunk[col] = (
                chunk[col]
                .replace(np.inf, fmax)
                .replace(-np.inf, fmin)
                .fillna(fmed)
            )

        X = chunk[present].to_numpy(dtype=float)
        scaler.partial_fit(X)
        total_rows += len(chunk)
        print(f"  Chunk {i + 1:>4}: {len(chunk):>8,} rows  |  cumulative: {total_rows:>12,}")

    # ── Save ───────────────────────────────────────────────────────────────
    pkl_path = PROCESSED / "scaler.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(scaler, f)
    print("[OK] Saved {}  ({:,} bytes)".format(pkl_path, pkl_path.stat().st_size))

    if HAS_JOBLIB:
        jbl_path = PROCESSED / "scaler.joblib"
        joblib.dump(scaler, jbl_path)
        print("[OK] Saved {}  ({:,} bytes)".format(jbl_path, jbl_path.stat().st_size))
    else:
        print("  joblib not found -- only pickle format saved.")

    # ── Quick sanity check ─────────────────────────────────────────────────
    print("\n=== Scaler Parameters ===")
    print("  n_features_in_ : {}".format(scaler.n_features_in_))
    print("  mean_          : {}".format(np.round(scaler.mean_, 4).tolist()))
    print("  scale_         : {}".format(np.round(scaler.scale_, 4).tolist()))
    print("\nTotal training rows processed: {:,}".format(total_rows))


if __name__ == "__main__":
    main()
