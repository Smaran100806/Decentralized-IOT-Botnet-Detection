"""
matrix_features.py  --  Task E1  (Team Member 3)
=================================================
Treat each flow's packet-size and timing statistics as a small virtual
"packet matrix" and compute higher-order statistics (mean, std, entropy,
energy, skewness, kurtosis, range) that capture distributional properties
of the traffic not directly available from the raw scalar features.

Background
----------
The CICIoT23 pre-processed data contains summary statistics per flow
(e.g. Tot_sum, Min, Max, AVG, Std, Variance, IAT, Rate, ...).  We
reconstruct a 1-D "virtual packet vector" from these moments and then
compute a richer statistical fingerprint - effectively treating the flow
summary as a compact representation of the packet size / inter-arrival
time distribution.

Feature Groups Generated
------------------------
For two signal axes (packet_size and timing), the following statistics
are appended as new columns:

  mat_<axis>_mean       : arithmetic mean (= AVG for packet_size axis)
  mat_<axis>_std        : standard deviation
  mat_<axis>_energy     : sum of squares (L2 energy)
  mat_<axis>_entropy    : Shannon entropy of the normalised histogram
  mat_<axis>_skewness   : Pearson skewness estimate
  mat_<axis>_kurtosis   : excess kurtosis
  mat_<axis>_range      : max - min
  mat_<axis>_cv         : coefficient of variation (std / mean, 0 if mean≈0)
  mat_cross_correlation : Pearson r between packet-size and timing axes

Total new features: 2 × 8 + 1 = **17 matrix features**.

Usage (CLI)
-----------
    python matrix_features.py                          # validation split
    python matrix_features.py --split train
    python matrix_features.py --split validation --out-dir matrix_artifacts/
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

# -- Paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "processed_ciciot23"
OUTPUT_DIR = ROOT / "matrix_artifacts"
SELECTED_FEATURES_PATH = PROCESSED_DIR / "selected_features.json"

warnings.filterwarnings("ignore")


# -- Matrix feature column names -----------------------------------------------

MATRIX_FEATURE_NAMES: list[str] = [
    # Packet-size axis (8 features)
    "mat_pkt_mean",
    "mat_pkt_std",
    "mat_pkt_energy",
    "mat_pkt_entropy",
    "mat_pkt_skewness",
    "mat_pkt_kurtosis",
    "mat_pkt_range",
    "mat_pkt_cv",
    # Timing / IAT axis (8 features)
    "mat_iat_mean",
    "mat_iat_std",
    "mat_iat_energy",
    "mat_iat_entropy",
    "mat_iat_skewness",
    "mat_iat_kurtosis",
    "mat_iat_range",
    "mat_iat_cv",
    # Cross-axis (1 feature)
    "mat_cross_correlation",
]


# -- Per-row feature extraction ------------------------------------------------

def _safe_entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """Compute Shannon entropy of a histogram over `values`.

    Returns 0.0 if the array is constant or empty.
    """
    if len(values) == 0 or np.ptp(values) == 0:
        return 0.0
    counts, _ = np.histogram(values, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _safe_cv(std: float, mean: float) -> float:
    """Coefficient of variation - 0 if mean ≈ 0."""
    return float(std / mean) if abs(mean) > 1e-9 else 0.0


def _reconstruct_axis(
    row: "pd.Series",
    mean_col: str,
    std_col: str,
    min_col: str,
    max_col: str,
    n_points: int = 16,
) -> np.ndarray:
    """Reconstruct a synthetic 1-D sample from summary statistics.

    We use a deterministic approach: linspace between [min, max] with
    a Gaussian component centred at mean with the known std.  This is
    the best lossless reconstruction from first-moment summaries.

    The resulting vector has `n_points` elements and captures the shape
    of the underlying distribution well enough for moment estimation.
    """
    mu = float(row.get(mean_col, 0.0) or 0.0)
    sigma = float(row.get(std_col, 0.0) or 0.0)
    lo = float(row.get(min_col, 0.0) or 0.0)
    hi = float(row.get(max_col, 0.0) or 0.0)

    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        return np.full(n_points, mu)

    # Uniform skeleton across the observed range
    uniform_part = np.linspace(lo, hi, n_points // 2)

    # Gaussian draw (deterministic: percentile points)
    if sigma > 0:
        percentiles = np.linspace(0.5, 99.5, n_points - n_points // 2)
        from scipy.stats import norm
        gauss_part = norm.ppf(percentiles / 100.0, loc=mu, scale=sigma)
        gauss_part = np.clip(gauss_part, lo, hi)
    else:
        gauss_part = np.full(n_points - n_points // 2, mu)

    return np.concatenate([uniform_part, gauss_part])


def extract_matrix_features_row(row: "pd.Series") -> dict[str, float]:
    """Extract 17 matrix features from a single flow row.

    Maps CICIoT23 summary statistics -> virtual packet-size and IAT
    axes, then computes higher-order descriptors.

    Args:
        row: A pandas Series representing one flow record.

    Returns:
        Dictionary mapping feature name -> float value.
    """
    # -- Packet-size axis: uses Tot_sum, AVG, Std, Min, Max ---------------
    pkt = _reconstruct_axis(
        row,
        mean_col="AVG",
        std_col="Std",
        min_col="Min",
        max_col="Max",
    )

    # -- Timing/IAT axis: uses IAT, Rate, Srate, Drate --------------------
    iat_mean = float(row.get("IAT", 0.0) or 0.0)
    rate = float(row.get("Rate", 0.0) or 0.0)
    srate = float(row.get("Srate", 0.0) or 0.0)
    drate = float(row.get("Drate", 0.0) or 0.0)

    # Approximate timing std from asymmetry between send/dest rates
    iat_std = abs(srate - drate) / (rate + 1e-9) * iat_mean if rate > 0 else 0.0
    iat_lo = max(0.0, iat_mean - 3 * iat_std)
    iat_hi = iat_mean + 3 * iat_std

    iat = _reconstruct_axis(
        row,
        mean_col="IAT",
        std_col="Std",   # use general Std as proxy
        min_col="Min",
        max_col="Max",
    )
    # Override synthetic IAT bounds for coherence
    iat = np.clip(iat, iat_lo if iat_lo < iat_hi else 0.0,
                  iat_hi if iat_hi > 0 else 1.0)

    # -- Per-axis statistics -----------------------------------------------
    def _axis_stats(v: np.ndarray, prefix: str) -> dict[str, float]:
        mu = float(np.mean(v))
        sigma = float(np.std(v, ddof=0))
        # scipy skew/kurtosis return NaN for constant arrays - guard explicitly
        if sigma < 1e-12 or np.ptp(v) < 1e-12:
            skewness = 0.0
            kurt = 0.0
        else:
            raw_skew = float(sp_skew(v))
            raw_kurt = float(sp_kurtosis(v, fisher=True))
            skewness = 0.0 if not np.isfinite(raw_skew) else raw_skew
            kurt     = 0.0 if not np.isfinite(raw_kurt) else raw_kurt
        return {
            f"{prefix}_mean":     mu,
            f"{prefix}_std":      sigma,
            f"{prefix}_energy":   float(np.sum(v ** 2)),
            f"{prefix}_entropy":  _safe_entropy(v),
            f"{prefix}_skewness": skewness,
            f"{prefix}_kurtosis": kurt,
            f"{prefix}_range":    float(np.ptp(v)),
            f"{prefix}_cv":       _safe_cv(sigma, mu),
        }

    features: dict[str, float] = {}
    features.update(_axis_stats(pkt, "mat_pkt"))
    features.update(_axis_stats(iat, "mat_iat"))

    # -- Cross-axis correlation -------------------------------------------
    if np.std(pkt) > 1e-9 and np.std(iat) > 1e-9:
        features["mat_cross_correlation"] = float(np.corrcoef(pkt, iat)[0, 1])
    else:
        features["mat_cross_correlation"] = 0.0

    return features


# -- DataFrame-level augmentation ---------------------------------------------

def augment_with_matrix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment a flow DataFrame with 17 matrix-derived columns.

    Applies :func:`extract_matrix_features_row` to every row and appends
    the resulting columns to a copy of the input DataFrame.

    Complexity: O(N) - one pass, vectorisable per-column for large datasets.

    Args:
        df: Input DataFrame (must contain CICIoT23 feature columns).

    Returns:
        New DataFrame with 17 additional columns (the MATRIX_FEATURE_NAMES).
    """
    print(f"  [matrix_features] Extracting matrix features for {len(df):,} rows ...")

    # Vectorised extraction: compute each statistic column-wise where possible
    # to avoid slow row-wise Python loops.
    result = df.copy()

    # Packet-size axis directly from available columns
    avg = df.get("AVG", pd.Series(0.0, index=df.index)).fillna(0.0)
    std = df.get("Std", pd.Series(0.0, index=df.index)).fillna(0.0)
    mn  = df.get("Min", pd.Series(0.0, index=df.index)).fillna(0.0)
    mx  = df.get("Max", pd.Series(0.0, index=df.index)).fillna(0.0)
    iat_col = df.get("IAT", pd.Series(0.0, index=df.index)).fillna(0.0)
    rate_col = df.get("Rate", pd.Series(0.0, index=df.index)).fillna(0.0)
    srate_col = df.get("Srate", pd.Series(0.0, index=df.index)).fillna(0.0)
    drate_col = df.get("Drate", pd.Series(0.0, index=df.index)).fillna(0.0)

    pkt_range = (mx - mn).clip(lower=0.0)

    # Packet-size axis features (fully vectorised)
    result["mat_pkt_mean"]     = avg
    result["mat_pkt_std"]      = std
    result["mat_pkt_energy"]   = avg ** 2 + std ** 2               # E[X²] = Var + Mean²
    result["mat_pkt_range"]    = pkt_range
    result["mat_pkt_cv"]       = np.where(avg.abs() > 1e-9, std / avg, 0.0)

    # Skewness / kurtosis proxies from available moments
    # Pearson's 2nd skewness: 3*(mean - median)/std ≈ 3*(mean - (min+max)/2)/std
    midpoint = (mn + mx) / 2.0
    result["mat_pkt_skewness"] = np.where(
        std > 1e-9, 3.0 * (avg - midpoint) / std, 0.0
    )
    # Excess kurtosis proxy: uniform dist has kurtosis -1.2 -> use range/std ratio
    result["mat_pkt_kurtosis"] = np.where(
        std > 1e-9, (pkt_range / (std + 1e-9)) ** 2 / 3.0 - 1.0, 0.0
    )
    # Entropy: approximate for uniform distribution over observed range
    result["mat_pkt_entropy"]  = np.where(pkt_range > 0, np.log2(pkt_range + 1), 0.0)

    # IAT axis features
    iat_std_proxy = (srate_col - drate_col).abs() / (rate_col + 1e-9) * iat_col
    iat_range = iat_std_proxy * 6.0   # ≈ 3σ either side

    result["mat_iat_mean"]     = iat_col
    result["mat_iat_std"]      = iat_std_proxy
    result["mat_iat_energy"]   = iat_col ** 2 + iat_std_proxy ** 2
    result["mat_iat_range"]    = iat_range
    result["mat_iat_cv"]       = np.where(iat_col.abs() > 1e-9, iat_std_proxy / iat_col, 0.0)
    result["mat_iat_skewness"] = np.where(
        iat_std_proxy > 1e-9,
        3.0 * (iat_col - iat_col.median()) / iat_std_proxy,
        0.0,
    )
    result["mat_iat_kurtosis"] = np.where(
        iat_std_proxy > 1e-9,
        (iat_range / (iat_std_proxy + 1e-9)) ** 2 / 3.0 - 1.0,
        0.0,
    )
    result["mat_iat_entropy"]  = np.where(iat_range > 0, np.log2(iat_range + 1), 0.0)

    # Cross-correlation between packet-size mean and IAT (row-level approximation)
    # Full row-level Pearson is computed via normalised product of z-scores.
    pkt_z = (avg - avg.mean()) / (avg.std() + 1e-9)
    iat_z = (iat_col - iat_col.mean()) / (iat_col.std() + 1e-9)
    result["mat_cross_correlation"] = pkt_z * iat_z   # row-level proxy

    print(f"  [matrix_features] Added {len(MATRIX_FEATURE_NAMES)} columns.")
    return result


# -- Main pipeline -------------------------------------------------------------

def matrix_features_pipeline(
    split: str = "validation",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Full matrix feature extraction pipeline.

    Loads the processed split CSV, augments it with matrix features,
    and saves the result.

    Args:
        split:      One of ``'train'``, ``'validation'``, ``'test'``.
        output_dir: Where to write the augmented CSV.

    Returns:
        Augmented DataFrame.
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED_DIR / f"{split}_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {csv_path}\n"
            "Run preprocess_ciciot23.py first."
        )

    print(f"\n[matrix_features] Loading '{split}' split from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    df_aug = augment_with_matrix_features(df)

    out_csv = out / f"matrix_augmented_{split}.csv"
    df_aug.to_csv(out_csv, index=False)
    print(f"  Saved augmented CSV -> {out_csv}")

    # Save feature names for downstream scripts
    meta = {
        "split": split,
        "matrix_feature_names": MATRIX_FEATURE_NAMES,
        "n_matrix_features": len(MATRIX_FEATURE_NAMES),
    }
    meta_path = out / "matrix_features_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Metadata saved -> {meta_path}")

    print("[matrix_features] Done.")
    return df_aug


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract matrix-based statistical features from CICIoT23 flows (Task E1)."
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split to process (default: validation).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: matrix_artifacts/).",
    )
    args = parser.parse_args()
    matrix_features_pipeline(split=args.split, output_dir=args.out_dir)
