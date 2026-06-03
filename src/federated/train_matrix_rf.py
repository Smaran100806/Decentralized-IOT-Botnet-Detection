"""
train_matrix_rf.py  --  Task E2  (Team Member 3)
=================================================
Extends Member 1's baseline trainer to accept matrix-derived feature columns
(from matrix_features.py / Task E1) in addition to the original 17 features.

Evaluation Protocol
-------------------
Trains three Random Forest configurations and compares:
  1. Baseline RF   - original 17 selected features only.
  2. Matrix RF     - original 17 + 17 matrix features (34 total).
  3. Combined RF   - original 17 + matrix + spectral (if available).

All models use the same hyperparameter configuration as Member 1's
best model (loaded from models/best_binary_name.txt) so the comparison
is fair and directly extends the centralised baseline (E2 requirement).

Usage
-----
    python train_matrix_rf.py
    python train_matrix_rf.py --split validation --n-trees 100
    python train_matrix_rf.py --skip-matrix-gen   # if CSV already exists
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR  = ROOT / "processed_ciciot23"
MATRIX_DIR     = ROOT / "matrix_artifacts"
GRAPH_DIR      = ROOT / "graph_artifacts"
MODELS_DIR     = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MATRIX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_selected_features() -> list[str]:
    with (PROCESSED_DIR / "selected_features.json").open() as fh:
        return json.load(fh)["selected_features"]


def _load_matrix_features_meta() -> list[str]:
    meta_path = MATRIX_DIR / "matrix_features_meta.json"
    if not meta_path.exists():
        return []
    with meta_path.open() as fh:
        return json.load(fh)["matrix_feature_names"]


def save_pkl(obj, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=5)


def _get_rf_config() -> dict:
    """Read Member 1's best RF hyperparameters, falling back to defaults."""
    best_name_path = MODELS_DIR / "best_binary_name.txt"
    if not best_name_path.exists():
        return {"n_estimators": 100, "max_depth": 20, "n_jobs": -1, "random_state": 42}
    best_name = best_name_path.read_text().strip()
    if best_name == "rf":
        best_model_path = MODELS_DIR / "best_binary_model.pkl"
        if best_model_path.exists():
            with best_model_path.open("rb") as fh:
                model = pickle.load(fh)
            return {
                "n_estimators": getattr(model, "n_estimators", 100),
                "max_depth":    getattr(model, "max_depth", None),
                "n_jobs":       -1,
                "random_state": 42,
                "class_weight": getattr(model, "class_weight", None),
            }
    # Fallback: use a solid default RF config
    return {"n_estimators": 100, "max_depth": 20, "n_jobs": -1, "random_state": 42}


def evaluate_rf(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    rf_config: dict,
) -> dict:
    """Train one RF configuration and return a metrics dictionary."""
    print(f"\n  Training [{name}] - {X_train.shape[1]} features ...")
    t0 = time.perf_counter()
    clf = RandomForestClassifier(**rf_config)
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc = float("nan")

    metrics = {
        "model":       name,
        "n_features":  X_train.shape[1],
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "precision":   float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_test, y_pred, zero_division=0)),
        "macro_f1":    float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc":     roc,
        "train_time_s": round(elapsed, 2),
    }
    print(f"    Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
          f"ROC={metrics['roc_auc']:.4f}  ({elapsed:.1f}s)")

    # Save model
    safe_name = name.lower().replace(" ", "_").replace("+", "_")
    save_pkl(clf, MODELS_DIR / f"matrix_rf_{safe_name}.pkl")
    return metrics, clf


def generate_report(all_metrics: list[dict], output_path: Path) -> None:
    """Write a Markdown comparison report for E2."""
    lines = ["# Matrix Feature RF Experiment Report\n",
             "> **Task E2 - Team Member 3**\n",
             "> Compares baseline RF vs. matrix-augmented RF.\n\n",
             "## Results Summary\n\n",
             "| Model | Features | Accuracy | F1 | Macro-F1 | ROC-AUC | Time(s) |\n",
             "|-------|----------|----------|----|----------|---------|--------|\n"]
    for m in all_metrics:
        lines.append(
            f"| {m['model']} | {m['n_features']} | {m['accuracy']:.4f} | "
            f"{m['f1']:.4f} | {m['macro_f1']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['train_time_s']:.1f} |\n"
        )
    lines.append("\n---\n*Generated by train_matrix_rf.py - Team Member 3.*\n")
    output_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n  Report saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def train_matrix_rf_pipeline(
    split: str = "validation",
    n_trees: Optional[int] = None,
    skip_matrix_gen: bool = False,
) -> list[dict]:
    """Full E2 pipeline: generate matrix features -> train & compare models.

    Args:
        split:           Dataset split to use.
        n_trees:         Override n_estimators (None = use best model config).
        skip_matrix_gen: Skip matrix_features.py if CSV already exists.

    Returns:
        List of metrics dicts for all trained models.
    """
    # -- E1: Generate matrix features if needed --------------------------------
    matrix_csv = MATRIX_DIR / f"matrix_augmented_{split}.csv"
    if not skip_matrix_gen or not matrix_csv.exists():
        print("\n[E2] Running matrix feature extraction (E1) ...")
        from src.federated.matrix_features import matrix_features_pipeline
        matrix_features_pipeline(split=split)

    # -- Load data -------------------------------------------------------------
    print(f"\n[E2] Loading matrix-augmented data from {matrix_csv} ...")
    df = pd.read_csv(matrix_csv)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    base_feats   = [f for f in _load_selected_features()  if f in df.columns]
    matrix_feats = [f for f in _load_matrix_features_meta() if f in df.columns]

    # Also pick up spectral features if available
    spectral_csv = GRAPH_DIR / f"spectral_augmented_{split}.csv"
    spectral_feats: list[str] = []
    spectral_df = None
    if spectral_csv.exists():
        spectral_df = pd.read_csv(spectral_csv)
        spectral_cols = [c for c in spectral_df.columns
                         if c.startswith("spectral_") or c == "fiedler_value"]
        spectral_feats = spectral_cols
        print(f"  Found spectral CSV - {len(spectral_feats)} spectral columns.")

    if "label_binary" not in df.columns:
        raise ValueError("label_binary column not found. Check CSV.")

    y = df["label_binary"].values
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    y_train, y_test = y[idx_train], y[idx_test]

    # RF config (inherits Member 1's best hyperparameters)
    rf_config = _get_rf_config()
    if n_trees is not None:
        rf_config["n_estimators"] = n_trees
    rf_config["class_weight"] = "balanced"

    all_metrics = []

    # -- Model 1: Baseline RF (original 17 features) ---------------------------
    X_base = df[base_feats].values.astype(np.float32)
    m, _ = evaluate_rf(
        "Baseline RF", X_base[idx_train], X_base[idx_test],
        y_train, y_test, base_feats, rf_config
    )
    all_metrics.append(m)

    # -- Model 2: Matrix RF (17 base + 17 matrix = 34 features) ---------------
    if matrix_feats:
        X_mat = df[base_feats + matrix_feats].values.astype(np.float32)
        m, clf_mat = evaluate_rf(
            "Matrix RF", X_mat[idx_train], X_mat[idx_test],
            y_train, y_test, base_feats + matrix_feats, rf_config
        )
        all_metrics.append(m)

        # Save as the primary E2 output model
        save_pkl(clf_mat, MODELS_DIR / "matrix_rf_best.pkl")
        (MODELS_DIR / "matrix_rf_feature_names.json").write_text(
            json.dumps({"features": base_feats + matrix_feats}), encoding="utf-8"
        )
    else:
        print("  [WARNING] No matrix features found - skipping Matrix RF.")

    # -- Model 3: Combined RF (base + matrix + spectral) -----------------------
    if matrix_feats and spectral_feats and spectral_df is not None:
        # Merge spectral columns onto matrix df by index alignment
        spectral_df = spectral_df.reset_index(drop=True)
        df_merged = df.copy().reset_index(drop=True)
        for col in spectral_feats:
            if col in spectral_df.columns:
                df_merged[col] = spectral_df[col]

        all_feats = base_feats + matrix_feats + spectral_feats
        X_comb = df_merged[all_feats].values.astype(np.float32)
        m, _ = evaluate_rf(
            "Combined RF (Base+Matrix+Spectral)",
            X_comb[idx_train], X_comb[idx_test],
            y_train, y_test, all_feats, rf_config
        )
        all_metrics.append(m)

    # -- Summary ---------------------------------------------------------------
    print("\n-- E2 Summary ---------------------------------------------------")
    print(f"  {'Model':<38} {'Feats':>5} {'F1':>8} {'ROC-AUC':>9}")
    print("  " + "-" * 64)
    for r in all_metrics:
        print(f"  {r['model']:<38} {r['n_features']:>5} {r['f1']:>8.4f} {r['roc_auc']:>9.4f}")

    generate_report(all_metrics, ROOT / "matrix_experiment_report.md")
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train matrix-augmented RF - Task E2."
    )
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--n-trees", type=int, default=None)
    parser.add_argument("--skip-matrix-gen", action="store_true",
                        help="Skip matrix_features.py if CSV already exists.")
    args = parser.parse_args()
    train_matrix_rf_pipeline(
        split=args.split,
        n_trees=args.n_trees,
        skip_matrix_gen=args.skip_matrix_gen,
    )
