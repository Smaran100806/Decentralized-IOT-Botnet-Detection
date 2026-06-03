"""
train_spectral_rf.py  —  Task D3  (Team Member 2)
==================================================
Train a Random Forest classifier on the **spectral-augmented** feature set
produced by ``spectral_features.py`` and compare validation metrics against
the plain baseline (original 17 features only).

Evaluation Protocol
--------------------
• Uses the **validation split** (never test — test is frozen).
• Trains two models:
    1. Baseline RF  — 17 original features (from ``selected_features.json``)
    2. Spectral RF  — original features + spectral augmentations
                      (eigenvalues, eigenvector projections, fiedler_value,
                       partition_id)
• Reports: Accuracy, Macro-F1, Binary F1, PR-AUC, ROC-AUC, confusion matrix.
• Saves results to ``spectral_experiment_report.md``.

Workflow
--------
If the spectral-augmented CSV does not exist yet, this script will
automatically invoke the full pipeline (graph_builder → graph_partition →
spectral_features) using default parameters before training.

Usage (CLI)
-----------
    python train_spectral_rf.py
    python train_spectral_rf.py --n-trees 200 --top-k 8 --n-partitions 4
    python train_spectral_rf.py --skip-pipeline   # if spectral CSV already exists
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# -- Paths ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "processed_ciciot23"
GRAPH_DIR = ROOT / "graph_artifacts"
SELECTED_FEATURES_PATH = PROCESSED_DIR / "selected_features.json"


# -- Feature loading ------------------------------------------------------------

def load_selected_features() -> list[str]:
    with SELECTED_FEATURES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["selected_features"]


# -- Pipeline invocation -------------------------------------------------------

def run_full_spectral_pipeline(
    split: str,
    k: int,
    window_size: int,
    n_partitions: int,
    top_k: int,
    max_rows: Optional[int],
) -> None:
    """Run graph_builder → graph_partition → spectral_features in sequence."""
    from src.graph.graph_builder import build_graph_pipeline
    from src.graph.graph_partition import partition_graph_pipeline
    from src.graph.spectral_features import spectral_features_pipeline

    print("=" * 60)
    print("[D3] Step 1/3 — Building device-flow graph …")
    print("=" * 60)
    build_graph_pipeline(
        split=split, k=k, window_size=window_size, max_rows=max_rows
    )

    print("\n" + "=" * 60)
    print("[D3] Step 2/3 — Partitioning graph …")
    print("=" * 60)
    partition_graph_pipeline(split=split, n_partitions=n_partitions)

    print("\n" + "=" * 60)
    print("[D3] Step 3/3 — Extracting spectral features …")
    print("=" * 60)
    spectral_features_pipeline(split=split, top_k=top_k)


# -- Model training & evaluation ------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 100,
    random_state: int = 42,
    label: str = "Model",
) -> dict:
    """Train a Random Forest and return a metrics dictionary.

    Args:
        X_train:       Training feature matrix.
        X_test:        Testing feature matrix.
        y_train:       Training binary target vector.
        y_test:        Testing binary target vector.
        feature_names: Column names (for feature importance reporting).
        n_estimators:  Number of trees.
        random_state:  Seed.
        label:         Human-readable model name for logging.

    Returns:
        Dictionary of evaluation metrics.
    """
    print(f"\n  Training {label} (n_estimators={n_estimators}) …")
    t0 = time.perf_counter()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"  Training complete in {elapsed:.1f}s.")

    y_pred = clf.predict(X_test)

    # Probability estimates for AUC metrics
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
    except Exception:
        roc_auc = float("nan")
        pr_auc = float("nan")

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    binary_f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    # Top-10 feature importances
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [(feature_names[i], float(importances[i])) for i in top_idx]

    print(
        f"  Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f} | "
        f"Binary-F1: {binary_f1:.4f} | ROC-AUC: {roc_auc:.4f}"
    )

    return {
        "label": label,
        "n_features": X_train.shape[1],
        "n_samples": X_test.shape[0],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "top_features": top_features,
        "training_time_s": elapsed,
    }


# -- Report generation ---------------------------------------------------------

def generate_report(
    baseline_metrics: dict,
    spectral_metrics: dict,
    config: dict,
    output_path: Path,
) -> None:
    """Write a comprehensive Markdown experiment report.

    Args:
        baseline_metrics: Metrics dict from baseline RF.
        spectral_metrics: Metrics dict from spectral RF.
        config:           Experiment configuration parameters.
        output_path:      Path to write the ``.md`` file.
    """

    def delta(key: str) -> str:
        b = baseline_metrics[key]
        s = spectral_metrics[key]
        if isinstance(b, float) and isinstance(s, float):
            diff = s - b
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.4f}"
        return "N/A"

    def fmt_top_features(features: list) -> str:
        rows = []
        for rank, (name, importance) in enumerate(features, 1):
            rows.append(f"| {rank} | `{name}` | {importance:.4f} |")
        return "\n".join(rows)

    def fmt_cm(cm: list) -> str:
        if len(cm) == 2:
            return (
                f"| | Pred Benign | Pred Attack |\n"
                f"|---|---|---|\n"
                f"| **True Benign** | {cm[0][0]:,} | {cm[0][1]:,} |\n"
                f"| **True Attack** | {cm[1][0]:,} | {cm[1][1]:,} |"
            )
        return str(cm)

    report = f"""# Spectral Feature Experiment Report

> **Task D3 — Team Member 2**
> Evaluating the impact of Graph Laplacian spectral features on Random Forest
> botnet detection performance using the CICIoT2023 dataset.

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset split | `{config['split']}` |
| Virtual device nodes (K) | {config['k']} |
| Temporal window size | {config['window_size']} rows |
| Graph partitions | {config['n_partitions']} |
| Spectral features (top-k) | {config['top_k']} |
| RF n_estimators | {config['n_estimators']} |
| Laplacian type | Normalized (Lₙ = I − D⁻¹ᐟ² A D⁻¹ᐟ²) |
| Random seed | 42 |

---

## Results Summary

| Metric | Baseline RF ({baseline_metrics['n_features']} features) | Spectral RF ({spectral_metrics['n_features']} features) | Δ (Spectral − Baseline) |
|--------|---------|---------|------|
| **Accuracy** | {baseline_metrics['accuracy']:.4f} | {spectral_metrics['accuracy']:.4f} | {delta('accuracy')} |
| **Macro F1** | {baseline_metrics['macro_f1']:.4f} | {spectral_metrics['macro_f1']:.4f} | {delta('macro_f1')} |
| **Binary F1** | {baseline_metrics['binary_f1']:.4f} | {spectral_metrics['binary_f1']:.4f} | {delta('binary_f1')} |
| **ROC-AUC** | {baseline_metrics['roc_auc']:.4f} | {spectral_metrics['roc_auc']:.4f} | {delta('roc_auc')} |
| **PR-AUC** | {baseline_metrics['pr_auc']:.4f} | {spectral_metrics['pr_auc']:.4f} | {delta('pr_auc')} |
| Training time (s) | {baseline_metrics['training_time_s']:.1f} | {spectral_metrics['training_time_s']:.1f} | — |

---

## Baseline RF — Detailed Results

### Confusion Matrix

{fmt_cm(baseline_metrics['confusion_matrix'])}

### Classification Report

```
{baseline_metrics['classification_report']}
```

### Top-10 Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
{fmt_top_features(baseline_metrics['top_features'])}

---

## Spectral RF — Detailed Results

### Confusion Matrix

{fmt_cm(spectral_metrics['confusion_matrix'])}

### Classification Report

```
{spectral_metrics['classification_report']}
```

### Top-10 Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
{fmt_top_features(spectral_metrics['top_features'])}

---

## Analysis

### Spectral Feature Contribution
The spectral augmentation adds **{spectral_metrics['n_features'] - baseline_metrics['n_features']}
new features** derived from the Graph Laplacian eigen-decomposition:
- `spectral_eigen_i`: Eigenvalues of the partition's normalized Laplacian.
- `spectral_proj_i`: Projection of each node onto the i-th eigenvector.
- `fiedler_value`: Algebraic connectivity (λ₁) of the partition.
- `partition_id`: Edge cluster assignment.

### Why Spectral Features Help (or Don't)
- **Coordinated botnets** generate correlated traffic patterns that manifest
  as anomalous eigenvalue distributions — unusually small Fiedler values
  indicate isolated, weakly-connected attack clusters.
- **Distributed DDoS** appears as dense subgraphs within a partition,
  reflected in larger spectral gaps.
- If the Δ metrics are marginal, it may indicate that the flow-level
  tabular features already capture most discriminative signal, and that
  graph structure adds redundant information.

### Limitations
- Virtual device nodes (K-Means clusters) are a proxy for real device
  identities. With raw IP addresses, graph quality would improve.
- The evaluation uses the **validation split only** (test is frozen).
  Final numbers will differ slightly after full test evaluation.
- The validation split was internally split 80/20 into train/test sets to
  evaluate out-of-sample performance without touching the frozen test set.

---

## Integration Note

These spectral features are ready to be merged with Team Member 1's
`baseline_evaluation.ipynb`.  Add a new cell that:
1. Loads `graph_artifacts/spectral_augmented_validation.csv`.
2. Extracts the spectral columns and the original 17 features.
3. Re-trains Member 1's best RF configuration on the augmented set.
4. Appends the results table to the centralized metrics notebook.

See `team_division_and_integration.md` § Integration Checkpoint.

---

*Generated automatically by `train_spectral_rf.py` — Team Member 2.*
"""

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\n  Report saved -> {output_path}")


# -- Main pipeline -------------------------------------------------------------

def train_spectral_rf_pipeline(
    split: str = "validation",
    k: int = 20,
    window_size: int = 500,
    n_partitions: int = 4,
    top_k: int = 8,
    n_estimators: int = 100,
    max_rows: Optional[int] = None,
    skip_pipeline: bool = False,
) -> None:
    """Full D3 pipeline: spectral extraction → RF training → comparison report.

    Args:
        split:         Dataset split to use.
        k:             Virtual device node count for graph builder.
        window_size:   Temporal window size.
        n_partitions:  Graph partitions count.
        top_k:         Spectral features per flow.
        n_estimators:  Random Forest tree count.
        max_rows:      Row cap for quick testing.
        skip_pipeline: Skip graph/spectral construction if CSV already exists.
    """
    spectral_csv = GRAPH_DIR / f"spectral_augmented_{split}.csv"

    if not skip_pipeline or not spectral_csv.exists():
        run_full_spectral_pipeline(
            split=split,
            k=k,
            window_size=window_size,
            n_partitions=n_partitions,
            top_k=top_k,
            max_rows=max_rows,
        )

    # -- Load data --------------------------------------------------------------
    print(f"\n[D3] Loading spectral-augmented CSV …")
    df = pd.read_csv(spectral_csv)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    original_features = load_selected_features()
    spectral_cols = (
        [f"spectral_eigen_{i}" for i in range(top_k)]
        + [f"spectral_proj_{i}" for i in range(top_k)]
        + ["fiedler_value", "partition_id"]
    )

    # Filter to only columns that actually exist in the CSV
    available_spectral = [c for c in spectral_cols if c in df.columns]
    available_original = [c for c in original_features if c in df.columns]

    if "label_binary" not in df.columns:
        raise ValueError("Column 'label_binary' not found. Check the CSV.")

    from sklearn.model_selection import train_test_split
    y = df["label_binary"].to_numpy()

    print("\n[D3] Splitting data into 80% train / 20% test (intra-validation split) …")
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    y_train = y[idx_train]
    y_test = y[idx_test]

    # -- Baseline RF ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[D3] Training Baseline RF (original 17 features) …")
    print("=" * 60)
    X_base = df[available_original].to_numpy(dtype=np.float32)
    X_base_train, X_base_test = X_base[idx_train], X_base[idx_test]
    
    baseline_metrics = train_and_evaluate(
        X_base_train, X_base_test, y_train, y_test, available_original, n_estimators=n_estimators,
        label="Baseline RF"
    )

    # -- Spectral RF ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[D3] Training Spectral RF (original + spectral features) …")
    print("=" * 60)
    all_features = available_original + available_spectral
    X_spec = df[all_features].to_numpy(dtype=np.float32)
    X_spec_train, X_spec_test = X_spec[idx_train], X_spec[idx_test]
    
    spectral_metrics = train_and_evaluate(
        X_spec_train, X_spec_test, y_train, y_test, all_features, n_estimators=n_estimators,
        label="Spectral RF"
    )

    # -- Generate report --------------------------------------------------------
    config = {
        "split": split,
        "k": k,
        "window_size": window_size,
        "n_partitions": n_partitions,
        "top_k": top_k,
        "n_estimators": n_estimators,
    }
    report_path = ROOT / "spectral_experiment_report.md"
    generate_report(baseline_metrics, spectral_metrics, config, report_path)

    print("\n" + "=" * 60)
    print("[D3] COMPLETE — Metric Delta (Spectral − Baseline):")
    for metric in ["accuracy", "macro_f1", "binary_f1", "roc_auc", "pr_auc"]:
        b = baseline_metrics[metric]
        s = spectral_metrics[metric]
        if isinstance(b, float) and isinstance(s, float):
            sign = "+" if (s - b) >= 0 else ""
            print(f"  {metric:12s}: {b:.4f} → {s:.4f}  ({sign}{s-b:.4f})")
    print("=" * 60)


# -- CLI entry-point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare Baseline vs. Spectral-augmented Random Forest "
            "on CICIoT2023 (Task D3)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="validation",
        help="Dataset split to use (default: validation; test is frozen).",
    )
    parser.add_argument("--k", type=int, default=20,
                        help="Virtual device nodes (default: 20).")
    parser.add_argument("--window-size", type=int, default=500,
                        help="Temporal window size in rows (default: 500).")
    parser.add_argument("--n-partitions", type=int, default=4,
                        help="Graph partitions (default: 4).")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Spectral features per flow (default: 8).")
    parser.add_argument("--n-trees", type=int, default=100,
                        help="Random Forest estimators (default: 100).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Row cap for quick testing.")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip graph/spectral steps if CSV already exists.")
    args = parser.parse_args()

    train_spectral_rf_pipeline(
        split=args.split,
        k=args.k,
        window_size=args.window_size,
        n_partitions=args.n_partitions,
        top_k=args.top_k,
        n_estimators=args.n_trees,
        max_rows=args.max_rows,
        skip_pipeline=args.skip_pipeline,
    )


if __name__ == "__main__":
    main()
