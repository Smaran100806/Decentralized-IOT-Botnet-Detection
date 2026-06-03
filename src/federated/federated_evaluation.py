"""
federated_evaluation.py  --  Task E5  (Team Member 3)
======================================================
Evaluate the federated global model on the frozen test set and produce
a comprehensive metrics report.  This fulfils the E5 requirement:

    "after a few rounds, evaluate the global model on the frozen test
     set and report metrics. Add a section to baseline_evaluation.ipynb."

This script acts as a standalone alternative to the notebook cell -
it generates `federated_evaluation_report.md` and also appends a
summary dict to `federated_artifacts/federated_evaluation_report.json`
that can be loaded inside `baseline_evaluation.ipynb` with one line.

Integration with baseline_evaluation.ipynb (E5)
-----------------------------------------------
To add the federated results to the notebook, add a cell:

    import json
    from pathlib import Path
    report = json.loads(
        (Path("federated_artifacts") / "federated_evaluation_report.json").read_text()
    )
    print(f"Federated Global Model - Accuracy: {report['accuracy']:.4f}  "
          f"F1: {report['f1']:.4f}  ROC-AUC: {report['roc_auc']:.4f}")

Usage
-----
    python federated_evaluation.py
    python federated_evaluation.py --model-path federated_artifacts/global_model_final.pkl
    python federated_evaluation.py --use-matrix-features
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR  = ROOT / "processed_ciciot23"
MATRIX_DIR     = ROOT / "matrix_artifacts"
MODELS_DIR     = ROOT / "models"
FEDERATED_DIR  = ROOT / "federated_artifacts"
FEDERATED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_selected_features() -> list[str]:
    with (PROCESSED_DIR / "selected_features.json").open() as fh:
        return json.load(fh)["selected_features"]


def _load_matrix_feature_names() -> list[str]:
    meta_path = MATRIX_DIR / "matrix_features_meta.json"
    if not meta_path.exists():
        return []
    with meta_path.open() as fh:
        return json.load(fh)["matrix_feature_names"]


def _load_test_data(use_matrix_features: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the frozen test split.

    Prefers matrix-augmented CSV if available and requested; falls back
    to the standard clean CSV.

    Args:
        use_matrix_features: Load from matrix_artifacts/ if True.

    Returns:
        (X, y, feature_names)
    """
    matrix_csv = MATRIX_DIR / "matrix_augmented_test.csv"
    clean_csv  = PROCESSED_DIR / "test_clean.csv"

    if use_matrix_features and matrix_csv.exists():
        df = pd.read_csv(matrix_csv)
        base_feats = _load_selected_features()
        mat_feats  = _load_matrix_feature_names()
        feature_cols = [f for f in base_feats + mat_feats if f in df.columns]
        print(f"  Using matrix-augmented test CSV ({len(feature_cols)} features).")
    elif clean_csv.exists():
        df = pd.read_csv(clean_csv)
        feature_cols = [f for f in _load_selected_features() if f in df.columns]
        print(f"  Using clean test CSV ({len(feature_cols)} features).")
    else:
        raise FileNotFoundError(
            f"No test CSV found at {clean_csv} or {matrix_csv}.\n"
            "Run preprocess_ciciot23.py (and optionally matrix_features.py) first."
        )

    if "label_binary" not in df.columns:
        raise ValueError("'label_binary' column missing from test CSV.")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_binary"].values.astype(np.int32)
    return X, y, feature_cols


def _align_features(X: np.ndarray, model) -> np.ndarray:
    """Pad or truncate X to match the model's expected feature count."""
    n_expected = model.n_features_in_
    if X.shape[1] == n_expected:
        return X
    if X.shape[1] > n_expected:
        return X[:, :n_expected]
    pad = np.zeros((X.shape[0], n_expected - X.shape[1]), dtype=np.float32)
    return np.hstack([X, pad])


def _fmt_confusion_matrix(cm: list) -> str:
    if len(cm) == 2:
        return (
            f"| | Pred Benign | Pred Attack |\n"
            f"|---|---|---|\n"
            f"| **True Benign** | {cm[0][0]:,} | {cm[0][1]:,} |\n"
            f"| **True Attack** | {cm[1][0]:,} | {cm[1][1]:,} |"
        )
    return str(cm)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_federated_model(
    model_path: Path,
    use_matrix_features: bool = False,
) -> dict:
    """Evaluate the federated global model on the frozen test set.

    Args:
        model_path:          Path to the final global model pickle.
        use_matrix_features: Use matrix-augmented test data if available.

    Returns:
        Metrics dictionary.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Global model not found: {model_path}\n"
            "Run: python fl_server.py --simulate --n-clients 2 --rounds 3"
        )

    print(f"\n[E5] Loading global federated model from {model_path} ...")
    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    n_trees = len(model.estimators_) if hasattr(model, "estimators_") else "?"
    print(f"  Model loaded - {n_trees} trees, {model.n_features_in_} features expected.")

    print("\n[E5] Loading frozen test set ...")
    X_test, y_test, feature_names = _load_test_data(use_matrix_features)
    print(f"  Test set: {X_test.shape[0]:,} rows, {X_test.shape[1]} features.")

    # Align features
    X_test = _align_features(X_test, model)

    print("\n[E5] Running inference ...")
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc    = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
    except Exception:
        roc = pr_auc = float("nan")

    acc      = float(accuracy_score(y_test, y_pred))
    f1       = float(f1_score(y_test, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cm       = confusion_matrix(y_test, y_pred).tolist()
    report   = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")

    return {
        "model_path":        str(model_path),
        "n_trees":           n_trees,
        "n_features":        int(model.n_features_in_),
        "test_samples":      int(len(y_test)),
        "accuracy":          acc,
        "f1":                f1,
        "macro_f1":          macro_f1,
        "roc_auc":           roc,
        "pr_auc":            pr_auc,
        "confusion_matrix":  cm,
        "classification_report": report,
    }


def write_evaluation_report(metrics: dict, round_metrics_path: Path) -> Path:
    """Write a Markdown evaluation report for E5."""
    round_metrics: list[dict] = []
    if round_metrics_path.exists():
        with round_metrics_path.open() as fh:
            round_metrics = json.load(fh)

    rounds_table = ""
    if round_metrics:
        rounds_table = (
            "\n## Federated Training Round Progression\n\n"
            "| Round | Global Accuracy | Global F1 | Clients |\n"
            "|-------|----------------|-----------|--------|\n"
        )
        for rm in round_metrics:
            rounds_table += (
                f"| {rm['round']} | {rm.get('global_accuracy', 'N/A')} | "
                f"{rm.get('global_f1', 'N/A')} | {rm.get('n_clients', 'N/A')} |\n"
            )

    report_md = f"""# Federated Model Evaluation Report

> **Task E5 - Team Member 3**
> Final evaluation of the federated global model on the **frozen test set**.

---

## Global Model Configuration

| Parameter | Value |
|-----------|-------|
| Model path | `{metrics['model_path']}` |
| Trees (pooled) | {metrics['n_trees']} |
| Features | {metrics['n_features']} |
| Test samples | {metrics['test_samples']:,} |

---

## Test Set Performance

| Metric | Federated Global Model |
|--------|----------------------|
| **Accuracy** | {metrics['accuracy']:.4f} |
| **Binary F1** | {metrics['f1']:.4f} |
| **Macro F1** | {metrics['macro_f1']:.4f} |
| **ROC-AUC** | {metrics['roc_auc']:.4f} |
| **PR-AUC** | {metrics['pr_auc']:.4f} |

---

## Confusion Matrix

{_fmt_confusion_matrix(metrics['confusion_matrix'])}

---

## Classification Report

```
{metrics['classification_report']}
```
{rounds_table}
---

## Integration with baseline_evaluation.ipynb

Add the following cell to the notebook (E5 integration):

```python
import json
from pathlib import Path

report = json.loads(
    (Path("federated_artifacts") / "federated_evaluation_report.json").read_text()
)
print(f"Federated Global Model Evaluation")
print(f"  Accuracy : {{report['accuracy']:.4f}}")
print(f"  F1       : {{report['f1']:.4f}}")
print(f"  ROC-AUC  : {{report['roc_auc']:.4f}}")
```

---
*Generated automatically by `federated_evaluation.py` - Team Member 3.*
"""
    report_path = FEDERATED_DIR / "federated_evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n  Markdown report saved -> {report_path}")

    json_path = FEDERATED_DIR / "federated_evaluation_report.json"
    with json_path.open("w", encoding="utf-8") as fh:
        # Remove the verbose classification_report string for the JSON
        compact = {k: v for k, v in metrics.items() if k != "classification_report"}
        json.dump(compact, fh, indent=2)
    print(f"  JSON summary saved   -> {json_path}")

    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate federated global model on frozen test set (Task E5)."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=FEDERATED_DIR / "global_model_final.pkl",
        help="Path to the final global model pickle.",
    )
    parser.add_argument(
        "--use-matrix-features",
        action="store_true",
        help="Use matrix-augmented test data if available.",
    )
    args = parser.parse_args()

    metrics = evaluate_federated_model(
        model_path=args.model_path,
        use_matrix_features=args.use_matrix_features,
    )

    round_metrics_path = FEDERATED_DIR / "federated_round_metrics.json"
    write_evaluation_report(metrics, round_metrics_path)
    print("\n[E5] Federated evaluation complete.")
