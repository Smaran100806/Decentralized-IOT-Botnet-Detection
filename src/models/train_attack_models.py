"""
train_attack_models.py
=======================
Trains 4 classifiers (LR, RF, XGBoost, LightGBM) for each of the 3 targeted
attacks using their attack-specific feature sets. Each attack is treated as a
binary one-vs-Benign classification problem.

Usage
-----
    python src/models/train_attack_models.py [--attack ddos_icmp|ddos_syn|mirai_greeth|all]

Prerequisites
-------------
    Run preprocess_attack_specific.py first to generate:
    processed_ciciot23/attack_specific/{attack}/train.csv  etc.

Outputs (per attack, under models/attack_specific/{attack}/)
-------
    lr.pkl   rf.pkl   xgb.pkl   lgbm.pkl
    best.pkl           ← copy of model with highest val F1
    best_name.txt
    metrics.json       ← all model metrics for dashboard consumption
    confusion_matrix.png
    feature_importance.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] xgboost not installed - XGB model will be skipped.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[!] lightgbm not installed - LGBM model will be skipped.")

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent.parent
DATA_BASE   = ROOT / "processed_ciciot23" / "attack_specific"
MODEL_BASE  = ROOT / "models" / "attack_specific"
REPORT_BASE = ROOT / "reports" / "attack_specific"

ALL_ATTACKS = ["ddos_icmp", "ddos_syn", "mirai_greeth"]

# ── Display labels ────────────────────────────────────────────────────────────
ATTACK_DISPLAY = {
    "ddos_icmp":    "DDoS-ICMP Flood",
    "ddos_syn":     "DDoS-SYN Flood",
    "mirai_greeth": "Mirai-Greeth_flood",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_pkl(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=5)


def load_attack_data(attack: str):
    """Load train/validation CSVs for an attack."""
    base = DATA_BASE / attack
    if not (base / "train.csv").exists():
        raise FileNotFoundError(
            f"Attack data not found at {base}. "
            "Run preprocess_attack_specific.py first."
        )
    train = pd.read_csv(base / "train.csv")
    val   = pd.read_csv(base / "validation.csv")
    meta_path = base / "metadata.json"
    with meta_path.open() as fh:
        meta = json.load(fh)
    features = meta["features"]
    return train, val, features, meta


# ── Model builders ────────────────────────────────────────────────────────────

def build_models(attack_ratio: float) -> dict:
    """
    Build model dict with class-weight balancing.
    attack_ratio = n_attack / (n_attack + n_benign)
    """
    # For XGB/LGBM, scale_pos_weight = n_negative / n_positive
    # We set pos=attack (1), neg=benign (0)
    n_pos = attack_ratio
    n_neg = 1.0 - attack_ratio
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    models: dict = {
        "lr": LogisticRegression(
            max_iter=300,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["xgb"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        )
    if HAS_LGB:
        models["lgbm"] = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            scale_pos_weight=scale_pos,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    return models


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    y_pred = model.predict(X_val)
    try:
        y_prob = model.predict_proba(X_val)[:, 1]
        roc    = float(roc_auc_score(y_val, y_prob))
    except Exception:
        roc = 0.0

    metrics = {
        "model":     name,
        "accuracy":  float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_val, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc":   roc,
    }
    print(f"    [{name.upper():5s}]  "
          f"Acc={metrics['accuracy']:.4f}  "
          f"P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}  "
          f"ROC={metrics['roc_auc']:.4f}")
    return metrics


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    attack_display: str,
    out_path: Path,
) -> list[list[int]]:
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    cm_list = cm.tolist()

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#13161e")
    ax.set_facecolor("#13161e")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"], color="#e8eaf0")
    ax.set_yticklabels(["Benign", "Attack"], color="#e8eaf0")
    ax.set_xlabel("Predicted", color="#8890a8")
    ax.set_ylabel("True", color="#8890a8")
    ax.set_title(f"{attack_display}\n{model_name} Confusion Matrix",
                 color="#e8eaf0", fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "#0d0f14",
                    fontsize=12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="#13161e")
    plt.close()
    return cm_list


# ── Feature importance ────────────────────────────────────────────────────────

def extract_feature_importance(model, model_name: str, features: list[str]) -> list[dict]:
    try:
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            pairs = sorted(zip(features, imps.tolist()), key=lambda x: -x[1])
            return [{"feature": f, "importance": round(v, 6)} for f, v in pairs]
        if hasattr(model, "coef_"):
            imps = np.abs(model.coef_[0])
            pairs = sorted(zip(features, imps.tolist()), key=lambda x: -x[1])
            return [{"feature": f, "importance": round(v, 6)} for f, v in pairs]
    except Exception:
        pass
    return [{"feature": f, "importance": 0.0} for f in features]


# ── Per-attack training ───────────────────────────────────────────────────────

def train_attack(attack: str) -> dict:
    display = ATTACK_DISPLAY.get(attack, attack)
    print(f"\n{'='*62}")
    print(f"  Attack: {display}")
    print(f"{'='*62}")

    train_df, val_df, features, data_meta = load_attack_data(attack)

    X_train = train_df[features].values.astype(float)
    y_train = train_df["label_binary"].values.astype(int)
    X_val   = val_df[features].values.astype(float)
    y_val   = val_df["label_binary"].values.astype(int)

    n_attack = int(y_train.sum())
    n_total  = len(y_train)
    attack_ratio = n_attack / n_total if n_total > 0 else 0.5

    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}")
    print(f"  Train class balance — attack: {n_attack:,}  "
          f"benign: {n_total - n_attack:,}  "
          f"({attack_ratio*100:.1f}% attack)")

    model_dir  = MODEL_BASE / attack
    report_dir = REPORT_BASE / attack
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(attack_ratio)
    results: list[dict] = []
    feature_importances: dict[str, list[dict]] = {}

    print("\n  Training & evaluating …")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  * {name.upper():5s} trained in {elapsed:.1f}s")

        metrics = evaluate(name, model, X_val, y_val)
        metrics["train_time_s"] = round(elapsed, 2)
        results.append(metrics)

        save_pkl(model, model_dir / f"{name}.pkl")

        fi = extract_feature_importance(model, name, features)
        feature_importances[name] = fi

    # Best model by val F1
    best    = max(results, key=lambda r: r["f1"])
    best_name = best["model"].lower()
    save_pkl(models[best_name], model_dir / "best.pkl")
    (model_dir / "best_name.txt").write_text(best_name)

    print(f"\n  [OK] Best model: {best['model']}  "
          f"(val F1={best['f1']:.4f}  ROC-AUC={best['roc_auc']:.4f})")

    # Confusion matrix for best model
    cm = plot_confusion(
        models[best_name], X_val, y_val,
        best["model"], display,
        report_dir / "confusion_matrix.png",
    )

    # Classification report
    y_pred_best = models[best_name].predict(X_val)
    clf_report  = classification_report(y_val, y_pred_best,
                                        target_names=["Benign", "Attack"],
                                        output_dict=True)

    # Build full metrics JSON (consumed by dashboard)
    metrics_json = {
        "attack": attack,
        "attack_display": display,
        "best_model": best_name,
        "features": features,
        "extra_features": data_meta.get("extra_features", []),
        "rows": data_meta.get("rows", {}),
        "results": results,
        "best": best,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "feature_importance": {
            name: feature_importances[name][:10]   # top-10 per model
            for name in feature_importances
        },
    }

    with (report_dir / "metrics.json").open("w") as fh:
        json.dump(metrics_json, fh, indent=2)

    # Save feature importance for best model separately
    with (report_dir / "feature_importance.json").open("w") as fh:
        json.dump(feature_importances.get(best_name, []), fh, indent=2)

    print(f"  Saved -> {model_dir}")
    print(f"  Report -> {report_dir}")
    return metrics_json


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train per-attack binary classifiers for CICIOT2023."
    )
    parser.add_argument(
        "--attack",
        choices=ALL_ATTACKS + ["all"],
        default="all",
        help="Which attack to train. Default: all.",
    )
    args = parser.parse_args()

    attacks = ALL_ATTACKS if args.attack == "all" else [args.attack]

    all_results: dict[str, dict] = {}
    for attack in attacks:
        result = train_attack(attack)
        if result:
            all_results[attack] = result

    # Combined summary for dashboard
    summary_path = REPORT_BASE / "attack_models_summary.json"
    REPORT_BASE.mkdir(parents=True, exist_ok=True)
    summary = {
        "attacks_trained": list(all_results.keys()),
        "summary": {
            atk: {
                "best_model": r["best_model"],
                "best_f1":    r["best"]["f1"],
                "best_roc":   r["best"]["roc_auc"],
                "best_acc":   r["best"]["accuracy"],
                "features":   len(r["features"]),
            }
            for atk, r in all_results.items()
        },
    }
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*62}")
    print("[OK] All attack models trained.")
    print(f"\n  {'Attack':<20} {'Best Model':<8} {'F1':>8} {'ROC-AUC':>10}")
    print("  " + "-" * 52)
    for atk, r in all_results.items():
        print(f"  {ATTACK_DISPLAY.get(atk, atk):<20} "
              f"{r['best_model']:<8} "
              f"{r['best']['f1']:>8.4f} "
              f"{r['best']['roc_auc']:>10.4f}")
    print(f"\n  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
