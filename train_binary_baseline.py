"""
train_binary_baseline.py
========================
Trains four classifiers (LR, RandomForest, XGBoost, LightGBM) for
**binary** classification: Benign (0) vs. Attack (1).

Usage
-----
    python train_binary_baseline.py [--use-smote] [--out-dir .]

Expects the processed_ciciot23/ directory produced by preprocess_ciciot23.py to be
present in --out-dir (default: project root).

Outputs
-------
    models/binary_lr.pkl
    models/binary_rf.pkl
    models/binary_xgb.pkl
    models/binary_lgbm.pkl
    models/best_binary_model.pkl   ← copy of model with best val F1
    models/best_binary_name.txt    ← name of the best model
"""

import argparse
import json
import os
import pickle
import time
import warnings
import pandas as pd

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)


def evaluate(name: str, model, X_val, y_val) -> dict:
    y_pred = model.predict(X_val)
    metrics = {
        "model": name,
        "accuracy":  accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall":    recall_score(y_val, y_pred, zero_division=0),
        "f1":        f1_score(y_val, y_pred, zero_division=0),
    }
    print(f"\n  [{name}]")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1       : {metrics['f1']:.4f}")
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    out = args.out_dir
    model_dir = os.path.join(out, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("\n── Loading preprocessed data ───────────────────────────────────")
    data_dir = os.path.join(out, "processed_ciciot23")
    
    with open(os.path.join(data_dir, "selected_features.json"), "r") as f:
        selected_features = json.load(f)["selected_features"]
        
    train_file = "train_binary_smote.csv" if args.use_smote else "train_clean.csv"
    train_df = pd.read_csv(os.path.join(data_dir, train_file))
    val_df = pd.read_csv(os.path.join(data_dir, "validation_clean.csv"))
    
    X_train = train_df[selected_features].values
    y_train = train_df["label_binary"].values
    X_val = val_df[selected_features].values
    y_val = val_df["label_binary"].values
    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}")
    print(f"  Class balance (train): {np.bincount(y_train)}")

    models = {
        "lr": LogisticRegression(
            max_iter=100, solver="lbfgs", n_jobs=-1, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=50, max_depth=20, n_jobs=-1, random_state=42
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric="logloss", n_jobs=-1, random_state=42,
            tree_method="hist",
        ),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, verbose=-1
        ),
    }

    results = []
    print("\n── Training & Evaluating ───────────────────────────────────────")
    for name, model in models.items():
        print(f"\n  Training {name.upper()} …")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"    Training time: {elapsed:.1f}s")
        metrics = evaluate(name.upper(), model, X_val, y_val)
        metrics["train_time_s"] = round(elapsed, 2)
        results.append(metrics)
        save(model, os.path.join(model_dir, f"binary_{name}.pkl"))
        print(f"    Saved → models/binary_{name}.pkl")

    # ── Pick best model ──────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["f1"])
    best_name = best["model"].lower()
    save(models[best_name], os.path.join(model_dir, "best_binary_model.pkl"))
    with open(os.path.join(model_dir, "best_binary_name.txt"), "w") as f:
        f.write(best_name)
    print(f"\n✅ Best binary model: {best['model']}  (val F1 = {best['f1']:.4f})")
    print(f"   Saved → models/best_binary_model.pkl\n")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────")
    print(f"  {'Model':<8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time(s)':>9}")
    print("  " + "-" * 57)
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(f"  {r['model']:<8} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['train_time_s']:>9.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--out-dir", default=".",
                        help="Directory containing the processed_ciciot23 directory (default: .)")
    args = parser.parse_args()
    main(args)
