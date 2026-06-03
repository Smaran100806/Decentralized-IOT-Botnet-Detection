"""
train_family_baseline.py
========================
Trains four classifiers (LR, RandomForest, XGBoost, LightGBM) for
**multi-class** attack family classification (33 classes).

Usage
-----
    python train_family_baseline.py [--out-dir .]

Expects the processed_ciciot23/ directory produced by preprocess_ciciot23.py to be
present in --out-dir (default: project root).

Outputs
-------
    models/family_lr.pkl
    models/family_rf.pkl
    models/family_xgb.pkl
    models/family_lgbm.pkl
    models/best_family_model.pkl   ← model with best val Macro-F1
    models/best_family_name.txt
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
    accuracy_score, f1_score, classification_report,
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)


def evaluate(name: str, model, X_val, y_val, target_names) -> dict:
    y_pred = model.predict(X_val)
    acc       = accuracy_score(y_val, y_pred)
    macro_f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    metrics = {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    print(f"\n  [{name}]")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    Macro-F1     : {macro_f1:.4f}")
    print(f"    Weighted-F1  : {weighted_f1:.4f}")

    # Per-class breakdown (top 5 worst classes by F1)
    report = classification_report(
        y_val, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {k: v["f1-score"] for k, v in report.items()
                 if k not in ("accuracy", "macro avg", "weighted avg")}
    worst = sorted(per_class.items(), key=lambda x: x[1])[:5]
    if worst:
        print("    Worst 5 classes by F1:")
        for cls, f1 in worst:
            print(f"      {cls:<40} F1={f1:.3f}")
    return metrics


def main(args):
    out = args.out_dir
    model_dir = os.path.join(out, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("\n── Loading preprocessed data ───────────────────────────────────")
    data_dir = os.path.join(out, "processed_ciciot23")
    
    with open(os.path.join(data_dir, "selected_features.json"), "r") as f:
        selected_features = json.load(f)["selected_features"]
        
    with open(os.path.join(data_dir, "label_family_map.json"), "r") as f:
        family_map = json.load(f)
    n_classes = len(family_map)
    target_names = [family_map[str(i)] for i in range(n_classes)]
        
    train_df = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation_clean.csv"))
    
    X_train = train_df[selected_features].values.astype(np.float32)
    y_train = train_df["label_family_id"].values
    X_val = val_df[selected_features].values.astype(np.float32)
    y_val = val_df["label_family_id"].values

    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}")
    print(f"  Number of classes: {n_classes}")

    models = {
        "lr": LogisticRegression(
            max_iter=100, solver="lbfgs", n_jobs=2,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=50, max_depth=20, n_jobs=2, random_state=42,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric="mlogloss", n_jobs=-1, random_state=42,
            num_class=n_classes, tree_method="hist",
        ),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
            num_class=n_classes, verbose=-1,
            objective="multiclass",
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
        metrics = evaluate(name.upper(), model, X_val, y_val, target_names)
        metrics["train_time_s"] = round(elapsed, 2)
        results.append(metrics)
        save(model, os.path.join(model_dir, f"family_{name}.pkl"))
        print(f"    Saved → models/family_{name}.pkl")

    # Best model
    best = max(results, key=lambda r: r["macro_f1"])
    best_name = best["model"].lower()
    save(models[best_name], os.path.join(model_dir, "best_family_model.pkl"))
    with open(os.path.join(model_dir, "best_family_name.txt"), "w") as f:
        f.write(best_name)
    print(f"\n✅ Best family model: {best['model']}  (val Macro-F1 = {best['macro_f1']:.4f})")
    print(f"   Saved → models/best_family_model.pkl\n")

    print("\n── Summary ─────────────────────────────────────────────────────")
    print(f"  {'Model':<8} {'Accuracy':>9} {'Macro-F1':>9} {'WtedF1':>8} {'Time(s)':>9}")
    print("  " + "-" * 48)
    for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
        print(f"  {r['model']:<8} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} "
              f"{r['weighted_f1']:>8.4f} {r['train_time_s']:>9.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".",
                        help="Directory containing the processed_ciciot23 directory (default: .)")
    args = parser.parse_args()
    main(args)
