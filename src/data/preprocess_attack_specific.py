"""
preprocess_attack_specific.py
==============================
Builds per-attack feature sets for DDoS-ICMP Flood, DDoS-SYN Flood, and
Mirai-Greeth_flood by augmenting the globally-selected 17 features with
attack-specific extras that were dropped during the global feature-selection
step (due to high correlation or appearing in only one ranking method).

Pipeline per attack:
  1. Load raw CICIOT23 CSVs (train / validation / test)
  2. Filter to [attack rows] + [benign rows]
  3. Build the attack-specific feature superset
  4. Clean: inf → train max/min, NaN → train median (train stats only)
  5. Apply log1p to high-skew features (Rate, flow_duration, Number, Covariance)
     -- NOTE: flow_duration zeros are PRESERVED (they are the attack signal)
  6. Fit StandardScaler on train only; transform all splits
  7. Save per-attack CSVs + scaler + metadata to
     processed_ciciot23/attack_specific/{attack}/

Usage
-----
    python src/data/preprocess_attack_specific.py [--max-rows N]

Outputs (under processed_ciciot23/attack_specific/)
-------
    ddos_icmp/  ddos_syn/  mirai_greeth/
    Each contains: train.csv  validation.csv  test.csv
                   scaler.pkl  metadata.json
    Plus: attack_metadata.json  (top-level summary)
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
DATA_ROOT   = ROOT / "CICIOT23"
OUTPUT_BASE = ROOT / "processed_ciciot23" / "attack_specific"

# ── Attack label patterns (match label_original from raw CSVs) ────────────────
ATTACK_LABEL_PATTERNS: dict[str, list[str]] = {
    "ddos_icmp":   ["DDoS-ICMP_Flood", "ICMP_Flood"],
    "ddos_syn":    ["DDoS-SYN_Flood",  "SYN_Flood"],
    "mirai_greeth":["Mirai-Greeth_flood"],
}

BENIGN_PATTERNS = ["benign", "Benign", "BENIGN"]

# ── Attack-specific feature supersets ─────────────────────────────────────────
# Base 17 features from selected_features.json
BASE_FEATURES = [
    "Header_Length", "Duration", "syn_flag_number", "ack_flag_number",
    "syn_count", "urg_count", "rst_count", "HTTPS", "UDP", "ICMP",
    "Tot sum", "Min", "Max", "AVG", "Tot size", "Covariance", "Variance",
]

# Extra features added back per attack (were dropped in global selection)
EXTRA_FEATURES: dict[str, list[str]] = {
    "ddos_icmp": [
        "Protocol Type",   # CRITICAL — value=1 for ICMP, locked identifier
        "Rate",            # CRITICAL — spikes to thousands pkt/s during flood
        "flow_duration",   # HIGH     — near-zero is attack signal, DO NOT impute
        "Number",          # HIGH     — flood volume indicator
        "TCP",             # MEDIUM   — should be 0 for ICMP flood (negative signal)
    ],
    "ddos_syn": [
        "TCP",             # CRITICAL — must be 1 for SYN flood
        "Rate",            # CRITICAL — extreme packet rate
        "flow_duration",   # HIGH     — near-zero burst window
        "fin_flag_number", # HIGH     — absent in SYN flood (no connection close)
        "psh_flag_number", # MEDIUM   — absent (no data payload)
        "Number",          # MEDIUM   — volume indicator
    ],
    "mirai_greeth": [
        "Protocol Type",   # CRITICAL — value=47 for GRE, strongest discriminator
        "Rate",            # HIGH     — sustained high rate
        "flow_duration",   # HIGH     — longer than ICMP/SYN bursts
        "Number",          # MEDIUM   — packet count
        "TCP",             # MEDIUM   — should be 0 (GRE is not TCP)
    ],
}

# log1p is applied to these BEFORE scaling (high skew, as noted in attack_features.html)
# flow_duration is log-transformed but zeros are preserved first (zeros = attack signal)
LOG1P_FEATURES = ["Rate", "Number", "Covariance"]
LOG1P_WITH_ZERO_PRESERVE = ["flow_duration"]  # zeros kept, positive values log-transformed

# All raw column names that may exist in the dataset (handle naming variations)
COLUMN_ALIASES: dict[str, list[str]] = {
    "Protocol Type":    ["Protocol Type", "Protocol_Type", "protocol_type", "protocoltype"],
    "Rate":             ["Rate", "rate"],
    "flow_duration":    ["flow_duration", "Flow Duration", "flow duration"],
    "Number":           ["Number", "number"],
    "TCP":              ["TCP", "tcp"],
    "fin_flag_number":  ["fin_flag_number", "FIN Flag Count"],
    "psh_flag_number":  ["psh_flag_number", "PSH Flag Count"],
}

CANONICAL_NAMES: dict[str, str] = {alias: canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases}


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_column_name(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the actual column name in df that maps to canonical, or None."""
    if canonical in df.columns:
        return canonical
    for alias in COLUMN_ALIASES.get(canonical, []):
        if alias in df.columns:
            return alias
    return None


def load_raw(path: Path, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    # Standardise column names via alias map
    df.rename(columns=CANONICAL_NAMES, inplace=True)
    return df


def detect_label_column(df: pd.DataFrame) -> str:
    for name in ["label", "Label", "LABEL"]:
        if name in df.columns:
            return name
    return df.columns[-1]


def filter_attack_vs_benign(
    df: pd.DataFrame,
    attack_patterns: list[str],
    label_col: str,
) -> pd.DataFrame:
    """Keep all rows. We are doing One-vs-Rest (Target Attack vs All Others)."""
    return df.copy()


def build_binary_target(df: pd.DataFrame, attack_patterns: list[str], label_col: str) -> pd.DataFrame:
    """Add label_binary (1=target attack, 0=benign AND all other attacks) and label_original."""
    df = df.copy()
    df["label_original"] = df[label_col].astype(str)
    labels = df[label_col].astype(str)
    is_attack = labels.str.contains("|".join(attack_patterns), case=False, na=False)
    df["label_binary"] = np.where(is_attack, 1, 0)
    return df


def get_attack_features(attack: str, df: pd.DataFrame) -> list[str]:
    """Return resolved feature columns available in df for this attack."""
    wanted = BASE_FEATURES + EXTRA_FEATURES.get(attack, [])
    available = []
    seen = set()
    for feat in wanted:
        resolved = resolve_column_name(df, feat)
        if resolved and resolved not in seen:
            available.append(resolved)
            seen.add(resolved)
        elif resolved is None:
            print(f"  [!] Feature '{feat}' not found in dataset - skipping.")
    return available


def clean_numeric(
    train: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """Replace inf/NaN using train-computed statistics. No leakage."""
    stats: dict[str, dict] = {}
    train = train.copy()
    others = {k: v.copy() for k, v in others.items()}

    for col in features:
        for frame in [train] + list(others.values()):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        finite = train[col].replace([np.inf, -np.inf], np.nan).dropna()
        finite = finite[np.isfinite(finite)]
        fill_max    = float(finite.max())    if not finite.empty else 0.0
        fill_min    = float(finite.min())    if not finite.empty else 0.0
        fill_median = float(finite.median()) if not finite.empty else 0.0
        stats[col] = {"max": fill_max, "min": fill_min, "median": fill_median}

        for frame in [train] + list(others.values()):
            frame[col] = frame[col].replace(np.inf, fill_max).replace(-np.inf, fill_min)
            # For flow_duration: DO NOT impute zeros — zeros are attack signal
            if col == "flow_duration":
                # Only fill true NaN, not zeros
                frame[col] = frame[col].where(frame[col].notna(), fill_median)
            else:
                frame[col] = frame[col].fillna(fill_median)

    return train, others, stats


def apply_log1p_transforms(
    frames: list[pd.DataFrame],
    features: list[str],
) -> None:
    """Apply log1p in-place to high-skew features present in frames."""
    for col in LOG1P_FEATURES:
        if col in features:
            for frame in frames:
                if col in frame.columns:
                    frame[col] = np.log1p(np.clip(frame[col], 0, None))

    for col in LOG1P_WITH_ZERO_PRESERVE:
        if col in features:
            for frame in frames:
                if col in frame.columns:
                    # Preserve zeros; apply log1p only to positive values
                    mask = frame[col] > 0
                    frame.loc[mask, col] = np.log1p(frame.loc[mask, col])


def scale(
    train: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(train[features])

    train = train.copy()
    train[features] = scaler.transform(train[features])

    scaled_others: dict[str, pd.DataFrame] = {}
    for name, frame in others.items():
        f = frame.copy()
        f[features] = scaler.transform(frame[features])
        scaled_others[name] = f

    return train, scaled_others, scaler


# ── Main processing per attack ────────────────────────────────────────────────

def process_attack(
    attack: str,
    patterns: list[str],
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    raw_test: pd.DataFrame,
    label_col: str,
    out_dir: Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Processing: {attack.upper().replace('_', '-')}")
    print(f"{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Filter attack vs benign
    train_f = filter_attack_vs_benign(raw_train, patterns, label_col)
    val_f   = filter_attack_vs_benign(raw_val,   patterns, label_col)
    test_f  = filter_attack_vs_benign(raw_test,  patterns, label_col)

    print(f"  Rows after filter — train: {len(train_f):,}  "
          f"val: {len(val_f):,}  test: {len(test_f):,}")

    if len(train_f) == 0:
        print(f"  [!] No rows found for attack '{attack}'. Skipping.")
        return {}

    # 2. Build binary targets (One-vs-Rest)
    train_f = build_binary_target(train_f, patterns, label_col)
    val_f   = build_binary_target(val_f,   patterns, label_col)
    test_f  = build_binary_target(test_f,  patterns, label_col)

    attack_train = int(train_f["label_binary"].sum())
    benign_train = int((train_f["label_binary"] == 0).sum())
    print(f"  Train class balance - target attack: {attack_train:,}  rest (benign + other attacks): {benign_train:,}")

    # 3. Resolve feature columns
    features = get_attack_features(attack, train_f)
    extra    = [f for f in features if f not in BASE_FEATURES]
    print(f"  Feature count: {len(features)} "
          f"(17 base + {len(extra)} extra: {extra})")

    # 4. Ensure all frames have feature columns (fill missing with 0)
    for frame in [train_f, val_f, test_f]:
        for col in features:
            if col not in frame.columns:
                frame[col] = 0.0

    # 5. Clean
    train_c, others_c, stats = clean_numeric(
        train_f, {"validation": val_f, "test": test_f}, features
    )
    val_c  = others_c["validation"]
    test_c = others_c["test"]

    # 6. Log1p transforms
    apply_log1p_transforms([train_c, val_c, test_c], features)

    # 7. Scale
    train_s, others_s, scaler = scale(
        train_c, {"validation": val_c, "test": test_c}, features
    )
    val_s  = others_s["validation"]
    test_s = others_s["test"]

    # 8. Save
    label_cols = ["label_binary", "label_original"]
    cols_to_save = features + [c for c in label_cols if c in train_s.columns]

    train_s[cols_to_save].to_csv(out_dir / "train.csv",      index=False)
    val_s[cols_to_save].to_csv(  out_dir / "validation.csv", index=False)
    test_s[cols_to_save].to_csv( out_dir / "test.csv",       index=False)

    with (out_dir / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)

    meta = {
        "attack": attack,
        "label_patterns": patterns,
        "features": features,
        "base_features": [f for f in features if f in BASE_FEATURES],
        "extra_features": extra,
        "log1p_applied": [f for f in LOG1P_FEATURES + LOG1P_WITH_ZERO_PRESERVE
                          if f in features],
        "zero_preserved": ["flow_duration"] if "flow_duration" in features else [],
        "rows": {
            "train_total":  len(train_s),
            "train_attack": attack_train,
            "train_benign": benign_train,
            "validation":   len(val_s),
            "test":         len(test_s),
        },
        "numeric_fill_stats": stats,
        "scaler_path": str(out_dir / "scaler.pkl"),
    }
    with (out_dir / "metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  [OK] Saved -> {out_dir}")
    return meta


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-attack feature CSVs for the 3 targeted attacks."
    )
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Row cap per split (dev mode). Omit for full dataset.")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] Attack-specific preprocessing")
    print(f"  Raw data root : {DATA_ROOT}")
    print(f"  Output base   : {OUTPUT_BASE}")

    # Load raw CSVs once (shared across attacks)
    train_path = DATA_ROOT / "train"      / "train.csv"
    val_path   = DATA_ROOT / "validation" / "validation.csv"
    test_path  = DATA_ROOT / "test"       / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Raw train CSV not found at {train_path}. "
            "Ensure CICIOT23/train/train.csv exists."
        )

    print(f"\nLoading raw CSVs (max_rows={args.max_rows}) …")
    raw_train = load_raw(train_path, args.max_rows)
    raw_val   = load_raw(val_path,   args.max_rows)
    raw_test  = load_raw(test_path,  args.max_rows)
    label_col = detect_label_column(raw_train)

    print(f"  Loaded — train: {len(raw_train):,}  "
          f"val: {len(raw_val):,}  test: {len(raw_test):,}")
    print(f"  Label column: '{label_col}'")
    print(f"  Columns ({len(raw_train.columns)}): "
          f"{list(raw_train.columns[:10])} …")

    # Unique labels in train
    unique_labels = sorted(raw_train[label_col].astype(str).unique())
    print(f"  Unique labels in train ({len(unique_labels)}): {unique_labels}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_meta: dict[str, dict] = {}

    for attack, patterns in ATTACK_LABEL_PATTERNS.items():
        meta = process_attack(
            attack=attack,
            patterns=patterns,
            raw_train=raw_train,
            raw_val=raw_val,
            raw_test=raw_test,
            label_col=label_col,
            out_dir=OUTPUT_BASE / attack,
        )
        if meta:
            all_meta[attack] = meta

    # Top-level summary
    summary = {
        "run_timestamp": ts,
        "attacks_processed": list(all_meta.keys()),
        "attacks": all_meta,
        "log1p_policy": {
            "features": LOG1P_FEATURES,
            "zero_preserve_features": LOG1P_WITH_ZERO_PRESERVE,
            "note": (
                "flow_duration zeros are kept as-is (they are the DDoS attack signal). "
                "Only positive flow_duration values get log1p applied."
            ),
        },
    }
    with (OUTPUT_BASE / "attack_metadata.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print("[OK] All attacks processed.")
    print(f"  Output: {OUTPUT_BASE}")
    print(f"  Summary: {OUTPUT_BASE / 'attack_metadata.json'}")


if __name__ == "__main__":
    main()
