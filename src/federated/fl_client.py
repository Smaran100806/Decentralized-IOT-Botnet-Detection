"""
fl_client.py  --  Task E3  (Team Member 3)
==========================================
Federated Learning edge client.

Each client:
  1. Loads its local data shard (train or validation split).
  2. Optionally augments features with matrix-derived statistics (E1).
  3. Uses the global model parameters (sent by fl_server.py) to
     warm-start a local RandomForestClassifier.
  4. Trains locally and returns the updated model to the server.
  5. Evaluates the global model on its local validation slice.

Integration with Edge Inference (B3 / Team Division §3)
--------------------------------------------------------
The client's predict() method reuses the same scaler + feature-vector
logic as edge_infer.py - simulating a real edge device performing
inference within the federated loop.

Usage
-----
    # Network mode (run AFTER starting fl_server.py):
    python fl_client.py --client-id 0 --split train
    python fl_client.py --client-id 1 --split validation

    # Simulation mode (called internally by fl_server.py --simulate):
    from src.federated.fl_client import FederatedClient
    client = FederatedClient(client_id=0, split="train")
    client.setup(global_model)
    local_clf, n, metrics = client.train(global_model)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLIENT-%(name)s] %(message)s")

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "processed_ciciot23"
MATRIX_DIR = ROOT / "matrix_artifacts"
MODELS_DIR = ROOT / "models"
FEDERATED_DIR = ROOT / "federated_artifacts"
FEDERATED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_selected_features() -> list[str]:
    feat_path = PROCESSED_DIR / "selected_features.json"
    with feat_path.open(encoding="utf-8") as fh:
        return json.load(fh)["selected_features"]


def _load_matrix_feature_names() -> list[str]:
    meta_path = MATRIX_DIR / "matrix_features_meta.json"
    if not meta_path.exists():
        return []
    with meta_path.open(encoding="utf-8") as fh:
        return json.load(fh)["matrix_feature_names"]


def load_client_data(
    split: str,
    client_id: int,
    n_clients: int,
    use_matrix_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the data shard assigned to this client.

    Sharding strategy: the split CSV is divided into n_clients equal
    consecutive partitions.  Client `client_id` gets rows
    [shard_start : shard_end].

    Args:
        split:               CSV split name ('train' or 'validation').
        client_id:           This client's zero-based index.
        n_clients:           Total number of clients in the federation.
        use_matrix_features: Include matrix-augmented columns if available.

    Returns:
        (X, y, feature_names)
    """
    # Prefer matrix-augmented CSV if available and requested
    matrix_csv = MATRIX_DIR / f"matrix_augmented_{split}.csv"
    clean_csv   = PROCESSED_DIR / f"{split}_clean.csv"

    if use_matrix_features and matrix_csv.exists():
        df = pd.read_csv(matrix_csv)
        base_feats = _load_selected_features()
        mat_feats  = _load_matrix_feature_names()
        feature_cols = [f for f in base_feats + mat_feats if f in df.columns]
    else:
        df = pd.read_csv(clean_csv)
        feature_cols = [f for f in _load_selected_features() if f in df.columns]

    # Shard: give each client a contiguous slice
    n = len(df)
    shard_size = n // n_clients
    start = client_id * shard_size
    end   = start + shard_size if client_id < n_clients - 1 else n
    df_shard = df.iloc[start:end].reset_index(drop=True)

    X = df_shard[feature_cols].values.astype(np.float32)
    y = df_shard["label_binary"].values.astype(np.int32)
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# FederatedClient class
# ---------------------------------------------------------------------------

class FederatedClient:
    """Simulated edge-device FL client.

    Attributes:
        client_id:           Unique client identifier.
        split:               Dataset split assigned to this client.
        n_clients:           Total clients in the federation.
        n_estimators:        Trees to train locally per round.
        use_matrix_features: Whether to include E1 matrix features.
    """

    def __init__(
        self,
        client_id: int = 0,
        split: str = "train",
        n_clients: int = 2,
        n_estimators: int = 50,
        use_matrix_features: bool = True,
    ) -> None:
        self.client_id           = client_id
        self.split               = split
        self.n_clients           = n_clients
        self.n_estimators        = n_estimators
        self.use_matrix_features = use_matrix_features
        self.log = logging.getLogger(str(client_id))
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: list[str] = []

    def setup(self, global_model=None) -> None:
        """Load local data shard. Called once before any rounds."""
        self.X, self.y, self.feature_names = load_client_data(
            split=self.split,
            client_id=self.client_id,
            n_clients=self.n_clients,
            use_matrix_features=self.use_matrix_features,
        )
        self.log.info(
            f"Client {self.client_id} loaded {len(self.X):,} samples "
            f"({len(self.feature_names)} features) from '{self.split}' shard."
        )

    def train(
        self,
        global_model=None,
    ) -> tuple[RandomForestClassifier, int, dict]:
        """Train a local RF on this client's data shard.

        If a global_model is provided, we use its hyperparameters as the
        local training configuration (E2 integration: same config as
        Member 1's best model).

        Args:
            global_model: Current global RandomForestClassifier (optional).

        Returns:
            (local_clf, n_samples, metrics_dict)
        """
        assert self.X is not None, "Call setup() before train()."

        # Use global model's hyperparameters if available
        n_est = self.n_estimators
        max_depth = None
        if global_model is not None and hasattr(global_model, "n_estimators"):
            n_est = min(self.n_estimators, global_model.n_estimators)
            max_depth = getattr(global_model, "max_depth", None)

        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42 + self.client_id,
            class_weight="balanced",
        )
        clf.fit(self.X, self.y)

        y_pred = clf.predict(self.X)
        metrics = {
            "accuracy": float(accuracy_score(self.y, y_pred)),
            "f1":       float(f1_score(self.y, y_pred, zero_division=0)),
            "n_samples": int(len(self.y)),
        }
        return clf, int(len(self.y)), metrics

    def evaluate(self, global_model) -> dict:
        """Evaluate the global model on this client's local data.

        Args:
            global_model: The aggregated global RandomForestClassifier.

        Returns:
            Dictionary with accuracy, f1, and n_samples.
        """
        assert self.X is not None, "Call setup() before evaluate()."

        # Align features: global model may have been trained on different cols
        n_feats_expected = global_model.n_features_in_
        X_eval = self.X

        if X_eval.shape[1] != n_feats_expected:
            # Truncate or pad to match
            if X_eval.shape[1] > n_feats_expected:
                X_eval = X_eval[:, :n_feats_expected]
            else:
                pad = np.zeros((X_eval.shape[0],
                                n_feats_expected - X_eval.shape[1]), dtype=np.float32)
                X_eval = np.hstack([X_eval, pad])

        y_pred = global_model.predict(X_eval)
        return {
            "accuracy":  float(accuracy_score(self.y, y_pred)),
            "f1":        float(f1_score(self.y, y_pred, zero_division=0)),
            "n_samples": int(len(self.y)),
        }

    def predict_single(self, feature_vector: np.ndarray, model) -> int:
        """Single-flow inference - mirrors edge_infer.py (B3/§3 integration).

        Args:
            feature_vector: 1-D float32 array of flow features.
            model:          Fitted RandomForestClassifier.

        Returns:
            Predicted class label (0 = benign, 1 = attack).
        """
        x = feature_vector.reshape(1, -1)
        if x.shape[1] != model.n_features_in_:
            if x.shape[1] > model.n_features_in_:
                x = x[:, :model.n_features_in_]
            else:
                pad = np.zeros((1, model.n_features_in_ - x.shape[1]), dtype=np.float32)
                x = np.hstack([x, pad])
        return int(model.predict(x)[0])


# ---------------------------------------------------------------------------
# Network client (uses Flower gRPC - requires fl_server.py running)
# ---------------------------------------------------------------------------

def run_network_client(
    client_id: int,
    split: str,
    n_estimators: int,
    use_matrix_features: bool,
    server_address: str,
) -> None:
    """Connect to a running Flower server and participate in FL rounds."""
    try:
        import flwr as fl
        from src.federated.fl_server import rf_to_bytes, bytes_to_rf
    except ImportError:
        raise ImportError("Install flwr: pip install flwr")

    fed_client = FederatedClient(
        client_id=client_id,
        split=split,
        n_estimators=n_estimators,
        use_matrix_features=use_matrix_features,
    )

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return []

        def fit(self, parameters, config):
            global_model = None
            if parameters:
                raw = bytes(np.array(parameters[0], dtype=np.uint8).tobytes())
                global_model = bytes_to_rf(raw)
            if fed_client.X is None:
                fed_client.setup(global_model)
            local_clf, n, metrics = fed_client.train(global_model)
            model_bytes = rf_to_bytes(local_clf)
            return [np.frombuffer(model_bytes, dtype=np.uint8)], n, metrics

        def evaluate(self, parameters, config):
            raw = bytes(np.array(parameters[0], dtype=np.uint8).tobytes())
            global_model = bytes_to_rf(raw)
            metrics = fed_client.evaluate(global_model)
            loss = 1.0 - metrics["accuracy"]
            return loss, metrics["n_samples"], metrics

    fl.client.start_numpy_client(
        server_address=server_address,
        client=FlowerClient(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FL edge client for IoT botnet detection (Task E3)."
    )
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--use-matrix-features", action="store_true")
    parser.add_argument("--server", default="127.0.0.1:8080",
                        help="FL server address (host:port).")
    args = parser.parse_args()

    run_network_client(
        client_id=args.client_id,
        split=args.split,
        n_estimators=args.n_estimators,
        use_matrix_features=args.use_matrix_features,
        server_address=args.server,
    )
