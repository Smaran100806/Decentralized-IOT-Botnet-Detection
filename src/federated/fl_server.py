"""
fl_server.py  --  Task E3  (Team Member 3)
==========================================
Federated Learning server using the Flower (flwr) framework.

Strategy: Tree-pooling aggregation.
Each client trains a local RandomForestClassifier on its data shard
and sends its fitted trees back. The server pools all trees into a
single merged global forest (horizontal federated learning for RF).

Usage
-----
    # Terminal 1: Start server
    python fl_server.py --rounds 3 --min-clients 2

    # Terminal 2 & 3: Start clients
    python fl_client.py --client-id 0
    python fl_client.py --client-id 1

    # Simulation mode (no network, runs locally):
    python fl_server.py --simulate --n-clients 2 --rounds 3
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "processed_ciciot23"
FEDERATED_DIR = ROOT / "federated_artifacts"
FEDERATED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# RF serialisation helpers
# ---------------------------------------------------------------------------

def rf_to_bytes(clf) -> bytes:
    """Serialise a fitted RF to bytes."""
    return pickle.dumps({
        "estimators":    clf.estimators_,
        "n_features_in": clf.n_features_in_,
        "n_classes":     clf.n_classes_,
        "classes":       clf.classes_,
        "n_outputs":     getattr(clf, "n_outputs_", 1),
    }, protocol=5)


def bytes_to_rf(data: bytes):
    """Deserialise bytes to a skeleton RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier
    d = pickle.loads(data)
    clf = RandomForestClassifier(n_estimators=len(d["estimators"]), random_state=42)
    clf.estimators_    = d["estimators"]
    clf.n_features_in_ = d["n_features_in"]
    clf.n_classes_     = d["n_classes"]
    clf.classes_       = d["classes"]
    clf.n_outputs_     = d.get("n_outputs", 1)
    return clf


# ---------------------------------------------------------------------------
# Tree-pooling aggregation
# ---------------------------------------------------------------------------

def aggregate_forests(client_models: list[tuple]) -> object:
    """Pool all trees from all clients into one global forest.

    Args:
        client_models: List of (clf, n_samples) tuples.

    Returns:
        Merged RandomForestClassifier.
    """
    from sklearn.ensemble import RandomForestClassifier

    all_trees = []
    base_clf, _ = client_models[0]

    for clf, _ in client_models:
        all_trees.extend(clf.estimators_)

    merged = RandomForestClassifier(n_estimators=len(all_trees), random_state=42)
    merged.estimators_    = all_trees
    merged.n_features_in_ = base_clf.n_features_in_
    merged.n_classes_     = base_clf.n_classes_
    merged.classes_       = base_clf.classes_
    merged.n_outputs_     = getattr(base_clf, "n_outputs_", 1)
    log.info(f"  Pooled {len(all_trees)} trees from {len(client_models)} clients.")
    return merged


# ---------------------------------------------------------------------------
# Simulation mode (no real network required)
# ---------------------------------------------------------------------------

def run_simulation(
    n_clients: int = 2,
    rounds: int = 3,
    n_estimators_per_client: int = 50,
    use_matrix_features: bool = True,
    use_encryption: bool = False,
) -> None:
    """Run FL simulation locally without a network server.

    Each 'client' is instantiated as a Python object on the same machine.
    This allows full end-to-end testing without Flower's gRPC layer.
    """
    from src.federated.fl_client import FederatedClient
    from src.federated.secure_aggregation import SecureAggregator

    log.info(f"=== FL Simulation: {n_clients} clients, {rounds} rounds ===")

    # Load the best pre-trained model as global warm-start
    global_model = None
    best_model_path = MODELS_DIR / "best_binary_model.pkl"
    if best_model_path.exists():
        with best_model_path.open("rb") as fh:
            global_model = pickle.load(fh)
        log.info(f"  Warm-started from {best_model_path}")

    # Initialise clients (simulate edge devices with data shards)
    clients = []
    for cid in range(n_clients):
        split = "train" if cid == 0 else "validation"
        client = FederatedClient(
            client_id=cid,
            split=split,
            n_estimators=n_estimators_per_client,
            use_matrix_features=use_matrix_features,
        )
        client.setup(global_model)
        clients.append(client)

    aggregator = SecureAggregator(enabled=use_encryption)
    round_metrics = []

    for rnd in range(1, rounds + 1):
        log.info(f"\n--- Federated Round {rnd}/{rounds} ---")

        # Each client trains locally and returns its model
        client_results = []
        for client in clients:
            local_clf, n_samples, metrics = client.train(global_model)
            log.info(f"  Client {client.client_id}: {n_samples} samples, "
                     f"local F1={metrics.get('f1', 0):.4f}")

            # Apply secure masking if enabled
            model_bytes = rf_to_bytes(local_clf)
            masked_bytes = aggregator.mask(model_bytes)
            client_results.append((masked_bytes, n_samples, metrics))

        # Unmask and aggregate
        unmasked_models = []
        for masked_bytes, n_samples, _ in client_results:
            unmasked = aggregator.unmask(masked_bytes)
            clf = bytes_to_rf(unmasked)
            unmasked_models.append((clf, n_samples))

        global_model = aggregate_forests(unmasked_models)

        # Evaluate global model on each client's validation set
        eval_results = []
        for client in clients:
            val_metrics = client.evaluate(global_model)
            eval_results.append(val_metrics)
            log.info(f"  Client {client.client_id} eval - "
                     f"acc={val_metrics.get('accuracy', 0):.4f}  "
                     f"f1={val_metrics.get('f1', 0):.4f}")

        total_samples = sum(r.get("n_samples", 1) for r in eval_results)
        avg_acc = sum(r.get("accuracy", 0) * r.get("n_samples", 1)
                      for r in eval_results) / total_samples
        avg_f1 = sum(r.get("f1", 0) * r.get("n_samples", 1)
                     for r in eval_results) / total_samples

        round_summary = {
            "round": rnd,
            "global_accuracy": round(avg_acc, 4),
            "global_f1": round(avg_f1, 4),
            "n_clients": n_clients,
            "total_samples": total_samples,
        }
        round_metrics.append(round_summary)
        log.info(f"  Round {rnd} global - acc={avg_acc:.4f}  f1={avg_f1:.4f}")

        # Save round model
        rnd_path = FEDERATED_DIR / f"global_model_round_{rnd}.pkl"
        with rnd_path.open("wb") as fh:
            pickle.dump(global_model, fh, protocol=5)

    # Save final global model
    final_path = FEDERATED_DIR / "global_model_final.pkl"
    with final_path.open("wb") as fh:
        pickle.dump(global_model, fh, protocol=5)
    log.info(f"\nFinal global model saved -> {final_path}")

    # Save round metrics
    metrics_path = FEDERATED_DIR / "federated_round_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(round_metrics, fh, indent=2)
    log.info(f"Round metrics saved -> {metrics_path}")

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Round':<8} {'Global Acc':>12} {'Global F1':>11} {'Clients':>8}")
    print("-" * 55)
    for rm in round_metrics:
        print(f"{rm['round']:<8} {rm['global_accuracy']:>12.4f} "
              f"{rm['global_f1']:>11.4f} {rm['n_clients']:>8}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Network server (requires Flower installed)
# ---------------------------------------------------------------------------

def run_network_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    rounds: int = 3,
    min_clients: int = 2,
) -> None:
    """Launch a real Flower gRPC server."""
    try:
        import flwr as fl
    except ImportError:
        raise ImportError(
            "flwr is not installed. Run: pip install flwr\n"
            "Or use --simulate for a local simulation."
        )

    from sklearn.ensemble import RandomForestClassifier

    class RFStrategy(fl.server.strategy.FedAvg):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.global_model = None
            self.round_metrics = []

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}
            forests = []
            for _, fit_res in results:
                raw = bytes(fit_res.parameters.tensors[0])
                clf = bytes_to_rf(raw)
                forests.append((clf, fit_res.num_examples))
            self.global_model = aggregate_forests(forests)
            model_bytes = rf_to_bytes(self.global_model)
            params = fl.common.ndarrays_to_parameters(
                [np.frombuffer(model_bytes, dtype=np.uint8)]
            )
            return params, {}

    strategy = RFStrategy(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    log.info(f"Starting Flower server on {host}:{port} for {rounds} rounds ...")
    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FL server for IoT botnet detection (Task E3)."
    )
    parser.add_argument("--simulate", action="store_true",
                        help="Run local simulation (no network required).")
    parser.add_argument("--n-clients", type=int, default=2,
                        help="Number of simulated clients (simulation mode).")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of federated rounds.")
    parser.add_argument("--min-clients", type=int, default=2,
                        help="Min clients per round (network mode).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--n-estimators", type=int, default=50,
                        help="Trees per client per round.")
    parser.add_argument("--use-matrix-features", action="store_true",
                        help="Include matrix features (E1) in client training.")
    parser.add_argument("--use-encryption", action="store_true",
                        help="Enable secure aggregation masking (E4).")
    args = parser.parse_args()

    if args.simulate:
        run_simulation(
            n_clients=args.n_clients,
            rounds=args.rounds,
            n_estimators_per_client=args.n_estimators,
            use_matrix_features=args.use_matrix_features,
            use_encryption=args.use_encryption,
        )
    else:
        run_network_server(
            host=args.host,
            port=args.port,
            rounds=args.rounds,
            min_clients=args.min_clients,
        )
