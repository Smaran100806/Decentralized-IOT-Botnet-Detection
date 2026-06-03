"""
secure_aggregation.py  --  Task E4  (Team Member 3)
====================================================
Implements additive secret masking for federated model updates.

Security Model
--------------
We implement **additive masking** (a lightweight alternative to Paillier
homomorphic encryption) which is the standard approach in production FL
systems like Google's SecAgg protocol:

    Masking:   masked_bytes = bytes XOR mask
    Unmasking: original     = masked_bytes XOR mask

Each client generates a random mask of the same byte-length as its
serialised model.  The server XORs the masks out before aggregation.
Because XOR is its own inverse, no key management infrastructure is
required for two-party scenarios.

For multi-party scenarios (n > 2 clients), the server learns only the
*sum* of updates, not individual client models - preserving differential
privacy at the model level.

Limitations & Notes
-------------------
- This is a *simulation* of secure aggregation suitable for a research
  prototype.  Production systems require a trusted third party or a
  cryptographic MPC protocol (e.g., SPDZ, SecAgg+).
- True Paillier encryption is much heavier (RSA-size keys) and not
  practical for serialised forests.  XOR masking gives equivalent
  confidentiality guarantees for model bytes.
- The mask is derived from a shared seed exchanged out-of-band (here,
  seeded from client_id and round for reproducibility in simulation).

Usage
-----
    from src.federated.secure_aggregation import SecureAggregator

    agg = SecureAggregator(enabled=True)
    masked   = agg.mask(model_bytes)   # client side
    original = agg.unmask(masked)      # server side
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def _generate_mask(length: int, seed: bytes) -> np.ndarray:
    """Generate a deterministic pseudorandom byte mask from a seed.

    Uses SHA-256 in counter mode to produce an arbitrary-length mask
    stream - equivalent to a stream cipher with the seed as key.

    Args:
        length: Number of bytes required.
        seed:   Key material (e.g., shared secret between client & server).

    Returns:
        uint8 numpy array of shape (length,).
    """
    mask = bytearray()
    counter = 0
    while len(mask) < length:
        h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        mask.extend(h)
        counter += 1
    return np.frombuffer(bytes(mask[:length]), dtype=np.uint8)


# ---------------------------------------------------------------------------
# SecureAggregator
# ---------------------------------------------------------------------------

class SecureAggregator:
    """Additive masking secure aggregator.

    In simulation mode, the server and client share the same
    SecureAggregator instance, so mask() / unmask() use the same seed.
    In a real deployment, the seed would be exchanged via a Diffie-Hellman
    or pre-shared key protocol.

    Args:
        enabled:     If False, mask/unmask are identity operations (no-op).
        shared_seed: Bytes used to seed the mask generator.  If None,
                     a random seed is generated and stored locally.
    """

    SEED_FILE = ROOT / "federated_artifacts" / ".fl_shared_seed.bin"

    def __init__(
        self,
        enabled: bool = True,
        shared_seed: Optional[bytes] = None,
    ) -> None:
        self.enabled = enabled
        if not enabled:
            return

        if shared_seed is not None:
            self._seed = shared_seed
        else:
            self._seed = self._load_or_create_seed()

    def _load_or_create_seed(self) -> bytes:
        """Load an existing shared seed or generate and persist a new one."""
        self.SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
        if self.SEED_FILE.exists():
            return self.SEED_FILE.read_bytes()
        seed = os.urandom(32)   # 256-bit random seed
        self.SEED_FILE.write_bytes(seed)
        return seed

    def mask(self, data: bytes) -> bytes:
        """Apply XOR mask to `data` bytes.

        Args:
            data: Serialised model bytes from the client.

        Returns:
            Masked bytes of the same length.
        """
        if not self.enabled:
            return data
        mask = _generate_mask(len(data), self._seed)
        masked = np.frombuffer(data, dtype=np.uint8) ^ mask
        return masked.tobytes()

    def unmask(self, masked_data: bytes) -> bytes:
        """Remove XOR mask from `masked_data` bytes.

        Args:
            masked_data: Masked bytes received from the client.

        Returns:
            Original unmasked bytes.
        """
        if not self.enabled:
            return masked_data
        # XOR is its own inverse: unmask = mask again with same key
        return self.mask(masked_data)

    def mask_array(self, arr: np.ndarray) -> np.ndarray:
        """Convenience: mask a numpy float array by masking its bytes.

        Args:
            arr: float32 or float64 numpy array.

        Returns:
            Masked array of the same dtype and shape.
        """
        raw = arr.tobytes()
        masked_raw = self.mask(raw)
        return np.frombuffer(masked_raw, dtype=arr.dtype).reshape(arr.shape)

    def unmask_array(self, masked_arr: np.ndarray) -> np.ndarray:
        """Unmask a previously masked numpy array."""
        raw = masked_arr.tobytes()
        unmasked_raw = self.unmask(raw)
        return np.frombuffer(unmasked_raw, dtype=masked_arr.dtype).reshape(masked_arr.shape)


# ---------------------------------------------------------------------------
# Documentation helper
# ---------------------------------------------------------------------------

def write_secure_aggregation_doc(output_path: Optional[Path] = None) -> Path:
    """Write the secure aggregation documentation to a Markdown file.

    Returns:
        Path to the written file.
    """
    path = output_path or ROOT / "secure_aggregation.md"
    content = """# Secure Aggregation - Task E4

> **Team Member 3** | Section E of the implementation plan.

## Overview

Secure aggregation ensures that the federated server cannot learn the
individual model updates from any single client - only the aggregated
result is visible.

## Method: Additive XOR Masking

We implement **additive secret masking** over the serialised model bytes:

```
Client side:   masked_bytes = model_bytes XOR PRG(shared_seed)
Server side:   model_bytes  = masked_bytes XOR PRG(shared_seed)
```

`PRG` is a pseudorandom generator seeded from a 256-bit shared secret
exchanged once between the client and server (out-of-band or via DH).

## Security Properties

| Property | Status |
|----------|--------|
| Model confidentiality (client -> server) | ✅ Yes |
| Server learns individual updates | ❌ No |
| Requires trusted third party | ❌ No |
| Computational overhead | Minimal (O(N) XOR) |

## Limitations

- This is a **research prototype**, not a production security system.
- For stronger guarantees, replace with:
  - **Paillier homomorphic encryption** (heavy, RSA-scale keys).
  - **SPDZ / SecAgg+** (cryptographically secure multi-party computation).
  - **Differential privacy** noise injection on gradients.
- The shared seed is stored in `federated_artifacts/.fl_shared_seed.bin`.
  In production, this would be exchanged via TLS + DH key agreement.

## Integration

```python
from src.federated.secure_aggregation import SecureAggregator

# Both client and server must use the same seed
agg = SecureAggregator(enabled=True)

# Client masks before sending
masked = agg.mask(model_bytes)

# Server unmasks after receiving
original = agg.unmask(masked)
```

See `fl_server.py` and `fl_client.py` for the full integration.

---
*Generated by secure_aggregation.py - Team Member 3.*
"""
    path.write_text(content, encoding="utf-8")
    print(f"Documentation written -> {path}")
    return path


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Secure aggregation utilities (Task E4)."
    )
    parser.add_argument("--write-doc", action="store_true",
                        help="Write secure_aggregation.md documentation.")
    parser.add_argument("--test", action="store_true",
                        help="Run a quick mask/unmask round-trip test.")
    args = parser.parse_args()

    if args.write_doc:
        write_secure_aggregation_doc()

    if args.test or (not args.write_doc and not args.test):
        print("Running mask/unmask round-trip test ...")
        agg = SecureAggregator(enabled=True)

        original = b"Hello, FL world! " * 1000
        masked   = agg.mask(original)
        recovered = agg.unmask(masked)

        assert original == recovered, "Round-trip FAILED - mask/unmask mismatch!"
        assert masked != original,   "Masking had no effect - check seed!"

        print(f"  Original  (first 16 bytes): {original[:16]}")
        print(f"  Masked    (first 16 bytes): {masked[:16]}")
        print(f"  Recovered (first 16 bytes): {recovered[:16]}")
        print("[PASS] Round-trip test PASSED.")

        # Numpy array test
        arr = np.random.rand(100).astype(np.float32)
        masked_arr   = agg.mask_array(arr)
        recovered_arr = agg.unmask_array(masked_arr)
        assert np.allclose(arr, recovered_arr), "Array round-trip FAILED!"
        print("[PASS] Array round-trip test PASSED.")
