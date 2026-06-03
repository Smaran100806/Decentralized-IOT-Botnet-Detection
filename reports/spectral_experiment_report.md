# Spectral Feature Experiment Report

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset split | `validation` |
| Virtual device nodes (K) | 20 |
| Temporal window size | 500 rows |
| Graph partitions | 4 |
| Spectral features (top-k) | 8 |
| RF n_estimators | 100 |
| Laplacian type | Normalized (Lₙ = I − D⁻¹ᐟ² A D⁻¹ᐟ²) |
| Random seed | 42 |

---

## Results Summary

| Metric | Baseline RF (17 features) | Spectral RF (35 features) | Δ (Spectral − Baseline) |
|--------|---------|---------|------|
| **Accuracy** | 0.9928 | 0.9925 | -0.0003 |
| **Macro F1** | 0.9208 | 0.9181 | -0.0027 |
| **Binary F1** | 0.9963 | 0.9961 | -0.0001 |
| **ROC-AUC** | 0.9980 | 0.9980 | +0.0000 |
| **PR-AUC** | 0.9999 | 1.0000 | +0.0000 |
| Training time (s) | 19.1 | 20.0 | — |

---

## Baseline RF — Detailed Results

### Confusion Matrix

| | Pred Benign | Pred Attack |
|---|---|---|
| **True Benign** | 4,661 | 843 |
| **True Attack** | 863 | 229,004 |

### Classification Report

```
              precision    recall  f1-score   support

           0       0.84      0.85      0.85      5504
           1       1.00      1.00      1.00    229867

    accuracy                           0.99    235371
   macro avg       0.92      0.92      0.92    235371
weighted avg       0.99      0.99      0.99    235371

```

### Top-10 Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `rst_count` | 0.2754 |
| 2 | `urg_count` | 0.2164 |
| 3 | `Variance` | 0.1982 |
| 4 | `Tot size` | 0.0755 |
| 5 | `AVG` | 0.0586 |
| 6 | `Covariance` | 0.0487 |
| 7 | `Header_Length` | 0.0377 |
| 8 | `Max` | 0.0356 |
| 9 | `Duration` | 0.0185 |
| 10 | `ack_flag_number` | 0.0119 |

---

## Spectral RF — Detailed Results

### Confusion Matrix

| | Pred Benign | Pred Attack |
|---|---|---|
| **True Benign** | 4,648 | 856 |
| **True Attack** | 915 | 228,952 |

### Classification Report

```
              precision    recall  f1-score   support

           0       0.84      0.84      0.84      5504
           1       1.00      1.00      1.00    229867

    accuracy                           0.99    235371
   macro avg       0.92      0.92      0.92    235371
weighted avg       0.99      0.99      0.99    235371

```

### Top-10 Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `urg_count` | 0.1927 |
| 2 | `rst_count` | 0.1805 |
| 3 | `Tot size` | 0.1194 |
| 4 | `Variance` | 0.1078 |
| 5 | `spectral_proj_6` | 0.0514 |
| 6 | `Covariance` | 0.0437 |
| 7 | `Duration` | 0.0433 |
| 8 | `spectral_proj_7` | 0.0371 |
| 9 | `Max` | 0.0365 |
| 10 | `Header_Length` | 0.0348 |

---

## Analysis

### Spectral Feature Contribution
The spectral augmentation adds **18 new features** derived from the Graph Laplacian eigen-decomposition:
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


