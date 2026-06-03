# Complexity Report — Baseline, Graph, and Spectral Pipeline

> Big-O analysis of the preprocessing, baseline modeling, graph construction, partitioning, and spectral feature extraction pipeline for the CICIoT2023 dataset, corresponding to Sections A, C, and D of the implementation plan.

---

## 1. Overview

The pipeline consists of three major phases:

```
Raw CSV flows
     │
     ▼
[Phase A] Preprocessing & Baseline Modeling 
     │
     ▼
[Phase B] Graph Construction & Partitioning 
     │
     ▼
[Phase C] Spectral Laplacian Feature Extraction 
     │
     ▼
Augmented Features & Final Classification
```

Variables used throughout:

| Symbol | Meaning |
|--------|---------|
| N | Total number of flow records (rows) |
| F | Number of selected features |
| K | Number of virtual device node clusters |
| W | Number of temporal windows = ⌈N / window_size⌉ |
| n_w | Average number of distinct node clusters per window |
| E | Number of edges in the graph (≤ K²/2) |
| P | Number of graph partitions |
| V_p | Average nodes per partition (= K / P) |
| I | Iterations (for KMeans or model training) |
| M | Number of trees (for Random Forest) |

---

## 2. Phase A — Preprocessing & Baseline Modeling (Section A)

### 2.1 Feature Selection (Correlation & Mutual Information)

```
T_corr = O(N · F_raw² + F_raw³)   — Vectorised Pearson Correlation
T_mi   = O(N log N · F_raw)       — Mutual Information 
S      = O(N · F_raw)             — Dataset in memory
```
*Note: F_raw is the original number of features  before selecting the top F.*

### 2.2 Baseline Model Training

**Random Forest (RF):**
```
T_train = O(M · N · F log N)  — Training M decision trees
S_train = O(M · nodes)        — Storing the forest structure
```

**Logistic Regression (LR):**
```
T_train = O(N · F · I)        — I iterations of gradient descent
S_train = O(F)                — Storing model weights
```

### 2.3 Baseline Model Inference

**Random Forest (RF):**
```
T_infer = O(M · depth)        — Traversing trees per sample
```

**Logistic Regression (LR):**
```
T_infer = O(F)                — Single dot product per sample
```

---

## 3. Phase C — Graph-Based Modeling (Section C)

### 3.1 Virtual Node Assignment (MiniBatchKMeans)

```
T_fit     = O(N · K · I)     — mini-batch gradient descent
T_predict = O(N · K)         — single distance pass for label assignment
```

With defaults K=20, I=100, batch=10,000:
- Each mini-batch: 10,000 × 20 distance computations = 200,000 ops.
- Batches per epoch ≈ N / 10,000.
- Total: ≈ N × K × I / batch_size = **Manageable in O(N)**.

### 3.2 Temporal Window Assignment

```
T = O(N)  — vectorised integer division
```

### 3.3 Graph Construction (Edge Accumulation)

```
T = O(W · n_w²)   — for each window, count all node-pair co-occurrences
S = O(E) = O(K²)  — edge weight dictionary
```

**Worst case analysis:**
- W = N / window_size
- n_w ≤ K (all K nodes appear in every window)
- Worst case: T = O(N / window_size · K²)

With defaults (window_size=500, K=20): W = 11,000 windows (for 5.5M records).
n_w² ≤ 400 comparisons per window.
Total ≈ 11,000 × 400 = **4.4M operations** — highly efficient.

### 3.4 Graph Partitioning

**METIS Partitioning (Primary):**
```
T = O(K log K)   — near-linear multilevel partitioning
```

**Spectral Clustering (Fallback):**
```
T_eigen  = O(K³)         — full symmetric eigen-decomposition
T_kmeans = O(K · P · I)  — K-Means on K-dim spectral embeddings
```

Both are negligible relative to graph construction because K ≤ 50 in all practical configurations.

---

## 4. Phase D — Spectral Analysis (Section D)

### 4.1 Laplacian Construction (per partition)

```
T_per_partition = O(V_p²)   — adjacency matrix allocation and fill
T_total         = O(P · V_p²) = O(K²/P)   — since V_p = K/P
```

### 4.2 Eigen-Decomposition (per partition)

```
T_per_partition = O(V_p³)          — numpy.linalg.eigh (symmetric)
T_total         = O(P · V_p³) = O(K³/P²)
```

With K=20, P=4: each partition has V_p=5 nodes → 5³=125 flops each.
Total eigen work ≈ 4 × 125 = **500 flops** — mathematically cheap.

### 4.3 DataFrame Augmentation

```
T = O(N)   — O(N) row iterations to assign spectral columns
S = O(N · k_spectral)   — k_spectral = 2·top_k + 2 new columns
```

---

## 5. End-to-End Pipeline Summary

| Stage | Time Complexity | Space Complexity | Bottleneck? |
|-------|----------------|-----------------|-------------|
| Preprocessing (Phase A)| O(N log N · F) | O(N · F) | Memory/Disk I/O |
| Model Train (Phase A) | O(M · N · F log N) | O(M · nodes) | **Yes (Training)** |
| Node assign (Phase C) | O(N · K · I) | O(N + K·F) | Moderate |
| Graph build (Phase C) | O(W · n_w²) | O(K²) | No |
| Partitioning (Phase C)| O(K³) or O(K log K)| O(K²) | No |
| Spectral (Phase D) | O(P · V_p³) | O(K²) | No |
| Augmentation (Phase D)| O(N) | O(N · k) | Moderate |

---


