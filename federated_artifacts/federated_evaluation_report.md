# Federated Model Evaluation Report

> **Task E5 - Team Member 3**
> Final evaluation of the federated global model on the **frozen test set**.

---

## Global Model Configuration

| Parameter | Value |
|-----------|-------|
| Model path | `C:\Users\Suhas Raghavendra\Desktop\Main EL\federated_artifacts\global_model_final.pkl` |
| Trees (pooled) | 100 |
| Features | 17 |
| Test samples | 1,176,851 |

---

## Test Set Performance

| Metric | Federated Global Model |
|--------|----------------------|
| **Accuracy** | 0.9931 |
| **Binary F1** | 0.9965 |
| **Macro F1** | 0.9247 |
| **ROC-AUC** | 0.9983 |
| **PR-AUC** | 1.0000 |

---

## Confusion Matrix

| | Pred Benign | Pred Attack |
|---|---|---|
| **True Benign** | 23,601 | 4,108 |
| **True Attack** | 4,032 | 1,145,110 |

---

## Classification Report

```
              precision    recall  f1-score   support

           0       0.85      0.85      0.85     27709
           1       1.00      1.00      1.00   1149142

    accuracy                           0.99   1176851
   macro avg       0.93      0.92      0.92   1176851
weighted avg       0.99      0.99      0.99   1176851

```

## Federated Training Round Progression

| Round | Global Accuracy | Global F1 | Clients |
|-------|----------------|-----------|--------|
| 1 | 0.9882 | 0.9939 | 2 |
| 2 | 0.9979 | 0.9989 | 2 |

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
print(f"  Accuracy : {report['accuracy']:.4f}")
print(f"  F1       : {report['f1']:.4f}")
print(f"  ROC-AUC  : {report['roc_auc']:.4f}")
```

---
*Generated automatically by `federated_evaluation.py` - Team Member 3.*
