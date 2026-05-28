# Baseline Classifier Results

## Models Evaluated

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| RandomForest | 0.9843 | 0.9910 | 0.9778 | 0.9991 |
| LogisticRegression | 0.9548 | 0.9724 | 0.9378 | 0.9890 |

**Best model:** RandomForest with F1 = 0.9843

## Top 5 Most Important Features (SHAP)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | exclamation_count | 0.1228 |
| 2 | caps_ratio | 0.1079 |
| 3 | urgency_score | 0.1062 |
| 4 | forward_slash_count | 0.0700 |
| 5 | avg_word_length | 0.0429 |

## Classification Report — RandomForest

```
              precision    recall  f1-score   support

  Legitimate       0.98      0.99      0.98       225
    Phishing       0.99      0.98      0.98       225

    accuracy                           0.98       450
   macro avg       0.98      0.98      0.98       450
weighted avg       0.98      0.98      0.98       450

```

## Charts
- `results/baseline_confusion_roc.png` — Confusion matrix + ROC curve
- `results/baseline_shap_importance.png` — SHAP feature importance

## Key Insights
- **Strongest feature:** exclamation_count
- **Dataset:** 3,000 emails (1,500 phishing + 1,500 legit)
