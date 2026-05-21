"""
train_baseline.py
─────────────────
Day 2 — Trains a RandomForest baseline classifier on hand-crafted features.
Run this script directly:
    python src/train_baseline.py

What it does:
    1. Loads dataset_v1.jsonl
    2. Extracts hand-crafted + TF-IDF features
    3. Trains RandomForest + Logistic Regression
    4. Evaluates both on test set (F1, precision, recall, ROC-AUC)
    5. Generates SHAP feature importance chart
    6. Saves results to results/baseline_results.md
    7. Saves best model to models/baseline_model.pkl
"""

import sys
import json
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed — saves charts to file
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

from src.data_utils import load_jsonl, train_test_val_split
from src.feature_engineering import FeatureExtractor, HandCraftedFeatures

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = Path("data/processed/dataset_v1.jsonl")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("  Day 2 — Baseline Classifier Training")
print("="*60)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("\n── Step 1: Loading dataset ──")
samples = load_jsonl(DATASET_PATH)
print(f"  Loaded {len(samples):,} samples")

train_s, val_s, test_s = train_test_val_split(samples)

# ── Step 2: Feature extraction ────────────────────────────────────────────────
print("\n── Step 2: Extracting features ──")
fe = FeatureExtractor(tfidf_max_features=5000)
X_train = fe.fit_transform(train_s)
X_val   = fe.transform(val_s)
X_test  = fe.transform(test_s)

y_train = np.array([s.label for s in train_s])
y_val   = np.array([s.label for s in val_s])
y_test  = np.array([s.label for s in test_s])

print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# Save feature extractor
fe.save(MODELS_DIR / "feature_extractor.pkl")

# ── Step 3: Train models ──────────────────────────────────────────────────────
print("\n── Step 3: Training models ──")

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = {}
for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)

    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]

    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "f1": f1, "precision": precision,
        "recall": recall, "roc_auc": roc_auc,
        "y_pred": y_pred, "y_prob": y_prob,
    }

    print(f"  ┌─────────────────────────────┐")
    print(f"  │  {name:<27}│")
    print(f"  │  F1        : {f1:.4f}          │")
    print(f"  │  Precision : {precision:.4f}          │")
    print(f"  │  Recall    : {recall:.4f}          │")
    print(f"  │  ROC-AUC   : {roc_auc:.4f}          │")
    print(f"  └─────────────────────────────┘")

# ── Step 4: Pick best model ───────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["f1"])
best       = results[best_name]
best_model = best["model"]
print(f"\n  Best model: {best_name}  (F1={best['f1']:.4f})")

# Save best model
with open(MODELS_DIR / "baseline_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"  Saved → models/baseline_model.pkl")

# ── Step 5: Confusion matrix ──────────────────────────────────────────────────
print("\n── Step 4: Generating charts ──")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Legit","Phishing"],
            yticklabels=["Legit","Phishing"])
axes[0].set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# ROC curve
RocCurveDisplay.from_predictions(y_test, best["y_prob"], ax=axes[1],
                                  name=best_name, color="#E05C5C")
axes[1].plot([0,1],[0,1],"k--", alpha=0.4)
axes[1].set_title("ROC Curve", fontweight="bold")

plt.tight_layout()
chart_path = RESULTS_DIR / "baseline_confusion_roc.png"
plt.savefig(chart_path, dpi=120, bbox_inches="tight")
print(f"  Saved → {chart_path}")
plt.close()

# ── Step 6: SHAP feature importance (hand-crafted features only) ──────────────
print("\n── Step 5: SHAP feature importance ──")
print("  Computing SHAP values (takes ~30 seconds)...")

# Extract only hand-crafted features for SHAP (TF-IDF is too large to visualise)
hcf = HandCraftedFeatures()
X_train_hc = hcf.transform_batch(train_s)
X_test_hc  = hcf.transform_batch(test_s)

rf_hc = RandomForestClassifier(
    n_estimators=200, random_state=42,
    class_weight="balanced", n_jobs=-1
)
rf_hc.fit(X_train_hc, y_train)

explainer   = shap.TreeExplainer(rf_hc)
shap_values = explainer.shap_values(X_test_hc)

# shap_values is [class0, class1] for RF — take class 1 (phishing)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

fig, ax = plt.subplots(figsize=(9, 5))
feature_names = HandCraftedFeatures.FEATURE_NAMES
mean_shap     = np.abs(sv).mean(axis=0)
sorted_idx    = np.argsort(mean_shap)

colors = ["#E05C5C" if mean_shap[i] > np.median(mean_shap) else "#4A9EBF"
          for i in sorted_idx]

ax.barh([feature_names[i] for i in sorted_idx], mean_shap[sorted_idx],
        color=colors, edgecolor="white")
ax.set_title("SHAP Feature Importance (hand-crafted features)\n"
             "Red = above median impact on phishing prediction",
             fontweight="bold")
ax.set_xlabel("Mean |SHAP value|")
sns.despine()
plt.tight_layout()
shap_path = RESULTS_DIR / "baseline_shap_importance.png"
plt.savefig(shap_path, dpi=120, bbox_inches="tight")
print(f"  Saved → {shap_path}")
plt.close()

# ── Step 7: Save results.md ───────────────────────────────────────────────────
print("\n── Step 6: Writing results.md ──")

top_features = sorted(
    zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True
)[:5]

md = f"""# Baseline Classifier Results

## Models Evaluated

| Model | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
"""
for name, r in results.items():
    md += f"| {name} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['roc_auc']:.4f} |\n"

md += f"""
**Best model:** {best_name} with F1 = {best['f1']:.4f}

## Top 5 Most Important Features (SHAP)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
"""
for i, (feat, val) in enumerate(top_features, 1):
    md += f"| {i} | {feat} | {val:.4f} |\n"

md += f"""
## Classification Report — {best_name}

```
{classification_report(y_test, best['y_pred'], target_names=['Legitimate','Phishing'])}
```

## Charts
- `results/baseline_confusion_roc.png` — Confusion matrix + ROC curve
- `results/baseline_shap_importance.png` — SHAP feature importance

## Key Insights
- **Target to beat on Day 5:** F1 = {best['f1']:.4f} with DistilBERT
- **Strongest feature:** {top_features[0][0]}
- **Dataset:** 3,000 emails (1,500 phishing + 1,500 legit)
"""

results_path = RESULTS_DIR / "baseline_results.md"
results_path.write_text(md)
print(f"  Saved → {results_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DAY 2 COMPLETE ✓")
print("="*60)
print(f"\n  Best model  : {best_name}")
print(f"  F1 Score    : {best['f1']:.4f}")
print(f"  Precision   : {best['precision']:.4f}")
print(f"  Recall      : {best['recall']:.4f}")
print(f"  ROC-AUC     : {best['roc_auc']:.4f}")
print(f"\n  Charts saved to : results/")
print(f"  Model saved to  : models/baseline_model.pkl")
print(f"  Results saved to: results/baseline_results.md")
print(f"\n  Target to beat on Day 5 with DistilBERT: F1 > {best['f1']:.4f}")
print("="*60 + "\n")
