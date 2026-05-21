"""
prepare_dataset.py
──────────────────
Day 4 — Merges real + synthetic emails, runs quality checks,
and produces the final dataset_v2.jsonl ready for DistilBERT.

Run:
    python src/prepare_dataset.py

What it does:
    1. Loads dataset_v1.jsonl (real emails)
    2. Loads synthetic_phishing.jsonl (generated emails)
    3. Runs quality checks and stats comparison
    4. Merges into dataset_v2.jsonl
    5. Saves train/val/test splits as separate files
    6. Prints full summary
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_utils import (
    EmailSample, load_jsonl, save_jsonl,
    train_test_val_split, samples_to_df
)

# ── Config ────────────────────────────────────────────────────────────────────
V1_PATH        = Path("data/processed/dataset_v1.jsonl")
SYNTHETIC_PATH = Path("data/processed/synthetic_phishing.jsonl")
V2_PATH        = Path("data/processed/dataset_v2.jsonl")
SPLITS_DIR     = Path("data/processed/splits")
RESULTS_DIR    = Path("results")

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("  Day 4 — Dataset Merge & Preparation")
print("="*60)

# ── Step 1: Load both datasets ────────────────────────────────────────────────
print("\n── Step 1: Loading datasets ──")
real_samples      = load_jsonl(V1_PATH)
synthetic_samples = load_jsonl(SYNTHETIC_PATH)

real_phishing = [s for s in real_samples if s.label == 1]
real_legit    = [s for s in real_samples if s.label == 0]

print(f"  Real dataset     : {len(real_samples):,} total")
print(f"    ├─ Phishing    : {len(real_phishing):,}")
print(f"    └─ Legitimate  : {len(real_legit):,}")
print(f"  Synthetic phishing: {len(synthetic_samples):,}")

# ── Step 2: Quality check on synthetic emails ─────────────────────────────────
print("\n── Step 2: Quality check on synthetic emails ──")

def quality_stats(samples, name):
    word_counts  = [len(s.body.split()) for s in samples]
    link_counts  = [len(re.findall(r'https?://', s.body)) for s in samples]
    has_subject  = sum(1 for s in samples if len(s.subject) > 3)
    has_cta      = sum(1 for s in samples
                       if any(w in s.body.lower()
                              for w in ['http','click','verify','confirm','access','reset']))

    print(f"\n  {name}:")
    print(f"    Count          : {len(samples):,}")
    print(f"    Avg body words : {np.mean(word_counts):.0f}")
    print(f"    Min body words : {np.min(word_counts)}")
    print(f"    Has subject    : {has_subject}/{len(samples)}")
    print(f"    Has CTA link   : {has_cta}/{len(samples)}")
    print(f"    Avg links/email: {np.mean(link_counts):.2f}")
    return word_counts

real_wc  = quality_stats(real_phishing,    "Real phishing")
synth_wc = quality_stats(synthetic_samples,"Synthetic phishing")
legit_wc = quality_stats(real_legit,       "Real legitimate")

# ── Step 3: Comparison chart ──────────────────────────────────────────────────
print("\n── Step 3: Generating comparison chart ──")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Word count distribution comparison
for data, color, label in [
    (real_wc,  "#E05C5C", "Real phishing"),
    (synth_wc, "#F0A500", "Synthetic phishing"),
    (legit_wc, "#4A9EBF", "Real legitimate"),
]:
    axes[0].hist(
        [min(x, 600) for x in data],
        bins=40, alpha=0.5, color=color,
        label=label, edgecolor="white"
    )
axes[0].set_title("Body Length Distribution\nReal vs Synthetic", fontweight="bold")
axes[0].set_xlabel("Word count (capped at 600)")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=8)
sns.despine(ax=axes[0])

# Lure type breakdown
lure_counts = Counter(s.lure_type for s in synthetic_samples)
lures  = list(lure_counts.keys())
counts = list(lure_counts.values())
colors = ["#E05C5C","#F0A500","#4A9EBF","#5CB85C","#9B59B6","#E67E22"]
axes[1].barh(lures, counts, color=colors[:len(lures)], edgecolor="white")
axes[1].set_title("Synthetic Emails by Lure Type", fontweight="bold")
axes[1].set_xlabel("Count")
sns.despine(ax=axes[1])

plt.tight_layout()
chart_path = RESULTS_DIR / "dataset_v2_comparison.png"
plt.savefig(chart_path, dpi=120, bbox_inches="tight")
print(f"  Saved → {chart_path}")
plt.close()

# ── Step 4: Merge datasets ────────────────────────────────────────────────────
print("\n── Step 4: Merging datasets ──")

# Strategy:
# - Keep all real legitimate emails (1,500)
# - Keep all real phishing emails (1,500)
# - Add all synthetic phishing (798)
# - Final: 1,500 legit vs 2,298 phishing
# - For balanced training: we'll use class weights in DistilBERT

import random
random.seed(42)

all_samples = real_legit + real_phishing + synthetic_samples
random.shuffle(all_samples)

total_phishing = sum(1 for s in all_samples if s.label == 1)
total_legit    = sum(1 for s in all_samples if s.label == 0)

print(f"  Merged dataset:")
print(f"    Total          : {len(all_samples):,}")
print(f"    Phishing       : {total_phishing:,} ({total_phishing/len(all_samples)*100:.1f}%)")
print(f"    Legitimate     : {total_legit:,} ({total_legit/len(all_samples)*100:.1f}%)")
print(f"    Sources        : real_spam, real_ham, synthetic")

save_jsonl(all_samples, V2_PATH)

# ── Step 5: Create train/val/test splits ──────────────────────────────────────
print("\n── Step 5: Creating splits ──")
train_s, val_s, test_s = train_test_val_split(all_samples)

save_jsonl(train_s, SPLITS_DIR / "train.jsonl")
save_jsonl(val_s,   SPLITS_DIR / "val.jsonl")
save_jsonl(test_s,  SPLITS_DIR / "test.jsonl")

print(f"\n  Train : {len(train_s):,}  "
      f"(phishing: {sum(1 for s in train_s if s.label==1):,}  "
      f"legit: {sum(1 for s in train_s if s.label==0):,})")
print(f"  Val   : {len(val_s):,}  "
      f"(phishing: {sum(1 for s in val_s if s.label==1):,}  "
      f"legit: {sum(1 for s in val_s if s.label==0):,})")
print(f"  Test  : {len(test_s):,}  "
      f"(phishing: {sum(1 for s in test_s if s.label==1):,}  "
      f"legit: {sum(1 for s in test_s if s.label==0):,})")

# ── Step 6: Save dataset card ─────────────────────────────────────────────────
print("\n── Step 6: Saving dataset card ──")

lure_breakdown = "\n".join(
    f"| {lure} | {count} |"
    for lure, count in sorted(lure_counts.items(), key=lambda x: -x[1])
)

card = f"""# Dataset Card — v2

## Summary
| Split | Total | Phishing | Legitimate |
|---|---|---|---|
| Train | {len(train_s):,} | {sum(1 for s in train_s if s.label==1):,} | {sum(1 for s in train_s if s.label==0):,} |
| Val   | {len(val_s):,} | {sum(1 for s in val_s if s.label==1):,} | {sum(1 for s in val_s if s.label==0):,} |
| Test  | {len(test_s):,} | {sum(1 for s in test_s if s.label==1):,} | {sum(1 for s in test_s if s.label==0):,} |
| **Total** | **{len(all_samples):,}** | **{total_phishing:,}** | **{total_legit:,}** |

## Sources
- Real phishing : SpamAssassin corpus (1,500 emails)
- Real legitimate: SpamAssassin ham corpus (1,500 emails)
- Synthetic      : Groq Llama-3.1 generated (798 emails)

## Synthetic Lure Types
| Lure Type | Count |
|---|---|
{lure_breakdown}

## Files
- `data/processed/dataset_v2.jsonl` — full merged dataset
- `data/processed/splits/train.jsonl`
- `data/processed/splits/val.jsonl`
- `data/processed/splits/test.jsonl`
- `results/dataset_v2_comparison.png` — distribution charts
"""

card_path = Path("data/processed/DATASET_CARD.md")
card_path.write_text(card)
print(f"  Saved → {card_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DAY 4 COMPLETE ✓")
print("="*60)
print(f"\n  dataset_v2.jsonl  : {len(all_samples):,} emails")
print(f"  Train split       : {len(train_s):,}")
print(f"  Val split         : {len(val_s):,}")
print(f"  Test split        : {len(test_s):,}")
print(f"\n  Ready for Day 5 — DistilBERT fine-tuning")
print(f"  Run: python src/train_distilbert.py")
print("="*60 + "\n")
