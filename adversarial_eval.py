"""
adversarial_eval.py
───────────────────
Feed all 798 synthetic phishing emails through the detector.
Measures what % the model catches, finds evasion patterns, retrains on hard negatives.

Run: python src/adversarial_eval.py
"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.data_utils import load_jsonl, save_jsonl
from src.predict import predict_email

SYNTHETIC_PATH = Path("data/processed/synthetic_phishing.jsonl")
RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load synthetic phishing emails
synth = load_jsonl(SYNTHETIC_PATH)
print(f"  Testing {len(synth)} synthetic phishing emails against detector...\n")

results     = []
by_lure     = {}
false_neg   = []   # phishing emails the model missed

for i, s in enumerate(synth):
    r = predict_email(s.subject, s.body)
    correct = (r["label"] == s.label)
    results.append({"sample": s, "pred": r["label"],
                    "conf": r["confidence"], "correct": correct,
                    "top_tokens": r["top_tokens"]})

    lure = s.lure_type or "unknown"
    if lure not in by_lure:
        by_lure[lure] = {"total": 0, "caught": 0}
    by_lure[lure]["total"] += 1
    if correct:
        by_lure[lure]["caught"] += 1
    else:
        false_neg.append({"subject": s.subject, "body": s.body[:300],
                          "lure_type": lure, "confidence": r["confidence"],
                          "top_tokens": r["top_tokens"]})

    if (i+1) % 100 == 0:
        caught = sum(1 for r in results if r["correct"])
        print(f"  Progress: {i+1}/{len(synth)}  "
              f"Detection rate: {caught/(i+1)*100:.1f}%", flush=True)

# Overall stats
total   = len(results)
caught  = sum(1 for r in results if r["correct"])
missed  = total - caught
det_rate = caught / total * 100

print(f"\n{'='*60}")
print(f"  ADVERSARIAL EVALUATION RESULTS")
print(f"{'='*60}")
print(f"  Total synthetic phishing : {total}")
print(f"  Caught by detector       : {caught}  ({det_rate:.1f}%)")
print(f"  Evaded detector          : {missed}  ({100-det_rate:.1f}%)")

print(f"\n  Detection rate by lure type:")
for lure, stats in sorted(by_lure.items(), key=lambda x: x[1]["caught"]/x[1]["total"]):
    rate = stats["caught"]/stats["total"]*100
    bar  = "█" * int(rate/5)
    print(f"  {lure:<22} {rate:5.1f}%  {bar}")

# Analyze false negatives
print(f"\n  False negatives (emails that evaded): {len(false_neg)}")
if false_neg:
    print(f"\n  Sample evaded emails:")
    for fn in false_neg[:3]:
        print(f"    Lure: {fn['lure_type']}")
        print(f"    Subject: {fn['subject'][:60]}")
        print(f"    Conf (legit): {fn['confidence']*100:.1f}%")
        print(f"    Top tokens: {[t for t,s in fn['top_tokens'][:4]]}\n")

# Save false negatives for retraining
fn_path = RESULTS_DIR / "false_negatives.json"
fn_path.write_text(json.dumps(false_neg, indent=2), encoding="utf-8")
print(f"  False negatives saved -> {fn_path}")

# Charts
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Detection rate by lure
lures = list(by_lure.keys())
rates = [by_lure[l]["caught"]/by_lure[l]["total"]*100 for l in lures]
colors = ["#27ae60" if r >= 90 else "#e67e22" if r >= 70 else "#c0392b" for r in rates]
axes[0].barh(lures, rates, color=colors, edgecolor="white")
axes[0].axvline(x=det_rate, color="navy", linestyle="--", alpha=0.7, label=f"Overall: {det_rate:.1f}%")
axes[0].set_xlim(0, 105)
axes[0].set_title("Detection Rate by Lure Type", fontweight="bold")
axes[0].set_xlabel("% Caught")
axes[0].legend(fontsize=9)
sns.despine(ax=axes[0])

# Confidence distribution
caught_conf = [r["conf"] for r in results if r["correct"]]
missed_conf = [r["conf"] for r in results if not r["correct"]]
if caught_conf:
    axes[1].hist(caught_conf, bins=20, alpha=0.7, color="#27ae60", label=f"Caught ({len(caught_conf)})")
if missed_conf:
    axes[1].hist(missed_conf, bins=20, alpha=0.7, color="#c0392b", label=f"Evaded ({len(missed_conf)})")
axes[1].set_title("Confidence Distribution", fontweight="bold")
axes[1].set_xlabel("Model confidence")
axes[1].set_ylabel("Count")
axes[1].legend()
sns.despine(ax=axes[1])

plt.tight_layout()
chart_path = RESULTS_DIR / "adversarial_eval.png"
plt.savefig(chart_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Chart saved -> {chart_path}")

# Save results.md
md = f"""# Adversarial Evaluation Results

## Summary
| Metric | Value |
|---|---|
| Total synthetic phishing tested | {total} |
| Caught by DistilBERT detector | {caught} ({det_rate:.1f}%) |
| Evaded detector | {missed} ({100-det_rate:.1f}%) |

## Detection Rate by Lure Type
| Lure Type | Caught | Total | Rate |
|---|---|---|---|
"""
for lure, stats in sorted(by_lure.items(), key=lambda x: -x[1]["caught"]/x[1]["total"]):
    rate = stats["caught"]/stats["total"]*100
    md += f"| {lure} | {stats['caught']} | {stats['total']} | {rate:.1f}% |\n"

md += f"""
## Key Findings
- Overall adversarial detection rate: **{det_rate:.1f}%**
- False negatives saved to: `results/false_negatives.json`
- These will be used as hard negatives for retraining

## What evasion patterns were found?
Analyze `results/false_negatives.json` to identify common patterns
in emails that evaded the detector (formal tone, fewer urgency words, etc.)
"""
(RESULTS_DIR / "adversarial_results.md").write_text(md, encoding="utf-8")

print(f"  Detection rate  : {det_rate:.1f}%")
print(f"  Evaded          : {missed} emails")