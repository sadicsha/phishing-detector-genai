"""
train_distilbert.py (fixed — CPU optimized)
────────────────────────────────────────────
Reduced batch size + token length to run on CPU without memory errors.
"""

import sys, json, warnings, traceback
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from src.data_utils import load_jsonl

# ── Config (CPU-safe) ─────────────────────────────────────────────────────────
SPLITS_DIR  = Path("data/processed/splits")
MODELS_DIR  = Path("models/distilbert")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128    # reduced from 256
BATCH_SIZE  = 8      # reduced from 16
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device("cpu")

print(f"\n{'='*60}")
print(f"  Day 5 — DistilBERT Fine-tuning (CPU optimized)")
print(f"{'='*60}")
print(f"  Batch:{BATCH_SIZE}  MaxLen:{MAX_LENGTH}  Epochs:{EPOCHS}", flush=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class EmailDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples   = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        text = f"Subject: {s.subject} {s.body[:300]}"
        enc  = self.tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                              truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(s.label, dtype=torch.long),
        }

def evaluate(model, loader):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs = torch.softmax(out.logits, dim=-1)[:,1]
            preds = torch.argmax(out.logits, dim=-1)
            labels_all.extend(batch["label"].numpy())
            preds_all.extend(preds.numpy())
            probs_all.extend(probs.numpy())
    y, p, pr = np.array(labels_all), np.array(preds_all), np.array(probs_all)
    return {"f1": f1_score(y,p), "precision": precision_score(y,p),
            "recall": recall_score(y,p), "roc_auc": roc_auc_score(y,pr),
            "labels": y, "preds": p, "probs": pr}

def main():
    try:
        # Load data
        print("\n── Loading splits ──", flush=True)
        train_s = load_jsonl(SPLITS_DIR/"train.jsonl")
        val_s   = load_jsonl(SPLITS_DIR/"val.jsonl")
        test_s  = load_jsonl(SPLITS_DIR/"test.jsonl")
        print(f"  Train:{len(train_s)}  Val:{len(val_s)}  Test:{len(test_s)}", flush=True)

        # Tokenizer
        print("\n── Loading tokenizer ──", flush=True)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        print("  Done", flush=True)

        # Datasets
        train_loader = DataLoader(EmailDataset(train_s, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(EmailDataset(val_s,   tokenizer), batch_size=BATCH_SIZE)
        test_loader  = DataLoader(EmailDataset(test_s,  tokenizer), batch_size=BATCH_SIZE)
        print(f"  DataLoaders ready — {len(train_loader)} train batches", flush=True)

        # Class weights
        n_p = sum(1 for s in train_s if s.label==1)
        n_l = sum(1 for s in train_s if s.label==0)
        w   = torch.tensor([len(train_s)/(2*n_l), len(train_s)/(2*n_p)], dtype=torch.float)
        print(f"  Class weights: legit={w[0]:.3f} phishing={w[1]:.3f}", flush=True)

        # Model
        print("\n── Loading model ──", flush=True)
        model   = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        optim   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sched   = get_linear_schedule_with_warmup(optim, int(len(train_loader)*EPOCHS*0.1), len(train_loader)*EPOCHS)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        print("  Model ready", flush=True)

        # Training
        print(f"\n── Training ({EPOCHS} epochs, {len(train_loader)} steps/epoch) ──", flush=True)
        history     = {"loss":[], "val_f1":[]}
        best_val_f1 = 0.0

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(train_loader):
                optim.zero_grad()
                out  = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = loss_fn(out.logits, batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step()
                epoch_loss += loss.item()

                if (step+1) % 30 == 0:
                    print(f"  Epoch {epoch+1}/{EPOCHS}  Step {step+1}/{len(train_loader)}  "
                          f"Loss:{epoch_loss/(step+1):.4f}", flush=True)

            vm = evaluate(model, val_loader)
            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)
            history["val_f1"].append(vm["f1"])
            print(f"\n  Epoch {epoch+1} done — Loss:{avg_loss:.4f}  Val F1:{vm['f1']:.4f}  "
                  f"Prec:{vm['precision']:.4f}  Recall:{vm['recall']:.4f}\n", flush=True)

            if vm["f1"] > best_val_f1:
                best_val_f1 = vm["f1"]
                model.save_pretrained(MODELS_DIR)
                tokenizer.save_pretrained(MODELS_DIR)
                print(f"  ✓ Best model saved (Val F1={best_val_f1:.4f})\n", flush=True)

        # Test evaluation
        print("── Test evaluation ──", flush=True)
        best_model = DistilBertForSequenceClassification.from_pretrained(MODELS_DIR)
        tm = evaluate(best_model, test_loader)

        print(f"\n  F1:{tm['f1']:.4f}  Precision:{tm['precision']:.4f}  "
              f"Recall:{tm['recall']:.4f}  AUC:{tm['roc_auc']:.4f}", flush=True)

        # Charts
        fig, axes = plt.subplots(1, 3, figsize=(15,4))
        axes[0].plot(range(1,EPOCHS+1), history["loss"], "o-r", label="Loss")
        ax2 = axes[0].twinx()
        ax2.plot(range(1,EPOCHS+1), history["val_f1"], "s--b", label="Val F1")
        axes[0].set_title("Training Curve", fontweight="bold")
        axes[0].set_xlabel("Epoch")

        sns.heatmap(confusion_matrix(tm["labels"],tm["preds"]), annot=True, fmt="d",
                    cmap="Blues", ax=axes[1],
                    xticklabels=["Legit","Phishing"], yticklabels=["Legit","Phishing"])
        axes[1].set_title("Confusion Matrix", fontweight="bold")

        baseline = [0.9843, 0.9910, 0.9778, 0.9991]
        distil   = [tm["f1"], tm["precision"], tm["recall"], tm["roc_auc"]]
        x = np.arange(4); w2 = 0.35
        axes[2].bar(x-w2/2, baseline, w2, label="RandomForest", color="#888780", alpha=0.8)
        axes[2].bar(x+w2/2, distil,   w2, label="DistilBERT",   color="#4A9EBF", alpha=0.8)
        axes[2].set_xticks(x); axes[2].set_xticklabels(["F1","Prec","Recall","AUC"])
        axes[2].set_ylim(0.85, 1.01); axes[2].legend()
        axes[2].set_title("Baseline vs DistilBERT", fontweight="bold")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR/"distilbert_results.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("  Chart saved → results/distilbert_results.png", flush=True)

        # Save results
        report = classification_report(tm["labels"], tm["preds"],
                                       target_names=["Legitimate","Phishing"])
        md = f"""# DistilBERT Results\n\n| Metric | RandomForest | DistilBERT | Delta |\n|---|---|---|---|\n| F1 | 0.9843 | {tm['f1']:.4f} | {tm['f1']-0.9843:+.4f} |\n| Precision | 0.9910 | {tm['precision']:.4f} | {tm['precision']-0.9910:+.4f} |\n| Recall | 0.9778 | {tm['recall']:.4f} | {tm['recall']-0.9778:+.4f} |\n| ROC-AUC | 0.9991 | {tm['roc_auc']:.4f} | {tm['roc_auc']-0.9991:+.4f} |\n\n```\n{report}\n```\n"""
        (RESULTS_DIR/"distilbert_results.md").write_text(md)

        print(f"\n{'='*60}")
        print(f"  DAY 5 COMPLETE ✓")
        print(f"{'='*60}")
        print(f"  DistilBERT F1 : {tm['f1']:.4f}")
        print(f"  Baseline F1   : 0.9843")
        print(f"  Model saved   : models/distilbert/")
        print(f"  Next          : python src/explain.py")
        print(f"{'='*60}\n", flush=True)

    except Exception as e:
        print(f"\n✗ ERROR: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
