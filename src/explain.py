"""
explain.py
──────────
SHAP token-level explainability for the DistilBERT classifier.
Run:
    python src/explain.py

What it does:
    1. Loads the fine-tuned DistilBERT model
    2. Computes SHAP values for sample emails
    3. Renders HTML heatmaps highlighting phishing tokens
    4. Saves 5 example heatmaps to results/heatmaps/
    5. Builds a predict.py inference module used by the Streamlit app
"""

import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.data_utils import load_jsonl

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models/distilbert")
RESULTS_DIR = Path("results/heatmaps")
SPLITS_DIR  = Path("data/processed/splits")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH  = 128

# ── Load model ────────────────────────────────────────────────────────────────
print("── Loading model ──")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODELS_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODELS_DIR)
model.eval()
print("  Model loaded ✓\n")

# ── Gradient-based token attribution ─────────────────────────────────────────
# We use input×gradient attribution — faster than SHAP for transformers
# and produces equally interpretable token-level importance scores.

def get_token_attributions(text: str):
    """
    Returns (tokens, scores, prediction, confidence) for a given email text.
    Scores are per-token importance values (positive = phishing signal).
    """
    enc = tokenizer(
        text, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Get token embeddings with gradient tracking
    embeddings = model.distilbert.embeddings(input_ids)
    embeddings.retain_grad()
    embeddings_input = embeddings.clone().requires_grad_(True)

    # Forward pass through transformer layers manually
    with torch.enable_grad():
        hidden = embeddings_input
        for layer in model.distilbert.transformer.layer:
            hidden = layer(hidden, attn_mask=attention_mask.unsqueeze(1).unsqueeze(1)
                          .expand(-1,-1,-1,hidden.shape[1]).float())[0]

        pooled  = hidden[:, 0]                        # [CLS] token
        pooled  = model.pre_classifier(pooled)
        pooled  = torch.relu(pooled)
        logits  = model.classifier(pooled)
        probs   = F.softmax(logits, dim=-1)

        # Backprop w.r.t. phishing class (index 1)
        probs[0, 1].backward()

    # Attribution = gradient × embedding (summed over hidden dim)
    grads  = embeddings_input.grad[0]                 # (seq_len, hidden)
    embeds = embeddings_input[0].detach()
    attrs  = (grads * embeds).sum(dim=-1)             # (seq_len,)
    attrs  = attrs.detach().numpy()

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Mask padding
    mask       = attention_mask[0].numpy().astype(bool)
    tokens     = [t for t, m in zip(tokens, mask) if m]
    attrs      = attrs[:len(tokens)]

    # Remove special tokens
    clean_tokens = []
    clean_attrs  = []
    for t, a in zip(tokens, attrs):
        if t in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        clean_tokens.append(t)
        clean_attrs.append(float(a))

    pred       = int(probs[0,1].item() > 0.5)
    confidence = float(probs[0, pred].item())

    return clean_tokens, clean_attrs, pred, confidence


def normalize_attrs(attrs):
    """Normalize attribution scores to [-1, 1] range."""
    a = np.array(attrs)
    max_abs = np.abs(a).max()
    if max_abs == 0:
        return a.tolist()
    return (a / max_abs).tolist()


# ── HTML heatmap renderer ─────────────────────────────────────────────────────

def render_heatmap_html(tokens, attrs, pred, confidence, subject="", label=None):
    """
    Render a single email's token attributions as a color-coded HTML block.
    Red tokens = strong phishing signal, blue = legit signal, white = neutral.
    """
    norm_attrs = normalize_attrs(attrs)

    # Reconstruct words from WordPiece tokens (## prefix = continuation)
    words, word_scores = [], []
    current_word, current_score = "", 0.0
    count = 0
    for token, score in zip(tokens, norm_attrs):
        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
            count += 1
        else:
            if current_word:
                words.append(current_word)
                word_scores.append(current_score / max(count, 1))
            current_word = token
            current_score = score
            count = 1
    if current_word:
        words.append(current_word)
        word_scores.append(current_score / max(count, 1))

    def score_to_color(score):
        """Map score in [-1, 1] to CSS rgba color."""
        if score > 0:
            intensity = min(int(score * 220), 220)
            return f"rgba(220,60,60,{intensity/255:.2f})"
        else:
            intensity = min(int(-score * 180), 180)
            return f"rgba(60,130,220,{intensity/255:.2f})"

    verdict_color = "#c0392b" if pred == 1 else "#27ae60"
    verdict_label = "PHISHING" if pred == 1 else "LEGITIMATE"
    actual_color  = "#c0392b" if label == 1 else ("#27ae60" if label == 0 else "#888")
    actual_label  = "Phishing" if label == 1 else ("Legitimate" if label == 0 else "Unknown")

    token_spans = ""
    for word, score in zip(words, word_scores):
        color = score_to_color(score)
        title = f"Attribution: {score:.3f}"
        token_spans += (
            f'<span title="{title}" style="background:{color};padding:2px 4px;'
            f'border-radius:3px;margin:1px;display:inline-block;'
            f'font-family:monospace;font-size:13px;">{word}</span> '
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Phishing Detector — Token Heatmap</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 860px; margin: 40px auto;
            padding: 20px; background: #f9f9f9; }}
    .card {{ background: white; border-radius: 10px; padding: 24px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }}
    .verdict {{ font-size: 22px; font-weight: bold; color: {verdict_color}; }}
    .conf {{ font-size: 14px; color: #666; margin-top: 4px; }}
    .label {{ font-size: 13px; color: {actual_color}; margin-top: 2px; }}
    .subject {{ font-size: 15px; font-weight: bold; margin: 16px 0 8px;
                color: #333; border-left: 3px solid {verdict_color};
                padding-left: 10px; }}
    .body-text {{ line-height: 2.2; }}
    .legend {{ display:flex; gap:20px; margin-top:16px; font-size:12px; color:#555; }}
    .legend-item {{ display:flex; align-items:center; gap:6px; }}
    .swatch {{ width:16px; height:16px; border-radius:3px; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="verdict">Verdict: {verdict_label}</div>
    <div class="conf">Confidence: {confidence*100:.1f}%</div>
    <div class="label">Actual label: {actual_label}</div>
    <div class="subject">Subject: {subject or "(no subject)"}</div>
    <div class="body-text">{token_spans}</div>
    <div class="legend">
      <div class="legend-item">
        <div class="swatch" style="background:rgba(220,60,60,0.7)"></div>
        Strong phishing signal
      </div>
      <div class="legend-item">
        <div class="swatch" style="background:rgba(60,130,220,0.5)"></div>
        Legit signal
      </div>
      <div class="legend-item">
        <div class="swatch" style="background:#eee"></div>
        Neutral
      </div>
    </div>
  </div>
</body>
</html>"""
    return html


# ── Generate example heatmaps ─────────────────────────────────────────────────

def main():
    print("── Loading test samples ──")
    test_s = load_jsonl(SPLITS_DIR / "test.jsonl")

    # Pick 3 phishing + 2 legit samples for heatmaps
    phishing = [s for s in test_s if s.label == 1][:3]
    legit    = [s for s in test_s if s.label == 0][:2]
    examples = phishing + legit

    print(f"  Generating heatmaps for {len(examples)} emails...\n")

    for i, sample in enumerate(examples):
        text   = f"Subject: {sample.subject} {sample.body[:300]}"
        label_name = "phishing" if sample.label == 1 else "legit"

        print(f"  [{i+1}/{len(examples)}] {label_name.upper()} — {sample.subject[:50]}...")

        try:
            tokens, attrs, pred, conf = get_token_attributions(text)

            html = render_heatmap_html(
                tokens=tokens, attrs=attrs,
                pred=pred, confidence=conf,
                subject=sample.subject,
                label=sample.label
            )

            fname = RESULTS_DIR / f"heatmap_{i+1}_{label_name}.html"
            fname.write_text(html, encoding="utf-8")

            verdict = "PHISHING" if pred == 1 else "LEGIT"
            correct = "✓" if pred == sample.label else "✗"
            print(f"     Predicted: {verdict} ({conf*100:.1f}%)  {correct}")
            print(f"     Saved: {fname}\n")

        except Exception as e:
            print(f"     Error: {e}\n")

    # ── Build predict.py for Streamlit ────────────────────────────────────────
    print("── Writing predict.py ──")
    predict_code = '''"""
predict.py
──────────
Inference module used by the Streamlit app.
Import this in app.py to classify any email and get token attributions.

Usage:
    from src.predict import predict_email
    result = predict_email(subject="...", body="...")
    print(result["verdict"], result["confidence"])
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODELS_DIR = Path("models/distilbert")
MAX_LENGTH = 128

_tokenizer = None
_model     = None

def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODELS_DIR)
        _model     = DistilBertForSequenceClassification.from_pretrained(MODELS_DIR)
        _model.eval()

def predict_email(subject: str, body: str) -> dict:
    """
    Classify an email and return prediction + token attributions.

    Returns:
        {
          "verdict":    "Phishing" | "Legitimate",
          "confidence": float (0-1),
          "label":      1 | 0,
          "tokens":     list of str,
          "scores":     list of float (attribution per token),
          "top_tokens": list of (token, score) sorted by |score|
        }
    """
    _load_model()

    text = f"Subject: {subject} {body[:300]}"
    enc  = _tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                      truncation=True, return_tensors="pt")

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    embeddings       = _model.distilbert.embeddings(input_ids)
    embeddings_input = embeddings.clone().requires_grad_(True)

    with torch.enable_grad():
        hidden = embeddings_input
        for layer in _model.distilbert.transformer.layer:
            hidden = layer(hidden,
                           attn_mask=(1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0)[0]
        pooled = torch.relu(_model.pre_classifier(hidden[:, 0]))
        logits = _model.classifier(pooled)
        probs  = F.softmax(logits, dim=-1)
        probs[0, 1].backward()

    grads  = embeddings_input.grad[0]
    embeds = embeddings_input[0].detach()
    attrs  = (grads * embeds).sum(dim=-1).detach().numpy()

    tokens = _tokenizer.convert_ids_to_tokens(input_ids[0])
    mask   = attention_mask[0].numpy().astype(bool)
    tokens = [t for t, m in zip(tokens, mask) if m]
    attrs  = attrs[:len(tokens)]

    clean_tokens, clean_attrs = [], []
    for t, a in zip(tokens, attrs):
        if t not in ("[CLS]", "[SEP]", "[PAD]"):
            clean_tokens.append(t)
            clean_attrs.append(float(a))

    max_abs = max(abs(a) for a in clean_attrs) if clean_attrs else 1
    norm    = [a / max_abs for a in clean_attrs]

    pred       = int(probs[0,1].item() > 0.5)
    confidence = float(probs[0, pred].item())

    top_tokens = sorted(zip(clean_tokens, norm),
                        key=lambda x: abs(x[1]), reverse=True)[:10]

    return {
        "verdict":    "Phishing" if pred == 1 else "Legitimate",
        "confidence": confidence,
        "label":      pred,
        "tokens":     clean_tokens,
        "scores":     norm,
        "top_tokens": top_tokens,
    }


if __name__ == "__main__":
    # Quick test
    result = predict_email(
        subject="URGENT: Your account has been suspended",
        body="Dear customer, click here immediately to verify your account or it will be closed."
    )
    print(f"Verdict    : {result['verdict']}")
    print(f"Confidence : {result['confidence']*100:.1f}%")
    print(f"Top tokens : {result['top_tokens'][:5]}")
'''

    predict_path = Path("src/predict.py")
    predict_path.write_text(predict_code)
    print(f"  Saved → {predict_path}")

    # Quick test of predict.py
    print("\n── Testing predict.py ──")
    from src.predict import predict_email
    r = predict_email(
        subject="URGENT: Your PayPal account has been suspended",
        body="Dear customer, click here immediately to verify your account or it will be permanently closed."
    )
    print(f"  Test email verdict    : {r['verdict']} ({r['confidence']*100:.1f}%)")
    print(f"  Top phishing tokens   : {[t for t,s in r['top_tokens'][:5]]}")

    print(f"  Heatmaps saved : results/heatmaps/ ({len(examples)} files)")
    print(f"  predict.py     : src/predict.py")
    print(f"\n  Open any .html file in results/heatmaps/ in your browser")
    print(f"  to see the token attribution heatmap.")

if __name__ == "__main__":
    main()
