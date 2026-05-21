import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.data_utils import load_jsonl

MODELS_DIR  = Path("models/distilbert")
RESULTS_DIR = Path("results/heatmaps")
SPLITS_DIR  = Path("data/processed/splits")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_LENGTH  = 128

print("\n" + "="*60)
print("  Day 6 - Explainability & Heatmaps")
print("="*60 + "\n")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODELS_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODELS_DIR)
model.eval()
print("  Model loaded\n")

def get_attributions(text):
    enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    # Hook to capture embeddings
    captured = {}
    def hook(module, inp, out):
        captured["emb"] = out
    h = model.distilbert.embeddings.register_forward_hook(hook)

    # Forward pass
    with torch.no_grad():
        out   = model(**enc)
        probs = F.softmax(out.logits, dim=-1)
        pred  = int(probs[0,1].item() > 0.5)
        conf  = float(probs[0,pred].item())
    h.remove()

    # Now do gradient pass on embeddings
    emb = captured["emb"].clone().detach().requires_grad_(True)

    # Replace embedding layer output with our tracked version
    def hook2(module, inp, out):
        return emb
    h2 = model.distilbert.embeddings.register_forward_hook(hook2)

    out2   = model(**enc)
    probs2 = F.softmax(out2.logits, dim=-1)
    probs2[0,1].backward()
    h2.remove()

    attrs  = (emb.grad[0] * emb[0]).sum(dim=-1).detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    m      = mask[0].numpy().astype(bool)
    tokens = [t for t,v in zip(tokens,m) if v]
    attrs  = attrs[:len(tokens)]

    ct, ca = [], []
    for t,a in zip(tokens, attrs):
        if t not in ("[CLS]","[SEP]","[PAD]"):
            ct.append(t); ca.append(float(a))
    return ct, ca, pred, conf

def normalize(attrs):
    a = np.array(attrs)
    m = np.abs(a).max()
    return (a/m).tolist() if m > 0 else a.tolist()

def render_html(tokens, attrs, pred, conf, subject="", label=None):
    norm = normalize(attrs)
    words, scores = [], []
    cw, cs, cnt = "", 0.0, 0
    for t,s in zip(tokens, norm):
        if t.startswith("##"):
            cw += t[2:]; cs += s; cnt += 1
        else:
            if cw: words.append(cw); scores.append(cs/max(cnt,1))
            cw, cs, cnt = t, s, 1
    if cw: words.append(cw); scores.append(cs/max(cnt,1))

    def color(s):
        if s > 0: return f"rgba(220,60,60,{min(s*0.9,0.9):.2f})"
        return f"rgba(60,130,220,{min(-s*0.7,0.7):.2f})"

    spans = " ".join(
        f'<span title="{s:.3f}" style="background:{color(s)};padding:2px 5px;'
        f'border-radius:3px;margin:1px;display:inline-block;font-size:13px;">{w}</span>'
        for w,s in zip(words,scores)
    )
    vc = "#c0392b" if pred==1 else "#27ae60"
    vl = "PHISHING" if pred==1 else "LEGITIMATE"
    al = "Phishing" if label==1 else ("Legitimate" if label==0 else "?")
    ac = "#c0392b" if label==1 else "#27ae60"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{font-family:Arial;max-width:860px;margin:40px auto;padding:20px;background:#f9f9f9}}
.card{{background:white;border-radius:10px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.verdict{{font-size:22px;font-weight:bold;color:{vc}}}
.conf{{font-size:13px;color:#666;margin:4px 0}}
.subj{{font-size:15px;font-weight:bold;margin:16px 0 10px;border-left:3px solid {vc};padding-left:10px}}
.body{{line-height:2.6}}
.legend{{display:flex;gap:20px;margin-top:16px;font-size:12px;color:#555}}
.li{{display:flex;align-items:center;gap:6px}}
.sw{{width:16px;height:16px;border-radius:3px}}</style></head>
<body><div class="card">
<div class="verdict">Verdict: {vl}</div>
<div class="conf">Confidence: {conf*100:.1f}% &nbsp;|&nbsp; Actual: <span style="color:{ac}">{al}</span></div>
<div class="subj">Subject: {subject or "(no subject)"}</div>
<div class="body">{spans}</div>
<div class="legend">
<div class="li"><div class="sw" style="background:rgba(220,60,60,.7)"></div>Phishing signal</div>
<div class="li"><div class="sw" style="background:rgba(60,130,220,.5)"></div>Legit signal</div>
</div></div></body></html>"""

# Generate heatmaps
test_s   = load_jsonl(SPLITS_DIR/"test.jsonl")
phishing = [s for s in test_s if s.label==1][:3]
legit    = [s for s in test_s if s.label==0][:2]
examples = phishing + legit
print(f"-- Generating {len(examples)} heatmaps --\n")

for i,s in enumerate(examples):
    lname = "phishing" if s.label==1 else "legit"
    print(f"  [{i+1}/{len(examples)}] {lname.upper()} -- {s.subject[:50]}...")
    try:
        tokens, attrs, pred, conf = get_attributions(f"Subject: {s.subject} {s.body[:300]}")
        html  = render_html(tokens, attrs, pred, conf, s.subject, s.label)
        fpath = RESULTS_DIR/f"heatmap_{i+1}_{lname}.html"
        fpath.write_text(html, encoding="utf-8")
        ok = "correct" if pred==s.label else "WRONG"
        print(f"     -> {'PHISHING' if pred==1 else 'LEGIT'} {conf*100:.1f}%  [{ok}]")
        print(f"     saved: {fpath}\n")
    except Exception as e:
        import traceback; traceback.print_exc()

# Write predict.py
predict_src = '''import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODELS_DIR = Path("models/distilbert")
MAX_LENGTH = 128
_tok = None
_mdl = None

def _load():
    global _tok, _mdl
    if _mdl is None:
        _tok = DistilBertTokenizerFast.from_pretrained(MODELS_DIR)
        _mdl = DistilBertForSequenceClassification.from_pretrained(MODELS_DIR)
        _mdl.eval()

def predict_email(subject, body):
    _load()
    text = f"Subject: {subject} {body[:300]}"
    enc  = _tok(text, max_length=MAX_LENGTH, padding="max_length",
                truncation=True, return_tensors="pt")

    captured = {}
    def hook(module, inp, out): captured["emb"] = out
    h = _mdl.distilbert.embeddings.register_forward_hook(hook)
    with torch.no_grad():
        out   = _mdl(**enc)
        probs = F.softmax(out.logits, dim=-1)
        pred  = int(probs[0,1].item() > 0.5)
        conf  = float(probs[0,pred].item())
    h.remove()

    emb = captured["emb"].clone().detach().requires_grad_(True)
    def hook2(module, inp, out): return emb
    h2 = _mdl.distilbert.embeddings.register_forward_hook(hook2)
    out2 = _mdl(**enc)
    F.softmax(out2.logits, dim=-1)[0,1].backward()
    h2.remove()

    attrs  = (emb.grad[0] * emb[0]).sum(dim=-1).detach().numpy()
    tokens = _tok.convert_ids_to_tokens(enc["input_ids"][0])
    m      = enc["attention_mask"][0].numpy().astype(bool)
    tokens = [t for t,v in zip(tokens,m) if v]
    attrs  = attrs[:len(tokens)]

    ct, ca = [], []
    for t,a in zip(tokens,attrs):
        if t not in ("[CLS]","[SEP]","[PAD]"): ct.append(t); ca.append(float(a))

    mx   = max(abs(a) for a in ca) if ca else 1
    norm = [a/mx for a in ca]
    top  = sorted(zip(ct,norm), key=lambda x: abs(x[1]), reverse=True)[:10]
    return {"verdict": "Phishing" if pred==1 else "Legitimate",
            "confidence": conf, "label": pred,
            "tokens": ct, "scores": norm, "top_tokens": top}

if __name__ == "__main__":
    r = predict_email("URGENT: Account suspended",
                      "Click here to verify your account immediately.")
    print(r["verdict"], f"{r['confidence']*100:.1f}%")
    print("Top tokens:", [t for t,s in r["top_tokens"][:5]])
'''
Path("src/predict.py").write_text(predict_src, encoding="utf-8")
print("-- predict.py saved -> src/predict.py\n")

print("="*60)
print("  DAY 6 COMPLETE")
print("="*60)
print("  Heatmaps -> results/heatmaps/  (open .html in browser)")
print("  predict.py -> src/predict.py")
print("  Next: python src/adversarial_eval.py")
print("="*60 + "\n")
