import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODELS_DIR = Path("models/distilbert")
HF_MODEL_REPO = "sadicsha/phishing-detector-distilbert"
MAX_LENGTH = 128
_tok = None
_mdl = None

def _load():
    global _tok, _mdl
    if _mdl is None:
        if MODELS_DIR.exists():
            model_path = MODELS_DIR
        else:
            model_path = HF_MODEL_REPO
        _tok = DistilBertTokenizerFast.from_pretrained(model_path)
        _mdl = DistilBertForSequenceClassification.from_pretrained(model_path)
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
