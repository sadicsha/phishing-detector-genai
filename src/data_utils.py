"""
data_utils.py
─────────────
Shared utilities for loading, parsing, normalising, and splitting email data.
Every other module imports from here — keep this file clean and well-tested.

Usage:
    from src.data_utils import EmailSample, load_dataset, parse_eml_file, train_test_val_split
"""

from __future__ import annotations

import email
import email.policy
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EmailSample:
    """Single normalised email sample used throughout the pipeline."""
    subject:    str           # cleaned subject line (empty string if missing)
    body:       str           # plain-text body (HTML stripped)
    sender:     str           # raw From: header value
    label:      int           # 1 = phishing, 0 = legitimate
    source:     str           # 'spamassassin' | 'enron' | 'synthetic'
    lure_type:  str = ""      # e.g. 'invoice', 'it_alert' — filled by generator
    raw_path:   str = ""      # original file path for traceability

    def to_combined_text(self, max_body_tokens: int = 256) -> str:
        """
        Returns subject + first N words of body as a single string.
        This is what gets fed to DistilBERT's tokenizer.
        """
        body_snippet = " ".join(self.body.split()[:max_body_tokens])
        return f"Subject: {self.subject} [SEP] {body_snippet}"

    def to_dict(self) -> dict:
        return asdict(self)


# ── .eml parser ───────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Very fast HTML stripper — no dependency on BS4 for core util."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    text = re.sub(r"\s+",    " ", text)
    return text.strip()


def _extract_body(msg: email.message.Message) -> str:
    """Walk MIME parts and return the best plain-text body."""
    plain_parts = []
    html_parts  = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ct == "text/plain":
                plain_parts.append(text)
            elif ct == "text/html":
                html_parts.append(text)
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
                if msg.get_content_type() == "text/html":
                    html_parts.append(text)
                else:
                    plain_parts.append(text)
        except Exception:
            pass

    if plain_parts:
        return " ".join(plain_parts).strip()
    if html_parts:
        return _strip_html(" ".join(html_parts))
    return ""


def parse_eml_file(path: str | Path, label: int, source: str) -> Optional[EmailSample]:
    """
    Parse a single .eml file into an EmailSample.
    Returns None if the file cannot be parsed or produces empty content.

    Args:
        path:   Path to the .eml file
        label:  1 for phishing, 0 for legitimate
        source: dataset name string

    Returns:
        EmailSample or None
    """
    try:
        with open(path, "rb") as f:
            msg = email.message_from_bytes(f.read(), policy=email.policy.compat32)
    except Exception:
        return None

    subject = str(msg.get("Subject", "")).strip()
    sender  = str(msg.get("From", "")).strip()
    body    = _extract_body(msg)

    # Reject samples that are basically empty
    combined = (subject + " " + body).strip()
    if len(combined.split()) < 5:
        return None

    return EmailSample(
        subject=subject,
        body=body,
        sender=sender,
        label=label,
        source=source,
        raw_path=str(path),
    )


# ── JSONL I/O ────────────────────────────────────────────────────────────────

def save_jsonl(samples: List[EmailSample], path: str | Path) -> None:
    """Save a list of EmailSamples to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    print(f"[data_utils] Saved {len(samples):,} samples → {path}")


def load_jsonl(path: str | Path) -> List[EmailSample]:
    """Load EmailSamples from a JSONL file."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(EmailSample(**d))
    return samples


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def samples_to_df(samples: List[EmailSample]) -> pd.DataFrame:
    """Convert list of EmailSamples to a pandas DataFrame."""
    return pd.DataFrame([s.to_dict() for s in samples])


def df_to_samples(df: pd.DataFrame) -> List[EmailSample]:
    """Convert a DataFrame back to EmailSample list."""
    return [EmailSample(**row) for row in df.to_dict("records")]


# ── Train / val / test split ──────────────────────────────────────────────────

def train_test_val_split(
    samples: List[EmailSample],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    random_state: int  = 42,
) -> Tuple[List[EmailSample], List[EmailSample], List[EmailSample]]:
    """
    Stratified split into train / val / test sets.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    labels = [s.label for s in samples]

    # First: split off test
    train_val, test = train_test_split(
        samples, test_size=test_ratio,
        stratify=labels, random_state=random_state
    )
    # Then: split train_val into train and val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    tv_labels = [s.label for s in train_val]
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted,
        stratify=tv_labels, random_state=random_state
    )

    print(f"[data_utils] Split → train:{len(train):,}  val:{len(val):,}  test:{len(test):,}")
    return train, val, test


# ── Dataset loader (orchestrates everything) ──────────────────────────────────

def load_dataset(jsonl_path: str | Path) -> Tuple[pd.DataFrame, List[EmailSample]]:
    """
    Load a JSONL dataset and return both a DataFrame and a list of EmailSamples.
    Convenience wrapper used by the EDA notebook and training scripts.
    """
    samples = load_jsonl(jsonl_path)
    df = samples_to_df(samples)
    print(f"[data_utils] Loaded {len(df):,} samples  |  "
          f"phishing:{df['label'].sum():,}  legit:{(df['label']==0).sum():,}")
    return df, samples


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test with a synthetic sample
    sample = EmailSample(
        subject="URGENT: Verify your account",
        body="Dear user, click here immediately to avoid suspension.",
        sender="support@paypa1.com",
        label=1,
        source="test"
    )
    print("Combined text:", sample.to_combined_text())
    print("Dict:", sample.to_dict())
    print("data_utils.py OK ✓")
