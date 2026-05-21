"""
feature_engineering.py
───────────────────────
Extracts hand-crafted features from EmailSample objects.
Used by the baseline classifier (Day 2) AND as supplementary
features alongside DistilBERT embeddings (Day 5).

Feature groups:
    1. TF-IDF text features (subject + body)
    2. Structural features (lengths, counts)
    3. Lexical / tone features (urgency, CAPS, punctuation)
    4. URL / link features (count, suspicious patterns)
    5. Sender features (domain anomalies)

Usage:
    from src.feature_engineering import FeatureExtractor
    fe = FeatureExtractor()
    fe.fit(train_samples)
    X_train = fe.transform(train_samples)
    X_test  = fe.transform(test_samples)
"""

from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import List

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_utils import EmailSample


# ── Constants ────────────────────────────────────────────────────────────────

URGENCY_KEYWORDS = [
    "urgent", "immediately", "action required", "verify your",
    "account suspended", "click here", "click now", "confirm your",
    "your password", "limited time", "expires", "expiring",
    "warning", "alert", "security notice", "winner", "won",
    "free gift", "claim your", "update your", "invoice attached",
    "payment required", "overdue", "final notice", "dear customer",
    "dear user", "dear valued",
]

SUSPICIOUS_SENDER_PATTERNS = [
    r"no.?reply",
    r"noreply",
    r"support@(?!google|microsoft|apple|amazon|paypal)",
    r"\d{4,}@",          # lots of numbers in local part
    r"@.*\.(xyz|tk|ml|ga|cf|gq|pw)$",   # suspicious TLDs
]

LOOKALIKE_BRANDS = [
    "paypa1", "paypa-l", "go0gle", "arnazon", "micosoft",
    "faceb00k", "lnstagram", "linkedln", "netfl1x",
]


# ── Hand-crafted feature extractor ────────────────────────────────────────────

class HandCraftedFeatures:
    """
    Extracts a fixed-length vector of interpretable numeric features
    from an EmailSample. These are the features we'll later explain with SHAP.
    """

    FEATURE_NAMES = [
        "body_word_count",       # total words in body
        "subject_word_count",    # total words in subject
        "total_chars",           # total characters in body
        "link_count",            # number of http(s):// occurrences
        "urgency_score",         # count of urgency keyword matches
        "caps_ratio",            # fraction of alpha chars that are uppercase
        "exclamation_count",     # number of ! in body
        "question_count",        # number of ? in body
        "reply_to_mismatch",     # 1 if Reply-To domain ≠ From domain
        "suspicious_sender",     # 1 if sender matches suspicious patterns
        "lookalike_domain",      # 1 if lookalike brand in sender
        "avg_word_length",       # average word length (short = spammy)
        "numeric_ratio",         # fraction of chars that are digits
        "has_html_tags",         # 1 if body still contains HTML tags
        "subject_all_caps",      # 1 if subject is ALL CAPS
        "dollar_count",          # number of $ symbols
        "forward_slash_count",   # number of / (URL indicators)
    ]

    def transform(self, sample: EmailSample) -> np.ndarray:
        body    = sample.body    or ""
        subject = sample.subject or ""
        sender  = sample.sender  or ""
        combined = (subject + " " + body).lower()

        words      = body.split()
        chars      = list(body)
        alpha_chars = [c for c in chars if c.isalpha()]

        # ── 1. Structural ─────────────────────────────────────────────────────
        body_word_count    = len(words)
        subject_word_count = len(subject.split())
        total_chars        = len(body)

        # ── 2. URL / link ─────────────────────────────────────────────────────
        link_count         = len(re.findall(r"https?://", body))
        forward_slash_count = body.count("/")

        # ── 3. Urgency ────────────────────────────────────────────────────────
        urgency_score = sum(combined.count(kw.lower()) for kw in URGENCY_KEYWORDS)

        # ── 4. Tone / style ───────────────────────────────────────────────────
        caps_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars else 0.0
        )
        exclamation_count = body.count("!")
        question_count    = body.count("?")
        dollar_count      = body.count("$")
        subject_all_caps  = int(subject == subject.upper() and len(subject) > 3)

        # ── 5. Vocabulary ─────────────────────────────────────────────────────
        avg_word_length = (
            np.mean([len(w) for w in words]) if words else 0.0
        )
        numeric_ratio = (
            sum(1 for c in chars if c.isdigit()) / len(chars)
            if chars else 0.0
        )
        has_html_tags = int(bool(re.search(r"<[a-zA-Z]+[^>]*>", body)))

        # ── 6. Sender analysis ────────────────────────────────────────────────
        # Reply-To vs From mismatch
        reply_to_mismatch = 0  # would need actual email headers; default 0
        suspicious_sender = int(
            any(re.search(p, sender.lower()) for p in SUSPICIOUS_SENDER_PATTERNS)
        )
        lookalike_domain = int(
            any(brand in sender.lower() for brand in LOOKALIKE_BRANDS)
        )

        return np.array([
            body_word_count, subject_word_count, total_chars,
            link_count, urgency_score, caps_ratio,
            exclamation_count, question_count, reply_to_mismatch,
            suspicious_sender, lookalike_domain, avg_word_length,
            numeric_ratio, has_html_tags, subject_all_caps,
            dollar_count, forward_slash_count,
        ], dtype=np.float32)

    def transform_batch(self, samples: List[EmailSample]) -> np.ndarray:
        """Transform a list of EmailSamples into an (N, 17) feature matrix."""
        return np.vstack([self.transform(s) for s in samples])


# ── Full feature extractor (TF-IDF + hand-crafted) ───────────────────────────

class FeatureExtractor:
    """
    Combines TF-IDF features with hand-crafted numeric features.
    Produces a sparse matrix suitable for scikit-learn classifiers.

    Typical usage:
        fe = FeatureExtractor()
        fe.fit(train_samples)
        X_train = fe.transform(train_samples)

        # Save / load
        fe.save('models/feature_extractor.pkl')
        fe2 = FeatureExtractor.load('models/feature_extractor.pkl')
    """

    def __init__(
        self,
        tfidf_max_features: int = 8000,
        tfidf_ngram_range: tuple = (1, 2),
    ):
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            sublinear_tf=True,           # log(1+tf) — helps with frequent spam words
            strip_accents="unicode",
            min_df=2,                    # ignore terms appearing in only 1 doc
        )
        self.scaler   = StandardScaler()
        self.hcf      = HandCraftedFeatures()
        self._fitted  = False

    def _get_texts(self, samples: List[EmailSample]) -> List[str]:
        return [s.to_combined_text() for s in samples]

    def fit(self, samples: List[EmailSample]) -> "FeatureExtractor":
        texts = self._get_texts(samples)
        self.tfidf.fit(texts)

        hc_matrix = self.hcf.transform_batch(samples)
        self.scaler.fit(hc_matrix)

        self._fitted = True
        print(f"[FeatureExtractor] Fitted on {len(samples):,} samples  |  "
              f"TF-IDF vocab: {len(self.tfidf.vocabulary_):,}  |  "
              f"Hand-crafted: {len(HandCraftedFeatures.FEATURE_NAMES)} features")
        return self

    def transform(self, samples: List[EmailSample]):
        """Returns a sparse matrix (N, tfidf_features + 17)."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")

        texts     = self._get_texts(samples)
        tfidf_mat = self.tfidf.transform(texts)

        hc_mat    = self.hcf.transform_batch(samples)
        hc_scaled = self.scaler.transform(hc_mat)

        return hstack([tfidf_mat, csr_matrix(hc_scaled)])

    def fit_transform(self, samples: List[EmailSample]):
        return self.fit(samples).transform(samples)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[FeatureExtractor] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureExtractor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = EmailSample(
        subject="URGENT: Your PayPal account has been suspended",
        body="Dear customer, click here immediately to verify your account. "
             "Your access will expire within 24 hours. https://paypa1.com/verify",
        sender="support@paypa1.com",
        label=1,
        source="test"
    )

    hcf = HandCraftedFeatures()
    vec = hcf.transform(sample)

    print("Feature names:")
    for name, val in zip(HandCraftedFeatures.FEATURE_NAMES, vec):
        print(f"  {name:<28} = {val:.3f}")
    print("\nfeature_engineering.py OK ✓")
