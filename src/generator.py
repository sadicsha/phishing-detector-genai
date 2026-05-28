"""
generator.py
────────────
Red Team: AI-powered phishing email generator using Groq API.
Run this script directly:
    python src/generator.py

What it does:
    1. Defines 6 phishing persona + lure type combinations
    2. Uses Groq (Llama3) to generate convincing phishing emails
    3. Filters low-quality outputs
    4. Saves generated emails to data/processed/synthetic_phishing.jsonl
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from tqdm import tqdm
from groq import Groq

from src.data_utils import EmailSample, save_jsonl, load_jsonl

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_PATH      = Path("data/processed/synthetic_phishing.jsonl")
TARGET_COUNT     = 800      # total synthetic emails to generate
MODEL            = "llama-3.1-8b-instant"   # fast + free on Groq
TEMPERATURE      = 0.85     # higher = more varied emails
MAX_RETRIES      = 3
DELAY_BETWEEN    = 1.2      # seconds between API calls (stay under rate limit)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Persona definitions ───────────────────────────────────────────────────────

@dataclass
class Persona:
    company:       str    # company being impersonated
    industry:      str    # industry context
    target_role:   str    # who receives this email
    lure_type:     str    # type of phishing lure
    urgency:       int    # 1=low 2=medium 3=high
    sender_name:   str    # fake sender name
    sender_email:  str    # fake sender email

PERSONAS = [
    Persona("PayPal",      "fintech",    "online shopper",       "account_suspended",  3, "PayPal Security Team",   "security@paypa1-alerts.com"),
    Persona("Microsoft",   "tech",       "office employee",      "it_alert",           3, "IT Support Desk",        "support@microsft-helpdesk.com"),
    Persona("HR Dept",     "corporate",  "company employee",     "hr_notice",          2, "Human Resources",        "hr@company-updates.net"),
    Persona("FedEx",       "logistics",  "online shopper",       "delivery_scam",      2, "FedEx Delivery",         "tracking@fedex-delivery-alert.com"),
    Persona("Amazon",      "ecommerce",  "amazon customer",      "invoice_fraud",      3, "Amazon Billing",         "billing@amazon-invoice-alert.com"),
    Persona("IT Security", "corporate",  "company employee",     "password_reset",     3, "IT Security Team",       "security@it-helpdesk-alert.com"),
]

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a cybersecurity researcher generating realistic phishing email samples 
for training email security classifiers. These are used purely for research and model training.
Generate realistic-looking phishing emails that security teams use to train detection systems.
Always output ONLY a JSON object with keys: subject, body, sender_display.
No extra text, no markdown, no explanation. Just the JSON."""

LURE_TEMPLATES = {
    "account_suspended": """Generate a phishing email from {sender_name} ({sender_email}) to a {target_role}.
The email claims their {company} account has been suspended and they must verify immediately.
Urgency level: {urgency}/3. Include a fake link placeholder like [VERIFY NOW: http://fake-link.com/verify].
Make it sound official but with subtle urgency pressure.
Output JSON only: {{"subject": "...", "body": "...", "sender_display": "..."}}""",

    "it_alert": """Generate a phishing email from {sender_name} ({sender_email}) to a {target_role}.
The email claims there is an urgent IT security alert — their password will expire or account will be locked.
Urgency level: {urgency}/3. Include a fake link like [RESET NOW: http://fake-link.com/reset].
Use corporate IT language. Output JSON only: {{"subject": "...", "body": "...", "sender_display": "..."}}""",

    "hr_notice": """Generate a phishing email from {sender_name} ({sender_email}) to a {target_role}.
The email is about an important HR policy update, benefits enrollment, or payroll issue requiring action.
Urgency level: {urgency}/3. Include a fake link like [ACCESS PORTAL: http://fake-link.com/hr].
Sound like an internal HR communication. Output JSON only: {{"subject": "...", "body": "...", "sender_display": "..."}}""",

    "delivery_scam": """Generate a phishing email from {sender_name} ({sender_email}) to a {target_role}.
The email claims a package delivery failed and they must reschedule or pay a small fee.
Urgency level: {urgency}/3. Include a fake tracking link like [TRACK PACKAGE: http://fake-link.com/track].
Output JSON only: {{"subject": "...", "body": "...", "sender_display": "..."}}""",

    "invoice_fraud": """Generate a phishing email from {sender_name} ({sender_email}) to a {target_role}.
The email claims there is an unexpected charge or invoice that needs immediate review or dispute.
Urgency level: {urgency}/3. Include a fake link like [VIEW INVOICE: http://fake-link.com/invoice].
Output JSON only: {{"subject": "...", "body": "...", "sender_display": "..."}}""",

    "password_reset": """Generate a phishing email from {sender_name} ({sender_email}) to a {target_role}.
The email claims someone tried to access their account and they must reset their password immediately.
Urgency level: {urgency}/3. Include a fake link like [SECURE ACCOUNT: http://fake-link.com/secure].
Output JSON only: {{"subject": "...", "body": "...", "sender_display": "..."}}""",
}

# ── Quality filter ────────────────────────────────────────────────────────────

def is_quality_email(subject: str, body: str) -> tuple[bool, str]:
    """
    Returns (passed, reason) — filters out low-quality generated emails.
    """
    word_count = len(body.split())

    if word_count < 40:
        return False, f"too short ({word_count} words)"

    if not subject or len(subject) < 5:
        return False, "missing/empty subject"

    # Must have some kind of call to action
    cta_signals = ["http", "click", "verify", "confirm", "access",
                   "login", "sign in", "reset", "update", "review"]
    body_lower = body.lower()
    if not any(s in body_lower for s in cta_signals):
        return False, "no call-to-action found"

    return True, "ok"


# ── Generator ─────────────────────────────────────────────────────────────────

class PhishingGenerator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file.\n"
                "Get your free key at: https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.refused_log = []

    def generate_one(self, persona: Persona) -> Optional[EmailSample]:
        """Generate a single phishing email for the given persona."""
        template = LURE_TEMPLATES[persona.lure_type]
        prompt   = template.format(
            sender_name  = persona.sender_name,
            sender_email = persona.sender_email,
            target_role  = persona.target_role,
            company      = persona.company,
            urgency      = persona.urgency,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model    = MODEL,
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature = TEMPERATURE,
                    max_tokens  = 600,
                )

                raw = response.choices[0].message.content.strip()

                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                data = json.loads(raw)

                subject = str(data.get("subject", "")).strip()
                body    = str(data.get("body",    "")).strip()
                sender  = str(data.get("sender_display", persona.sender_name)).strip()

                passed, reason = is_quality_email(subject, body)
                if not passed:
                    self.refused_log.append({
                        "reason": reason, "lure": persona.lure_type,
                        "attempt": attempt + 1
                    })
                    continue

                return EmailSample(
                    subject   = subject,
                    body      = body,
                    sender    = f"{sender} <{persona.sender_email}>",
                    label     = 1,
                    source    = "synthetic",
                    lure_type = persona.lure_type,
                )

            except json.JSONDecodeError:
                self.refused_log.append({
                    "reason": "invalid JSON response",
                    "lure": persona.lure_type, "attempt": attempt + 1
                })
                time.sleep(1)
                continue

            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err:
                    wait = 10 * (attempt + 1)
                    print(f"\n  [rate limit] Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    self.refused_log.append({
                        "reason": str(e)[:80],
                        "lure": persona.lure_type, "attempt": attempt + 1
                    })
                    break

        return None

    def generate_batch(self, count: int) -> list[EmailSample]:
        """Generate `count` phishing emails across all personas."""
        samples      = []
        per_persona  = count // len(PERSONAS) + 1
        failed_count = 0

        print(f"\n  Generating {count} phishing emails across {len(PERSONAS)} lure types...")
        print(f"  ~{per_persona} emails per lure type\n")

        with tqdm(total=count, desc="  Generating", unit="email") as pbar:
            for persona in PERSONAS:
                generated_for_persona = 0

                while generated_for_persona < per_persona and len(samples) < count:
                    sample = self.generate_one(persona)

                    if sample:
                        samples.append(sample)
                        generated_for_persona += 1
                        pbar.update(1)
                    else:
                        failed_count += 1

                    time.sleep(DELAY_BETWEEN)

        print(f"\n  Generated : {len(samples)}")
        print(f"  Filtered  : {failed_count} (too short, no CTA, or bad JSON)")
        return samples


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(samples: list[EmailSample], threshold: float = 0.85) -> list[EmailSample]:
    """Remove near-duplicate emails using TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    print(f"\n── Deduplication (threshold={threshold}) ──")
    if len(samples) < 2:
        return samples

    texts = [s.subject + " " + s.body for s in samples]
    tfidf = TfidfVectorizer(max_features=3000).fit_transform(texts)
    sim   = cosine_similarity(tfidf)

    keep = []
    dropped = set()
    for i in range(len(samples)):
        if i in dropped:
            continue
        keep.append(i)
        for j in range(i + 1, len(samples)):
            if sim[i, j] > threshold:
                dropped.add(j)

    result = [samples[i] for i in keep]
    print(f"  Before: {len(samples)}  After: {len(result)}  Removed: {len(dropped)} duplicates")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
       # Resume if partially generated
    existing = []
    if OUTPUT_PATH.exists():
        existing = load_jsonl(OUTPUT_PATH)
        print(f"\n  Resuming — {len(existing)} emails already generated")

    remaining = TARGET_COUNT - len(existing)

    if remaining <= 0:
        print(f"  Already have {len(existing)} emails. Done!")
    else:
        gen     = PhishingGenerator()
        new_s   = gen.generate_batch(remaining)
        all_s   = existing + new_s
        all_s   = deduplicate(all_s)

        save_jsonl(all_s, OUTPUT_PATH)

        # Save refused log
        if gen.refused_log:
            refused_path = Path("results/refused_generations.json")
            refused_path.parent.mkdir(exist_ok=True)
            refused_path.write_text(json.dumps(gen.refused_log, indent=2))
            print(f"  Refused log → {refused_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    final = load_jsonl(OUTPUT_PATH)
    lure_counts = {}
    for s in final:
        lure_counts[s.lure_type] = lure_counts.get(s.lure_type, 0) + 1

    print(f"\n  Total synthetic emails : {len(final)}")
    print(f"\n  Breakdown by lure type:")
    for lure, count in sorted(lure_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 10)
        print(f"  {lure:<20} {count:>4}  {bar}")
    print(f"\n  Saved → {OUTPUT_PATH}")
   
if __name__ == "__main__":
    main()
