"""
download_data.py (fixed)
────────────────────────
Downloads SpamAssassin corpus and parses emails correctly.
Handles the actual tarball structure SpamAssassin uses.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.data_utils import EmailSample, save_jsonl

RAW_DIR          = Path("data/raw")
PROCESSED_DIR    = Path("data/processed")
TARGET_PER_CLASS = 1500

SPAMASSASSIN_URLS = {
    "spam": [
        "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
    ],
    "ham": [
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    ],
}

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = dest_dir / url.split("/")[-1]
    if filename.exists():
        print(f"  [skip] Already downloaded: {filename.name}")
        return filename
    print(f"  [download] {filename.name} ...")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=filename.name) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
    return filename

def extract_tarball(tarball, extract_to):
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  [extract] {tarball.name} ...")
    with tarfile.open(tarball, "r:bz2") as tf:
        tf.extractall(extract_to)

def find_email_files(folder):
    files = []
    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        if f.name in {"cmds", "README", "SHA1SUMS", ".DS_Store"}:
            continue
        if f.suffix in {".tar", ".bz2", ".gz", ".md", ".txt"}:
            continue
        files.append(f)
    return files

def parse_email_file(path, label, source):
    import email as email_lib
    import re
    try:
        raw = path.read_bytes()
        try:
            msg = email_lib.message_from_bytes(raw)
        except Exception:
            msg = email_lib.message_from_string(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None

    subject = str(msg.get("Subject", "")).strip()
    sender  = str(msg.get("From",    "")).strip()
    body    = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body += payload.decode(charset, errors="replace") + " "
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")
            else:
                body = str(msg.get_payload() or "")
        except Exception:
            body = str(msg.get_payload() or "")

    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    if len((subject + " " + body).split()) < 5:
        return None

    return EmailSample(
        subject=subject, body=body[:5000], sender=sender,
        label=label, source=source, raw_path=str(path)
    )

def parse_folder(folder, label, source, limit=None):
    files = find_email_files(folder)
    print(f"  Found {len(files)} files in {folder}")
    samples = []
    for f in tqdm(files, desc="  Parsing", leave=False):
        s = parse_email_file(f, label, source)
        if s:
            samples.append(s)
        if limit and len(samples) >= limit:
            break
    print(f"  → {len(samples)} valid emails parsed")
    return samples

def main():
    print("\n" + "="*60)
    print("  Phase 1 · Data Download & Parsing")
    print("="*60 + "\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_phishing, all_legit = [], []

    print("── Step 1: Spam emails ──")
    for url in SPAMASSASSIN_URLS["spam"]:
        name    = url.split("/")[-1].replace(".tar.bz2", "")
        tarball = download_file(url, RAW_DIR / "spamassassin")
        dest    = RAW_DIR / "spamassassin" / "spam" / name
        if not dest.exists() or not any(dest.rglob("*")):
            extract_tarball(tarball, dest)
        # Show sample extracted paths for debugging
        extracted = list(dest.rglob("*"))
        print(f"  Extracted items: {len(extracted)}")
        if extracted:
            print(f"  Sample: {extracted[0]}")
        samples = parse_folder(dest, 1, "spamassassin",
                               limit=TARGET_PER_CLASS - len(all_phishing))
        all_phishing.extend(samples)
        if len(all_phishing) >= TARGET_PER_CLASS:
            break

    print(f"\nPhishing total: {len(all_phishing)}")

    print("\n── Step 2: Ham (legit) emails ──")
    for url in SPAMASSASSIN_URLS["ham"]:
        name    = url.split("/")[-1].replace(".tar.bz2", "")
        tarball = download_file(url, RAW_DIR / "spamassassin")
        dest    = RAW_DIR / "spamassassin" / "ham" / name
        if not dest.exists() or not any(dest.rglob("*")):
            extract_tarball(tarball, dest)
        extracted = list(dest.rglob("*"))
        print(f"  Extracted items: {len(extracted)}")
        samples = parse_folder(dest, 0, "spamassassin_ham",
                               limit=TARGET_PER_CLASS - len(all_legit))
        all_legit.extend(samples)
        if len(all_legit) >= TARGET_PER_CLASS:
            break

    print(f"\nLegit total: {len(all_legit)}")

    if not all_phishing and not all_legit:
        print("\n⚠ Still 0 emails parsed. Showing extracted structure:")
        for p in list((RAW_DIR / "spamassassin").rglob("*"))[:20]:
            print(f"  {p}")
        return

    import random
    random.seed(42)
    n       = min(len(all_phishing), len(all_legit), TARGET_PER_CLASS)
    dataset = random.sample(all_phishing, n) + random.sample(all_legit, n)
    random.shuffle(dataset)

    save_jsonl(dataset, PROCESSED_DIR / "dataset_v1.jsonl")
    print(f"\n✓ Done! {n} phishing + {n} legit = {len(dataset)} total")
    print("Next: run the EDA notebook\n")

if __name__ == "__main__":
    main()
