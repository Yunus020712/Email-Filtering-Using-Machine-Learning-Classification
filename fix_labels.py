# fix_labels.py
import pandas as pd, sys

SRC = "spam_assassin.csv"          # your master dataset
DST = "spam_assassin.csv"          # overwrite in-place

# Read: prefer UTF-8, fallback to latin-1 if needed
try:
    df = pd.read_csv(SRC, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(SRC, encoding="latin-1")

df.columns = df.columns.str.strip().str.lower()

# Harmonize to ['text','label']
if "target" in df.columns: df = df.rename(columns={"target":"label"})
if "category" in df.columns: df = df.rename(columns={"category":"label"})
if {"v2","v1"}.issubset(df.columns): df = df.rename(columns={"v2":"text","v1":"label"})
if "message" in df.columns: df = df.rename(columns={"message":"text"})
if "content" in df.columns: df = df.rename(columns={"content":"text"})
if "email" in df.columns: df = df.rename(columns={"email":"text"})
if "class" in df.columns: df = df.rename(columns={"class":"label"})
if "type" in df.columns and "text" in df.columns and "label" not in df.columns:
    df = df.rename(columns={"type":"label"})

if not {"text","label"}.issubset(df.columns):
    raise SystemExit("Dataset must have 'text' and 'label' columns after normalization.")

# Normalize labels
def norm(s: str) -> str:
    s = str(s).strip().lower().replace("-", "_")
    if s in {"ham","not spam","non spam","non_spam","not_spam","0"}: return "not_spam"
    if s in {"spam","1","junk"}: return "spam"
    if s in {"phish","phishing","3"}: return "phishing"
    return s

df["label"] = df["label"].map(norm)

# Keep only the 3 classes
df = df[df["label"].isin(["spam","not_spam","phishing"])].dropna(subset=["text"])
print("Label counts after cleanup:\n", df["label"].value_counts())

if "phishing" not in set(df["label"]):
    print("\n⚠️ No 'phishing' rows found. Add some labeled 'phishing' examples to train a 3-class model.")
    # still save cleaned file so you can add phishing later
    df.to_csv(DST, index=False, encoding="utf-8")
    sys.exit(0)

# ✅ Always write as UTF-8
df.to_csv(DST, index=False, encoding="utf-8")
print("\n✅ Saved cleaned dataset to", DST)
