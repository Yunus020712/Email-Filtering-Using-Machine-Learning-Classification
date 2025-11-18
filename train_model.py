# train_model.py — binary (spam / not_spam) with UTF-8 IO + autosampling
import os, json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


MODEL_PATH = "email_classifier.pkl"
VEC_PATH   = "vectorizer.pkl"
INFO_PATH  = "model_info.txt"
HIST_PATH  = "model_history.csv"

# ensure each class has at least this many rows (duplicates with replacement if needed)
MIN_PER_CLASS = 20
CLASSES = ("spam", "not_spam")  # 👈 phishing removed


def _normalize_label(s: str) -> str:
    """Map many legacy labels to our 2-class schema (phishing -> spam)."""
    s = ("" if s is None else str(s)).strip().lower().replace("-", "_")
    if s in {"ham", "not spam", "non spam", "non_spam", "not_spam", "0"}:
        return "not_spam"
    if s in {"phish", "phishing"}:
        return "spam"            # 👈 collapse phishing into spam
    if s in {"spam", "junk", "1"}:
        return "spam"
    return s  # unknown → will be filtered out later


def _read_dataset(path: str) -> pd.DataFrame:
    """Read CSV preferring UTF-8, falling back to latin-1."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    return df


def _ensure_min_per_class(df: pd.DataFrame, min_count: int = MIN_PER_CLASS) -> pd.DataFrame:
    """Upsample minority classes with replacement so stratified split works."""
    out = df.copy()
    vc = out["label"].value_counts()
    rng = np.random.default_rng(42)
    for cls in CLASSES:
        n = int(vc.get(cls, 0))
        if n == 0:
            raise ValueError(f"❌ Missing class in dataset: '{cls}'. Please add samples.")
        if n < min_count:
            need = min_count - n
            samples = out[out["label"] == cls]
            add = samples.sample(n=need, replace=True, random_state=int(rng.integers(0, 1_000_000)))
            out = pd.concat([out, add], ignore_index=True)
    return out


def train_model(dataset_path: str = "spam_assassin.csv"):
    """
    Train and evaluate the email classification model (binary).
    Saves: email_classifier.pkl, vectorizer.pkl, model_info.txt, model_history.csv
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # ---------- Load & harmonize ----------
    data = _read_dataset(dataset_path)
    data.columns = data.columns.str.strip().str.lower()

    # map common schema variants -> ["text","label"]
    if "target" in data.columns:   data = data.rename(columns={"target": "label"})
    if "category" in data.columns: data = data.rename(columns={"category": "label"})
    if "message" in data.columns:  data = data.rename(columns={"message": "text"})
    if "content" in data.columns:  data = data.rename(columns={"content": "text"})
    if {"v2","v1"}.issubset(data.columns): data = data.rename(columns={"v2": "text", "v1": "label"})
    if {"email","class"}.issubset(data.columns): data = data.rename(columns={"email": "text", "class": "label"})
    if "type" in data.columns and "text" in data.columns and "label" not in data.columns:
        data = data.rename(columns={"type": "label"})

    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("❌ Dataset must contain 'text' and 'label' columns.")

    # clean + normalize labels
    data = (data[["text", "label"]]
            .dropna()
            .drop_duplicates(subset=["text"])
            .reset_index(drop=True))
    data["text"]  = data["text"].astype(str)
    data["label"] = data["label"].map(_normalize_label)

    # keep only our 2 classes (after collapsing phishing->spam)
    data = data[data["label"].isin(CLASSES)].copy()
    total_rows = len(data)
    if total_rows < 50:
        raise ValueError("❌ Dataset too small — need at least 50 samples.")

    # ensure both classes exist & have enough examples for stratified split
    class_counts = data["label"].value_counts().to_dict()
    missing = [c for c in CLASSES if c not in class_counts]
    if missing:
        raise ValueError(f"❌ Missing classes in dataset: {missing}. Please include samples for both classes.")

    data = _ensure_min_per_class(data, MIN_PER_CLASS)

    # ---------- Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    # ---------- Vectorize ----------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # ---------- Train ----------
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # ---------- Eval ----------
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds) * 100.0
    report = classification_report(
        y_test, preds,
        labels=list(CLASSES),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0
    )

    # ---------- Save artifacts ----------
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"accuracy={acc:.2f}",
        f"trained_on={now_str}",
        f"total_samples={total_rows}",
        "classes=spam,not_spam",
        f"precision_spam={report.get('spam',{}).get('precision',0)*100:.2f}",
        f"recall_spam={report.get('spam',{}).get('recall',0)*100:.2f}",
        f"f1_spam={report.get('spam',{}).get('f1-score',0)*100:.2f}",
        f"precision_not_spam={report.get('not_spam',{}).get('precision',0)*100:.2f}",
        f"recall_not_spam={report.get('not_spam',{}).get('recall',0)*100:.2f}",
        f"f1_not_spam={report.get('not_spam',{}).get('f1-score',0)*100:.2f}",
        f"macro_precision={report.get('macro avg',{}).get('precision',0)*100:.2f}",
        f"macro_recall={report.get('macro avg',{}).get('recall',0)*100:.2f}",
        f"macro_f1={report.get('macro avg',{}).get('f1-score',0)*100:.2f}",
        f"weighted_precision={report.get('weighted avg',{}).get('precision',0)*100:.2f}",
        f"weighted_recall={report.get('weighted avg',{}).get('recall',0)*100:.2f}",
        f"weighted_f1={report.get('weighted avg',{}).get('f1-score',0)*100:.2f}",
        "metrics_json=" + json.dumps(report)
    ]
    Path(INFO_PATH).write_text("\n".join(lines), encoding="utf-8")

    # history for chart
    hist_entry = pd.DataFrame([{
        "timestamp": now_str,
        "accuracy": round(acc, 2),
        "samples": total_rows
    }])
    if os.path.exists(HIST_PATH):
        hist_entry.to_csv(HIST_PATH, mode="a", header=False, index=False)
    else:
        hist_entry.to_csv(HIST_PATH, index=False)

    print("✅ Model training completed")
    print(f"   → Accuracy: {acc:.2f}%")
    print(f"   → Class counts (after collapse): {class_counts}")
    print(f"   → Trained On: {now_str}")
    return acc, total_rows, now_str


if __name__ == "__main__":
    acc, samples, trained_on = train_model()
    print(f"\nModel Accuracy: {acc:.2f}% | Samples: {samples} | Time: {trained_on}")
