# backfill_labels.py
import sqlite3, shutil, os, joblib
from datetime import datetime

DB_PATH = "users.db"
TABLE   = "filtered_emails"   # change if your table name differs
MODEL  = "email_classifier.pkl"
VEC    = "vectorizer.pkl"

def backup_db():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = f"{os.path.splitext(DB_PATH)[0]}-backup-{ts}.db"
    shutil.copyfile(DB_PATH, dst)
    print(f"📦 DB backup created: {dst}")

def classify(vec, clf, subject, snippet, body=None):
    text = " ".join([(subject or ""), (snippet or ""), (body or "")]).strip()
    label = clf.predict(vec.transform([text]))[0]
    return label if label in ("spam", "not_spam", "phishing") else "spam"

def main():
    if not (os.path.exists(DB_PATH) and os.path.exists(MODEL) and os.path.exists(VEC)):
        raise SystemExit("❌ Missing users.db or model/vectorizer files.")

    backup_db()

    clf = joblib.load(MODEL)
    vec = joblib.load(VEC)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Count before
    cur.execute(f"""
      SELECT
        SUM(CASE WHEN classification='spam' THEN 1 ELSE 0 END),
        SUM(CASE WHEN classification='phishing' THEN 1 ELSE 0 END),
        SUM(CASE WHEN lower(classification) IN ('not_spam','non-spam','ham') THEN 1 ELSE 0 END)
      FROM {TABLE}
    """)
    before_spam, before_phish, before_not = [x or 0 for x in cur.fetchone()]
    print(f"Before → spam={before_spam}, phishing={before_phish}, not_spam={before_not}")

    # Fetch rows to relabel (all rows, or filter only spam/not_spam if you prefer)
    cur.execute(f"SELECT id, subject, snippet FROM {TABLE}")
    rows = cur.fetchall()

    updated = 0
    for _id, subject, snippet in rows:
        new_label = classify(vec, clf, subject, snippet)
        cur.execute(f"UPDATE {TABLE} SET classification=? WHERE id=?", (new_label, _id))
        updated += 1

    con.commit()

    # Count after
    cur.execute(f"""
      SELECT
        SUM(CASE WHEN classification='spam' THEN 1 ELSE 0 END),
        SUM(CASE WHEN classification='phishing' THEN 1 ELSE 0 END),
        SUM(CASE WHEN classification='not_spam' THEN 1 ELSE 0 END)
      FROM {TABLE}
    """)
    after_spam, after_phish, after_not = [x or 0 for x in cur.fetchone()]
    con.close()

    print(f"✅ Updated {updated} rows.")
    print(f"After  → spam={after_spam}, phishing={after_phish}, not_spam={after_not}")

if __name__ == "__main__":
    main()
