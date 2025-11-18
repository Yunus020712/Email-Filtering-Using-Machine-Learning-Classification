# migration_add_label.py
import sqlite3
from pathlib import Path

DB_PATH = "users.db"     # <-- your DB file 
TABLE   = "emails"       # <-- your emails table name

def table_exists(cur, table):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None

def column_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    if not table_exists(cur, TABLE):
        raise RuntimeError(f"Table '{TABLE}' not found in {DB_PATH}. Check your table name.")

    # 1) Add label column if missing
    if not column_exists(cur, TABLE, "label"):
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN label TEXT NOT NULL DEFAULT 'not_spam'")
        print("✅ Added column: label (TEXT, default 'not_spam')")
    else:
        print("ℹ️  Column 'label' already exists — skipping add.")

    # 2) Backfill from legacy is_spam if it exists
    if column_exists(cur, TABLE, "is_spam"):
        cur.execute(f"""
            UPDATE {TABLE}
            SET label = CASE
                WHEN is_spam = 1 THEN 'spam'
                ELSE label
            END
            WHERE label NOT IN ('spam','phishing','not_spam') OR label IS NULL;
        """)
        print("✅ Backfilled 'label' using 'is_spam' (1 → 'spam', others keep default).")
    else:
        print("ℹ️  Column 'is_spam' not found — no backfill performed.")

    con.commit()

    # 3) Show a quick summary
    try:
        cur.execute(f"""
            SELECT
              SUM(CASE WHEN label='spam' THEN 1 ELSE 0 END),
              SUM(CASE WHEN label='phishing' THEN 1 ELSE 0 END),
              SUM(CASE WHEN label='not_spam' THEN 1 ELSE 0 END)
            FROM {TABLE};
        """)
        spam, phishing, not_spam = cur.fetchone()
        print(f"Summary → spam={spam or 0}, phishing={phishing or 0}, not_spam={not_spam or 0}")
    except Exception:
        pass

    con.close()
    print("🎉 Migration complete.")

if __name__ == "__main__":
    main()
