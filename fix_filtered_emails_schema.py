# fix_filtered_emails_schema.py
import sqlite3
from pathlib import Path

DB_PATH = "users.db"
TABLE   = "filtered_emails"

def cols(cur):
    cur.execute(f"PRAGMA table_info({TABLE})")
    return {r[1] for r in cur.fetchall()}

def main():
    if not Path(DB_PATH).exists():
        raise SystemExit(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Ensure table exists (create minimal if missing)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            sender TEXT,
            subject TEXT,
            snippet TEXT,
            classification TEXT
        )
    """)
    current = cols(cur)

    # Add gmail_id if missing
    if "gmail_id" not in current:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN gmail_id TEXT")
        print("✅ Added column: gmail_id (TEXT)")
        current.add("gmail_id")

    # Add created_at if missing
    if "created_at" not in current:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN created_at TEXT")
        cur.execute(f"UPDATE {TABLE} SET created_at = strftime('%Y-%m-%d %H:%M:%S','now') WHERE created_at IS NULL")
        print("✅ Added column: created_at (TEXT) and backfilled existing rows")

    # Unique index on gmail_id (dedupe)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_filtered_emails_gmail_id ON filtered_emails(gmail_id)")
    con.commit()

    # Show final schema
    cur.execute(f"PRAGMA table_info({TABLE})")
    print("Columns:", [r[1] for r in cur.fetchall()])
    con.close()
    print("🎉 Schema fixed.")

if __name__ == "__main__":
    main()
