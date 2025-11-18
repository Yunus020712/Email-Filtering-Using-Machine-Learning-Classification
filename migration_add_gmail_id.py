# migration_add_gmail_id.py
import sqlite3
from pathlib import Path

DB_PATH = "users.db"
TABLE = "filtered_emails"

def col_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def main():
    if not Path(DB_PATH).exists():
        raise SystemExit(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # add gmail_id column if missing
    if not col_exists(cur, TABLE, "gmail_id"):
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN gmail_id TEXT")
        print("✅ Added column gmail_id (TEXT)")
    else:
        print("ℹ️ gmail_id already exists")

    # unique index for dedupe
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_filtered_emails_gmail_id ON filtered_emails(gmail_id)")
    print("✅ Ensured unique index on gmail_id")

    con.commit()
    con.close()
    print("🎉 Migration complete.")

if __name__ == "__main__":
    main()
