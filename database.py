import sqlite3

# Connect to database (creates file if not exists)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# --- Users Table ---
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT UNIQUE NOT NULL,
              role TEXT NOT NULL)''')

# --- Filtered Emails Table ---
c.execute('''CREATE TABLE IF NOT EXISTS filtered_emails
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              sender TEXT,
              subject TEXT,
              snippet TEXT,
              classification TEXT,
              user_email TEXT,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

# Example: add one admin account
try:
    c.execute("INSERT INTO users (email, role) VALUES (?, ?)",
              ("officialyunubs20@gmail.com", "admin"))
except sqlite3.IntegrityError:
    pass  # ignore if already exists

conn.commit()
conn.close()
print("Database setup complete.")
