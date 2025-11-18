import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

# Create users table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL
)
""")

# Example users(contoh)
users = [
    ("officialyunubs20@gmail.com", "admin"),
    ("user@example.com", "user")
]

for email, role in users:
    try:
        c.execute("INSERT INTO users (email, role) VALUES (?, ?)", (email, role))
    except sqlite3.IntegrityError:
        pass

conn.commit()
conn.close()

print("Database setup complete.")
