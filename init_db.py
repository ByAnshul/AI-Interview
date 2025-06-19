import sqlite3

# Connect to the database (or create it if not exists)
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Create table if not exists with new fields
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT,
        last_name TEXT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        bio TEXT,
        location TEXT,
        pronouns TEXT,
        website TEXT,
        calendar_link TEXT,
        resume_path TEXT,  -- file path for uploaded resume
        photo_path TEXT,   -- file path for uploaded profile photo
        skills TEXT,       -- comma-separated string
        twitter TEXT,
        linkedin TEXT,
        github TEXT,
        instagram TEXT
    )
''')

conn.commit()
conn.close()
print("✅ Extended users table created successfully with photo_path.")
