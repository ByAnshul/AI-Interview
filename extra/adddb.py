import sqlite3

# Path to your database file
DB_PATH = 'users.db'  # change this if needed

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Run the schema update
try:
    cursor.execute('ALTER TABLE users ADD COLUMN resume_text TEXT;')
    print("Column 'resume_text' added successfully.")
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        print("Column already exists.")
    else:
        print("Error:", e)

# Save and close
conn.commit()
conn.close()
