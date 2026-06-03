import sqlite3

# ---------------- DATABASE CONNECTION ----------------

conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

cursor = conn.cursor()

# ---------------- USERS TABLE ----------------

cursor.execute("""

CREATE TABLE IF NOT EXISTS users (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT,

    age INTEGER,

    gender TEXT
)

""")

# ---------------- REPORTS TABLE ----------------

cursor.execute("""

CREATE TABLE IF NOT EXISTS reports (

    report_id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_id INTEGER,

    emotional_state TEXT,

    stress_level TEXT,

    recommendation TEXT,

    journal_text TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id) REFERENCES users(id)
)

""")

conn.commit()

# ---------------- FUNCTIONS ----------------

def create_user(name, age, gender):

    cursor.execute(

        """
        INSERT INTO users (name, age, gender)
        VALUES (?, ?, ?)
        """,

        (name, age, gender)
    )

    conn.commit()

    return cursor.lastrowid

# ---------------- SAVE REPORT ----------------

def save_report(

    user_id,
    emotional_state,
    stress_level,
    recommendation,
    journal_text
):

    cursor.execute(

        """
        INSERT INTO reports (

            user_id,
            emotional_state,
            stress_level,
            recommendation,
            journal_text

        )

        VALUES (?, ?, ?, ?, ?)
        """,

        (
            user_id,
            emotional_state,
            stress_level,
            recommendation,
            journal_text
        )
    )

    conn.commit()

# ---------------- GET USER REPORTS ----------------

def get_user_reports(user_id):

    cursor.execute(

        """
        SELECT *
        FROM reports
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,

        (user_id,)
    )

    return cursor.fetchall()