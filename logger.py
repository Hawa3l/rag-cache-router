import sqlite3
DB_PATH = "logs.db"
def init_db() : 
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        question TEXT,
        answer TEXT,
        model_used TEXT,
        cache_hit INTEGER,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cost_usd REAL,
        latency_ms REAL
    )""")

    connection.commit()
    connection.close()


if __name__ == "__main__":
    init_db()
    print("Database is ready")
    
