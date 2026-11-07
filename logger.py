import sqlite3
from datetime import datetime
from config import PRICES, BASE_DIR
from pathlib import Path

DB_PATH = BASE_DIR / "logs.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    with get_connection() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            model_used TEXT,
            cache_hit INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            latency_ms REAL,
            pipeline TEXT,
            run_id TEXT
        )"""
        )
        # Migrate existing table if run_id column is missing
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(requests)")
        columns = [col[1] for col in cursor.fetchall()]
        if "run_id" not in columns:
            cursor.execute("ALTER TABLE requests ADD COLUMN run_id TEXT")
        connection.commit()


def cost_calculator(input_tokens, output_tokens, model):
    if model not in PRICES:
        return 0.0
    input_price, output_price = PRICES[model]
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def log_request(
    question,
    answer,
    model_used,
    cache_hit,
    input_tokens,
    output_tokens,
    latency_ms,
    pipeline,
    run_id=None,
):
    cost_usd = cost_calculator(input_tokens, output_tokens, model_used) if cache_hit == 0 else 0.0
    timestamp = datetime.now().isoformat()
    with get_connection() as connection:
        sql = (
            "INSERT INTO requests "
            "(timestamp, question, answer, model_used, cache_hit, "
            "input_tokens, output_tokens, cost_usd, latency_ms, pipeline, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        connection.execute(
            sql,
            (
                timestamp,
                question,
                answer,
                model_used,
                cache_hit,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_ms,
                pipeline,
                run_id,
            ),
        )
        connection.commit()


def show_all():
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM requests")
        for row in cursor.fetchall():
            print(row)


def clear_db():
    with get_connection() as connection:
        connection.execute("DELETE FROM requests")
        connection.commit()


if __name__ == "__main__":
    init_db()
    show_all()
