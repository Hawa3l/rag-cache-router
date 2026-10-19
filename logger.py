import sqlite3
from datetime import datetime
from config import PRICES

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
        latency_ms REAL,
        pipeline TEXT
    )""")

    connection.commit()
    connection.close()

def cost_calculator(input_tokens,output_tokens,model):
    input_price,output_price = PRICES[model]
    return (input_tokens / 1000000) * input_price + (output_tokens / 1000000) * output_price


def log_request(question,answer,model_used,cache_hit,input_tokens,output_tokens,latency_ms,pipeline):
    cost_usd = cost_calculator(input_tokens,output_tokens,model_used) if cache_hit == 0 else 0 
    timestamp = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as connection:
        sql_ = "INSERT INTO requests (timestamp,question,answer,model_used,cache_hit,input_tokens,output_tokens,cost_usd,latency_ms,pipeline) values(?,?,?,?,?,?,?,?,?,?)"
        connection.execute(sql_,(timestamp,question,answer,model_used,cache_hit,input_tokens,output_tokens,cost_usd,latency_ms,pipeline))
       

def show_all():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM requests")
    res = cursor.fetchall()
    for row in res:
        print(row)

if __name__ == "__main__":
    init_db()
    print("Database is ready")
    show_all()
