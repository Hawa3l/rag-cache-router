import sqlite3
from datetime import datetime
DB_PATH = "logs.db"
PRICES = {
    "claude-opus-5-5" : (4,20),
    "claude-fable-5-1" : (10,50),
    "claude-opus-5" : (5,25),
    "claude-sonnet-5" : (2,10),
    "claude-haiku-4-5-20251001" : (1,5),
    "claude-mythos-5" : (10,50)
}
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

def cost_calculator(input_tokens,output_tokens,model):
    input_price,output_price = PRICES[model]
    return (input_tokens / 1000000) * input_price + (output_tokens / 1000000) * output_price


def log_request(question,answer,model_used,cache_hit,input_tokens,output_tokens,latency_ms):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cost_usd = cost_calculator(input_tokens,output_tokens,model_used) if cache_hit == 0 else 0 
    timestamp = datetime.now().isoformat()
    sql_ = "INSERT INTO requests (timestamp,question,answer,model_used,cache_hit,input_tokens,output_tokens,cost_usd,latency_ms) values(?,?,?,?,?,?,?,?,?)"
    cursor.execute(sql_,(timestamp,question,answer,model_used,cache_hit,input_tokens,output_tokens,cost_usd,latency_ms))
    connection.commit()
    connection.close()


if __name__ == "__main__":
    init_db()
    print("Database is ready")
    log_request("What's the refund policy?",)
    
