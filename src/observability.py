import sqlite3
from pathlib import Path


# ==========================================
# Configuration
# ==========================================

DB_FOLDER = Path("data")
DB_FOLDER.mkdir(parents=True, exist_ok=True)

DB_FILE = DB_FOLDER / "observability.db"


# ==========================================
# Create database connection
# ==========================================

connection = sqlite3.connect(DB_FILE)

cursor = connection.cursor()


# ==========================================
# Create logs table
# ==========================================

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS pipeline_logs (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        question TEXT NOT NULL,

        vector_latency REAL,

        bm25_latency REAL,

        reranker_latency REAL,

        llm_latency REAL,

        total_latency REAL,

        retrieved_chunks INTEGER,

        valid_citations INTEGER,

        invalid_citations INTEGER,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
)


# ==========================================
# Save changes
# ==========================================

connection.commit()

connection.close()


print("=" * 60)
print("Observability database created successfully.")
print("=" * 60)

print(
    f"Database location: {DB_FILE}"
)