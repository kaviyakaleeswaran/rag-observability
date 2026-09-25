from pathlib import Path
import json

import chromadb


INPUT_FILE = Path("data/embeddings.json")
DB_FOLDER = "data/chroma_db"

COLLECTION_NAME = "cv_research_papers"


print("Loading embeddings...")

data = json.loads(
    INPUT_FILE.read_text(encoding="utf-8")
)


print("Creating ChromaDB...")

client = chromadb.PersistentClient(
    path=DB_FOLDER
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


ids = []
documents = []
metadatas = []
embeddings = []


for item in data:

    ids.append(item["chunk_id"])

    documents.append(item["text"])

    metadatas.append({
        "source": item["source"]
    })

    embeddings.append(item["embedding"])


print("Adding documents to ChromaDB...")

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


print()
print("Total documents in ChromaDB:", collection.count())
print("Database saved to:", DB_FOLDER)