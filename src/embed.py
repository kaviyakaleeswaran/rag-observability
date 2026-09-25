from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/chunks.json")
OUTPUT_FILE = Path("data/embeddings.json")

MODEL_NAME = "all-MiniLM-L6-v2"


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Reading chunks...")

chunks = json.loads(
    INPUT_FILE.read_text(encoding="utf-8")
)

texts = [chunk["text"] for chunk in chunks]

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

data = []

for chunk, embedding in zip(chunks, embeddings):

    data.append({
        "chunk_id": chunk["chunk_id"],
        "source": chunk["source"],
        "text": chunk["text"],
        "embedding": embedding.tolist()
    })


OUTPUT_FILE.write_text(
    json.dumps(data),
    encoding="utf-8"
)

print()
print("Total embeddings:", len(data))
print("Saved to:", OUTPUT_FILE)