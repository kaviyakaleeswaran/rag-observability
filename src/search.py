from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DB_FOLDER = "data/chroma_db"
COLLECTION_NAME = "cv_research_papers"

MODEL_NAME = "all-MiniLM-L6-v2"


client = chromadb.PersistentClient(
    path=DB_FOLDER
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

model = SentenceTransformer(MODEL_NAME)


question = input("Enter your question: ")

question_embedding = model.encode(
    [question]
)[0].tolist()


results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5
)


print("\nTop results:\n")

for i in range(5):

    print("=" * 70)

    print("Rank:", i + 1)

    print("Chunk ID:", results["ids"][0][i])

    print("Source:", results["metadatas"][0][i]["source"])

    print("Distance:", results["distances"][0][i])

    print("\nText:")

    print(results["documents"][0][i])

    print()