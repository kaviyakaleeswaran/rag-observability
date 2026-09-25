from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# =============================
# Configuration
# =============================

CHUNKS_FILE = Path("data/chunks.json")

DB_FOLDER = "data/chroma_db"
COLLECTION_NAME = "cv_research_papers"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

VECTOR_TOP_K = 20
BM25_TOP_K = 20
RRF_TOP_K = 20
FINAL_TOP_K = 5

RRF_K = 60


# =============================
# Load chunks
# =============================

print("Loading chunks...")

chunks = json.loads(
    CHUNKS_FILE.read_text(encoding="utf-8")
)

documents = [
    chunk["text"]
    for chunk in chunks
]

chunk_lookup = {
    chunk["chunk_id"]: chunk
    for chunk in chunks
}


# =============================
# Build BM25
# =============================

print("Building BM25 index...")

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)


# =============================
# Load ChromaDB
# =============================

print("Loading ChromaDB...")

client = chromadb.PersistentClient(
    path=DB_FOLDER
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# =============================
# Load embedding model
# =============================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# =============================
# Load reranker
# =============================

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)


# =============================
# Get question
# =============================

question = input(
    "\nEnter your question: "
)


# =============================
# VECTOR SEARCH
# =============================

print("\nRunning vector search...")

question_embedding = embedding_model.encode(
    [question]
)[0].tolist()

vector_results = collection.query(
    query_embeddings=[question_embedding],
    n_results=VECTOR_TOP_K
)

vector_ids = vector_results["ids"][0]


# =============================
# BM25 SEARCH
# =============================

print("Running BM25 search...")

query_tokens = question.lower().split()

bm25_scores = bm25.get_scores(
    query_tokens
)

bm25_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:BM25_TOP_K]

bm25_ids = [
    chunks[index]["chunk_id"]
    for index in bm25_indices
]


# =============================
# RECIPROCAL RANK FUSION
# =============================

print("Combining results using RRF...")

rrf_scores = {}


# Vector results

for rank, chunk_id in enumerate(
    vector_ids,
    start=1
):

    score = 1 / (RRF_K + rank)

    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + score
    )


# BM25 results

for rank, chunk_id in enumerate(
    bm25_ids,
    start=1
):

    score = 1 / (RRF_K + rank)

    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + score
    )


# Get top RRF candidates

rrf_candidates = sorted(
    rrf_scores,
    key=rrf_scores.get,
    reverse=True
)[:RRF_TOP_K]


# =============================
# CROSS-ENCODER RERANKING
# =============================

print("Reranking candidates...")

candidate_pairs = []

for chunk_id in rrf_candidates:

    candidate_text = chunk_lookup[
        chunk_id
    ]["text"]

    candidate_pairs.append(
        [question, candidate_text]
    )


reranker_scores = reranker.predict(
    candidate_pairs
)


# Combine IDs with reranker scores

reranked = list(
    zip(
        rrf_candidates,
        reranker_scores
    )
)


# Sort by reranker score

reranked.sort(
    key=lambda x: x[1],
    reverse=True
)


final_results = reranked[:FINAL_TOP_K]


# =============================
# Display final results
# =============================

print("\n")
print("=" * 70)
print("FINAL RERANKED RESULTS")
print("=" * 70)


for rank, (chunk_id, score) in enumerate(
    final_results,
    start=1
):

    chunk = chunk_lookup[chunk_id]

    print("\n" + "=" * 70)

    print("Rank:", rank)

    print("Chunk ID:", chunk_id)

    print("Source:", chunk["source"])

    print("Reranker Score:", float(score))

    print("\nText:")

    print(chunk["text"])


print("\n" + "=" * 70)
print("Retrieval + Reranking completed.")
print("=" * 70)