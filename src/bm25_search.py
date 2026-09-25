from pathlib import Path
import json

from rank_bm25 import BM25Okapi


INPUT_FILE = Path("data/chunks.json")


chunks = json.loads(
    INPUT_FILE.read_text(encoding="utf-8")
)


documents = [
    chunk["text"]
    for chunk in chunks
]


tokenized_documents = [
    document.lower().split()
    for document in documents
]


bm25 = BM25Okapi(tokenized_documents)


question = input("Enter your question: ")

query_tokens = question.lower().split()

scores = bm25.get_scores(query_tokens)


top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:5]


print("\nTop BM25 results:\n")


for rank, index in enumerate(top_indices, start=1):

    chunk = chunks[index]

    print("=" * 70)

    print("Rank:", rank)

    print("Chunk ID:", chunk["chunk_id"])

    print("Source:", chunk["source"])

    print("BM25 Score:", scores[index])

    print("\nText:")

    print(chunk["text"])

    print()