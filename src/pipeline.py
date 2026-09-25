import json
import sqlite3
import subprocess
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from src.citation_utils import extract_citations

# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = Path("data/chunks.json")
METADATA_FILE = Path("data/paper_metadata.json")

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "cv_research_papers"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "llama3.2:3b"

VECTOR_TOP_K = 20
BM25_TOP_K = 20
RRF_TOP_K = 20
FINAL_TOP_K = 5

RRF_K = 60

OBSERVABILITY_DB = "data/observability.db"


# ============================================================
# LOAD DATA
# ============================================================

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


# ============================================================
# BUILD LOOKUPS
# ============================================================

chunk_lookup = {
    chunk["chunk_id"]: chunk
    for chunk in chunks
}

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

chunk_ids = [
    chunk["chunk_id"]
    for chunk in chunks
]


# ============================================================
# BUILD BM25 INDEX
# ============================================================

tokenized_texts = [
    text.lower().split()
    for text in chunk_texts
]

bm25 = BM25Okapi(
    tokenized_texts
)


# ============================================================
# LOAD CHROMA
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# LOAD MODELS
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

reranker = CrossEncoder(
    RERANKER_MODEL
)


# ============================================================
# CITATION PARSER
# ============================================================

# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def run_pipeline(question):

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )


    # ========================================================
    # START TOTAL TIMER
    # ========================================================

    pipeline_start = time.perf_counter()


    # ========================================================
    # 1. VECTOR SEARCH
    # ========================================================

    vector_start = time.perf_counter()

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=VECTOR_TOP_K
    )

    vector_end = time.perf_counter()

    vector_latency = (
        vector_end - vector_start
    )


    vector_ranked = []

    for i, chunk_id in enumerate(
        vector_results["ids"][0]
    ):

        vector_ranked.append(
            (
                chunk_id,
                i + 1
            )
        )


    # ========================================================
    # 2. BM25 SEARCH
    # ========================================================

    bm25_start = time.perf_counter()

    bm25_scores = bm25.get_scores(
        question.lower().split()
    )

    bm25_sorted_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_ranked = []

    for rank, index in enumerate(
        bm25_sorted_indices[:BM25_TOP_K],
        start=1
    ):

        bm25_ranked.append(
            (
                chunk_ids[index],
                rank
            )
        )

    bm25_end = time.perf_counter()

    bm25_latency = (
        bm25_end - bm25_start
    )


    # ========================================================
    # 3. RECIPROCAL RANK FUSION
    # ========================================================

    rrf_scores = {}

    for chunk_id, rank in vector_ranked:

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += (
            1 / (RRF_K + rank)
        )


    for chunk_id, rank in bm25_ranked:

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += (
            1 / (RRF_K + rank)
        )


    rrf_ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    rrf_ranked = rrf_ranked[
        :RRF_TOP_K
    ]


    # ========================================================
    # 4. CROSS-ENCODER RERANKING
    # ========================================================

    reranker_start = time.perf_counter()

    reranker_inputs = []

    for chunk_id, score in rrf_ranked:

        text = chunk_lookup[
            chunk_id
        ]["text"]

        reranker_inputs.append(
            (
                question,
                text
            )
        )


    reranker_scores = reranker.predict(
        reranker_inputs
    )


    reranked_results = []

    for i, (chunk_id, rrf_score) in enumerate(
        rrf_ranked
    ):

        reranked_results.append(
            (
                chunk_id,
                float(
                    reranker_scores[i]
                )
            )
        )


    reranked_results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    final_results = reranked_results[
        :FINAL_TOP_K
    ]

    reranker_end = time.perf_counter()

    reranker_latency = (
        reranker_end - reranker_start
    )


    # ========================================================
    # 5. BUILD CONTEXT
    # ========================================================

    context_parts = []

    for chunk_id, score in final_results:

        chunk = chunk_lookup[
            chunk_id
        ]

        paper_key = chunk[
            "source"
        ]

        paper_info = metadata.get(
            paper_key,
            {}
        )

        title = paper_info.get(
            "title",
            paper_key
        )

        context_parts.append(
            f"""
PAPER TITLE:
{title}

CITATION ID:
{chunk_id}

CONTENT:
{chunk["text"]}
"""
        )


    context = "\n".join(
        context_parts
    )


    # ========================================================
    # 6. VALID CITATION IDS
    # ========================================================

    retrieved_chunk_ids = {
        chunk_id
        for chunk_id, score in final_results
    }

    citation_list = "\n".join(
        f"- {chunk_id}"
        for chunk_id, score in final_results
    )


    # ========================================================
    # 7. RAG PROMPT
    # ========================================================

    prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY the
research-paper context provided below.

Do not use outside knowledge.

Answer concisely and explain the answer
in your own words.

After the answer, add a section exactly named:

CITATIONS:

Under CITATIONS, list ONLY citation IDs
from the following allowed list:

{citation_list}

Only cite chunks that directly support
your answer.

Use this exact format:

CITATIONS:
- faster_rcnn_0
- faster_rcnn_2

Do not invent citation IDs.

Do not use placeholder IDs such as:
chunk_id_1
chunk_id_2

Do not include paper titles under CITATIONS.

Do not write anything after the citation list.

If the provided context is insufficient,
say that the provided research papers do
not contain enough information to answer
the question.

RESEARCH-PAPER CONTEXT:

{context}

USER QUESTION:

{question}
"""


    # ========================================================
    # 8. LLM GENERATION
    # ========================================================

    llm_start = time.perf_counter()

    result = subprocess.run(
        [
            "ollama",
            "run",
            OLLAMA_MODEL
        ],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180
    )

    llm_end = time.perf_counter()

    llm_latency = (
        llm_end - llm_start
    )


    if result.returncode != 0:

        raise RuntimeError(
            f"Ollama error: {result.stderr}"
        )


    answer_text = result.stdout.strip()


    # ========================================================
    # 9. CITATION VERIFICATION
    # ========================================================

    citation_ids = extract_citations(
        answer_text
    )


    valid_citations = [
        citation_id
        for citation_id in citation_ids
        if citation_id in retrieved_chunk_ids
    ]


    invalid_citations = [
        citation_id
        for citation_id in citation_ids
        if citation_id not in retrieved_chunk_ids
    ]


    # ========================================================
    # 10. TOTAL LATENCY
    # ========================================================

    pipeline_end = time.perf_counter()

    total_latency = (
        pipeline_end - pipeline_start
    )


    # ========================================================
    # 11. SAVE OBSERVABILITY
    # ========================================================

    save_observability(
        question,
        vector_latency,
        bm25_latency,
        reranker_latency,
        llm_latency,
        total_latency,
        len(retrieved_chunk_ids),
        len(valid_citations),
        len(invalid_citations)
    )


    # ========================================================
    # 12. RETURN RESULTS
    # ========================================================

    return {
        "answer": answer_text,
        "citations": citation_ids,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "retrieved_chunks": final_results,
        "vector_latency": vector_latency,
        "bm25_latency": bm25_latency,
        "reranker_latency": reranker_latency,
        "llm_latency": llm_latency,
        "total_latency": total_latency
    }


# ============================================================
# OPTIONAL COMMAND-LINE MODE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RAG PIPELINE")
    print("=" * 70)

    question = input(
        "\nEnter your research question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        exit()


    result = run_pipeline(
        question
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "RAG ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        result["answer"]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "CITATION VERIFICATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Retrieved chunks: "
        f"{len(result['retrieved_chunks'])}"
    )

    print(
        f"Valid citations: "
        f"{len(result['valid_citations'])}"
    )

    print(
        f"Invalid citations: "
        f"{len(result['invalid_citations'])}"
    )

    print(
        f"Total latency: "
        f"{result['total_latency']:.2f} seconds"
    )


    print(
        "\nVerified citation IDs:"
    )

    for citation in result[
        "valid_citations"
    ]:

        print(
            f"- {citation}"
        )


    print(
        "\nRAG PIPELINE COMPLETED SUCCESSFULLY."
    )