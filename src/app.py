import streamlit as st
import pandas as pd
import sqlite3
import sys
import json

# Allow importing files from src
sys.path.insert(0, "src")

from pipeline import run_pipeline


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# Load chunk data
# ---------------------------------------------------------

with open(
    "data/chunks.json",
    "r",
    encoding="utf-8"
) as file:

    all_chunks = json.load(file)


chunk_text_lookup = {
    chunk["chunk_id"]: chunk["text"]
    for chunk in all_chunks
}


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("📚 RAG Document Intelligence System")

st.write(
    "Ask questions about the research papers in the knowledge base. "
    "The system uses hybrid retrieval, cross-encoder reranking, "
    "LLM generation, citation verification, and observability."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("System Pipeline")

st.sidebar.markdown(
    """
    **1.** Vector Search  
    ↓  
    **2.** BM25 Keyword Search  
    ↓  
    **3.** Hybrid RRF Retrieval  
    ↓  
    **4.** Cross-Encoder Reranking  
    ↓  
    **5.** Llama 3.2 Generation  
    ↓  
    **6.** Citation Verification  
    ↓  
    **7.** SQLite Observability
    """
)


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is the main idea behind ResNet?"
)


# ---------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------

if st.button("🔍 Ask Question"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Running retrieval, reranking and LLM generation..."
        ):

            try:

                result = run_pipeline(question)

                st.session_state["result"] = result

            except Exception as error:

                st.error(
                    f"An error occurred while processing the question: {error}"
                )


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]


    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    st.subheader("💡 Answer")

    st.write(result["answer"])


    # -----------------------------------------------------
    # Citation information
    # -----------------------------------------------------

    st.subheader("📌 Citation Verification")

    valid_citations = result["valid_citations"]
    invalid_citations = result["invalid_citations"]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Valid Citations",
            len(valid_citations)
        )

    with col2:

        st.metric(
            "Invalid Citations",
            len(invalid_citations)
        )


    if valid_citations:

        st.success(
            "Valid citations: "
            + ", ".join(valid_citations)
        )


    if invalid_citations:

        st.error(
            "Invalid citations: "
            + ", ".join(invalid_citations)
        )


    # -----------------------------------------------------
    # Retrieved chunks
    # -----------------------------------------------------

    st.subheader("🔎 Retrieved Evidence")

    retrieved_chunks = result["retrieved_chunks"]

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        chunk_id = chunk[0]
        score = chunk[1]

        chunk_text = chunk_text_lookup.get(
            chunk_id,
            "Chunk text not found."
        )

        with st.expander(
            f"{index}. {chunk_id} — Score: {score:.4f}"
        ):

            st.write(
                chunk_text
            )


    # -----------------------------------------------------
    # Performance / Observability
    # -----------------------------------------------------

    st.subheader("📊 Pipeline Performance")

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:

        st.metric(
            "Vector Search",
            f"{result['vector_latency']:.2f}s"
        )

    with metric2:

        st.metric(
            "BM25",
            f"{result['bm25_latency']:.2f}s"
        )

    with metric3:

        st.metric(
            "Reranker",
            f"{result['reranker_latency']:.2f}s"
        )

    with metric4:

        st.metric(
            "LLM",
            f"{result['llm_latency']:.2f}s"
        )

    with metric5:

        st.metric(
            "Total",
            f"{result['total_latency']:.2f}s"
        )


# ---------------------------------------------------------
# Observability Dashboard
# ---------------------------------------------------------

st.divider()

st.subheader("📈 Observability Dashboard")

try:

    connection = sqlite3.connect(
        "data/observability.db"
    )

    query = """
        SELECT
            id,
            question,
            vector_latency,
            bm25_latency,
            reranker_latency,
            llm_latency,
            total_latency,
            retrieved_chunks,
            valid_citations,
            invalid_citations,
            created_at
        FROM pipeline_logs
        ORDER BY id DESC
    """

    logs = pd.read_sql_query(
        query,
        connection
    )

    connection.close()


    if len(logs) == 0:

        st.info(
            "No pipeline runs have been logged yet."
        )

    else:

        # Summary metrics

        total_runs = len(logs)

        average_latency = logs[
            "total_latency"
        ].mean()

        average_reranker = logs[
            "reranker_latency"
        ].mean()

        average_llm = logs[
            "llm_latency"
        ].mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Runs",
                total_runs
            )

        with col2:

            st.metric(
                "Average Total Latency",
                f"{average_latency:.2f}s"
            )

        with col3:

            st.metric(
                "Average Reranker",
                f"{average_reranker:.2f}s"
            )

        with col4:

            st.metric(
                "Average LLM",
                f"{average_llm:.2f}s"
            )


        # Latency chart

        st.markdown(
            "### Total Latency per Query"
        )

        chart_data = logs[
            [
                "id",
                "total_latency"
            ]
        ].set_index("id")

        st.line_chart(
            chart_data
        )


        # Logs table

        st.markdown(
            "### Pipeline Logs"
        )

        st.dataframe(
            logs,
            use_container_width=True
        )


except Exception as error:

    st.warning(
        f"Observability data could not be loaded: {error}"
    )