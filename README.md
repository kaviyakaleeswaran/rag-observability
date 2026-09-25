\# Hybrid RAG-Based Document Question Answering System with Observability



A research-paper question answering system that combines \*\*semantic vector search, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, local LLM generation, citation validation, and pipeline observability\*\*.



The system is built using real computer-vision research papers and provides answers together with the retrieved evidence and performance metrics.



\## Overview



Traditional RAG systems often rely only on vector similarity. This project uses a \*\*hybrid retrieval pipeline\*\* so that both semantic meaning and exact technical terms can contribute to document retrieval.



The complete pipeline is:



```text

User Question

&#x20;     ↓

Vector Search

&#x20;     +

BM25 Keyword Search

&#x20;     ↓

Reciprocal Rank Fusion (RRF)

&#x20;     ↓

Cross-Encoder Reranking

&#x20;     ↓

Top Evidence Chunks

&#x20;     ↓

Local Llama 3.2 LLM

&#x20;     ↓

Citation Extraction \& Validation

&#x20;     ↓

Answer + Evidence + Metrics

&#x20;     ↓

SQLite Observability

```



\## Key Features



\* PDF document ingestion and text extraction

\* Text chunking for retrieval

\* Sentence-transformer embeddings

\* ChromaDB vector search

\* BM25 keyword retrieval

\* Hybrid retrieval using Reciprocal Rank Fusion

\* Cross-encoder reranking

\* Local Llama 3.2 generation through Ollama

\* Citation ID extraction and validation

\* SQLite-based pipeline observability

\* Streamlit interface

\* Retrieval evidence display

\* Per-stage latency measurement

\* Evaluation set with automated pass/fail checking



\## Real Dataset



The system was tested using five real computer-vision research papers:



| Paper                                                                          | Topic                                      | Source           |

| ------------------------------------------------------------------------------ | ------------------------------------------ | ---------------- |

| Deep Residual Learning for Image Recognition                                   | Residual Networks / Image Classification   | arXiv:1512.03385 |

| Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | Object Detection                           | arXiv:1506.01497 |

| U-Net: Convolutional Networks for Biomedical Image Segmentation                | Image Segmentation                         | arXiv:1505.04597 |

| You Only Look Once: Unified, Real-Time Object Detection                        | Real-Time Object Detection                 | arXiv:1506.02640 |

| An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale     | Vision Transformers / Image Classification | arXiv:2010.11929 |



The paper metadata and source URLs are stored in:



```text

data/paper\_metadata.json

```



The downloaded PDFs are intentionally excluded from the Git repository through `.gitignore`.



\## Project Structure



```text

rag-observability/

│

├── data/

│   ├── evaluation.json

│   └── paper\_metadata.json

│

├── src/

│   ├── app.py

│   ├── bm25\_search.py

│   ├── chunk.py

│   ├── embed.py

│   ├── eval\_full.py

│   ├── hybrid\_search.py

│   ├── ingest.py

│   ├── observability.py

│   ├── pipeline.py

│   ├── search.py

│   └── vector\_store.py

│

├── tests/

├── test\_pdf.py

├── .gitignore

└── README.md

```



Generated files such as embeddings, extracted text, ChromaDB data, and SQLite logs are excluded from version control.



\## Technologies Used



\* \*\*Python\*\*

\* \*\*Streamlit\*\*

\* \*\*ChromaDB\*\*

\* \*\*Sentence Transformers\*\*

\* \*\*BM25\*\*

\* \*\*Cross-Encoder\*\*

\* \*\*Ollama\*\*

\* \*\*Llama 3.2\*\*

\* \*\*SQLite\*\*

\* \*\*PyPDF\*\*

\* \*\*NumPy\*\*

\* \*\*Pandas\*\*



\## Retrieval Pipeline



\### 1. Vector Retrieval



Documents are converted into embeddings using:



```text

all-MiniLM-L6-v2

```



The embeddings are stored in ChromaDB and used for semantic similarity search.



\### 2. BM25 Retrieval



BM25 provides keyword-based retrieval and helps when questions contain important technical terms or exact phrases.



\### 3. Hybrid Retrieval



The vector-search and BM25 rankings are combined using \*\*Reciprocal Rank Fusion (RRF)\*\*.



This allows the system to benefit from both:



\* semantic similarity

\* keyword matching



\### 4. Cross-Encoder Reranking



The hybrid results are reranked using:



```text

cross-encoder/ms-marco-MiniLM-L-6-v2

```



The reranker evaluates the relationship between the question and retrieved chunk text before selecting the final evidence.



\## Answer Generation



The final evidence is provided to a local:



```text

Llama 3.2 3B

```



model through Ollama.



The generation prompt instructs the model to answer using the retrieved evidence and provide citation IDs corresponding to retrieved chunks.



No paid external LLM API is required for the current implementation.



\## Citation Validation



The system extracts citation IDs from the generated answer and checks whether they correspond to retrieved chunks.



For example:



```text

CITATIONS:

\- resnet\_38

\- resnet\_35

```



The pipeline reports:



\* valid citations

\* invalid citations

\* retrieved evidence



\### Important limitation



The current citation verifier checks \*\*citation ID validity and retrieval membership\*\*. It does not perform a semantic fact-check of whether every generated statement is actually supported by the cited text.



Therefore, the project does not claim to be hallucination-proof.



\## Observability



Each pipeline execution records performance information in SQLite.



Tracked metrics include:



\* Vector search latency

\* BM25 latency

\* Cross-encoder reranking latency

\* LLM latency

\* Total pipeline latency

\* Number of retrieved chunks

\* Number of valid citations

\* Number of invalid citations

\* Timestamp



The Streamlit application displays these metrics together with the generated answer and retrieved evidence.



## Evaluation

The project includes a 10-question end-to-end evaluation set covering:

- ResNet
- Faster R-CNN
- U-Net
- YOLO
- Vision Transformer

A question passes when:

1. The expected paper/source appears in the retrieved evidence.
2. At least one valid citation is produced.
3. No invalid citations are produced.

Current full-pipeline evaluation result:

```text
Questions evaluated: 10
Passed:               10
Failed:                0
Pass rate:            100%
Required threshold:    70%
Status:               PASSED
\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/kaviyakaleeswaran/rag-observability.git

cd rag-observability

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



Activate it on Windows:



```cmd

venv\\Scripts\\activate

```



\### 3. Install dependencies



```bash

pip install chromadb sentence-transformers rank\_bm25 langchain langchain-community streamlit numpy pandas pypdf ollama torchvision

```



\### 4. Install Ollama



Install Ollama separately and pull the local model:



```bash

ollama pull llama3.2:3b

```



Make sure Ollama is running before executing the question-answering pipeline.



\## Preparing the Dataset



Place the research-paper PDFs inside:



```text

docs/

```



Then run the ingestion and indexing stages.



The generated files are stored locally under `data/` and are excluded from Git.



\## Running the Application



Start the Streamlit application:



```bash

streamlit run src/app.py

```



The application provides:



\* Question input

\* Generated answer

\* Citation verification

\* Retrieved evidence

\* Retrieval scores

\* Per-stage latency

\* Observability statistics



\## Running Evaluation



Run:



```bash

python src/eval\_full.py

```



The evaluation checks the configured questions and exits with a failure status when the pass rate falls below the required threshold.



\## Example Questions



Example questions include:



```text

What problem does ResNet solve?



How do residual learning and shortcut connections help train very deep networks?



What is the main idea behind the Region Proposal Network in Faster R-CNN?



What is the main purpose of the U-Net architecture?



Why can YOLO perform object detection in real time?



How does Vision Transformer apply the Transformer architecture to images?

```



\## Why This Project



This project demonstrates an end-to-end implementation of a modern RAG system rather than only calling an LLM API.



It combines:



```text

Information Retrieval

&#x20;       +

Natural Language Processing

&#x20;       +

LLM Application Development

&#x20;       +

Machine Learning

&#x20;       +

Software Engineering

&#x20;       +

Observability

&#x20;       +

Automated Evaluation

```



The project is particularly relevant to applications involving technical-document search, research assistance, enterprise knowledge bases, and question answering over private document collections.



\## Future Improvements



Possible extensions include:



\* Larger document collections

\* More extensive evaluation datasets

\* Semantic citation/faithfulness verification

\* Retrieval quality metrics such as Recall@K and MRR

\* Better experiment tracking

\* Containerized deployment

\* Cloud deployment

\* CI-based automated evaluation

\* Support for additional document formats



\## Author



\*\*Kaviya K\*\*



B.E. Electronics \& Communication Engineering

Thiagarajar College of Engineering



