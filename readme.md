# 🧠 Agentic RAG Research Assistant

## 📖 Overview

The **Agentic RAG Research Assistant** is an advanced, production-ready AI application designed to accelerate academic research. By combining **Agentic workflows (LangGraph)** with a **Hybrid Retrieval-Augmented Generation (RAG) pipeline**, this assistant intelligently retrieves, evaluates, and synthesizes complex academic papers from ArXiv.

Unlike standard RAG applications that naively inject documents into a prompt, this system features an autonomous agent capable of self-correction. It evaluates document relevance, rewrites poorly performing queries, and falls back to live web searches if local knowledge is insufficient. The codebase is heavily modularized, emphasizing clean architecture, separation of concerns, and scalability.

---

## ✨ Key Features

* **Agentic State Machine:** Powered by LangGraph, the assistant features deterministic routing, document grading, query rewriting, and automatic web-search fallbacks.
* **Advanced Hybrid Retrieval:** Combines dense vector search (ChromaDB + Gemini Embeddings) with sparse keyword search (BM25) to capture both semantic meaning and exact terminology.
* **Cross-Encoder Reranking:** Utilizes `ms-marco-MiniLM-L-6-v2` to intelligently rerank retrieved chunks, ensuring only the highest-fidelity context is passed to the LLM.
* **Automated Data Ingestion pipeline:** Seamlessly search, download, and parse ArXiv PDFs into machine-readable Markdown using PyMuPDF4LLM.
* **Modular Architecture:** Designed for enterprise scalability. Tools, retrieval logic, agent states, and UI components are strictly isolated for easy testing and iteration.
* **Rich UI/UX:** A comprehensive Streamlit interface featuring dynamic token counting, chat memory management, multi-tab data ingestion, and real-time processing indicators.

---

## 🏗️ System Architecture

The core of the application is a directed graph (LangGraph) that models the cognitive process of a human researcher:

1. **Dynamic Routing:** The agent analyzes the user's query and routes it either to a conversational node (for greetings/history) or the search node (for academic inquiries).
2. **Hybrid Search & Rerank:** The query is expanded and executed against ChromaDB and a BM25 index. Results are aggregated, deduplicated, and reranked.
3. **Context Grading:** An LLM evaluator checks if the retrieved documents actually answer the user's question.
4. **Self-Correction Loop:** If the documents are irrelevant, the agent rewrites the query and tries again. If it fails twice, it triggers a Tavily web search fallback.
5. **Generation:** Synthesizes the final answer with strict source citations.

---

## 📂 Project Structure

```text
research_assistant/
│
├── data/                   # Dynamic: Raw PDFs and Markdowns
├── workspace_data/         # Dynamic: ChromaDB persistence and BM25 indices
├── papers/                 # Dynamic: Downloaded ArXiv papers
│
├── config.py               # Global configurations, environment variables, and clients
├── utils.py                # Shared utilities and parsers
├── tools/                  
│   ├── __init__.py
│   ├── arxiv_client.py     # ArXiv API integrations and PDF downloading
│   └── web_search.py       # External web search handlers
│
├── rag/                    
│   ├── __init__.py
│   ├── embeddings.py       # Gemini Embedding initialization
│   ├── reranker.py         # CrossEncoder loading and caching
│   └── vector_store.py     # Hybrid search logic (Chroma + BM25)
│
├── agent/                  
│   ├── __init__.py
│   ├── state.py            # LangGraph TypedDict state definition
│   ├── nodes.py            # Cognitive nodes (Retrieve, Grade, Generate, etc.)
│   └── graph.py            # Workflow compilation and edge routing
│
├── app.py                  # Streamlit entry point and UI layout
├── requirements.txt        # Project dependencies
└── .env                    # Secret keys

```

---

## 🚀 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/agentic-rag-researcher.git
cd agentic-rag-researcher

```

**2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

**3. Install Dependencies**

```bash
pip install -r requirements.txt

```

**4. Configure Environment Variables**
Create a `.env` file in the root directory and add your API keys:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
HF_HUB_DOWNLOAD_TIMEOUT=600

```

**5. Run the Application**

```bash
streamlit run app.py

```

---

## 🛠️ Tech Stack

| Domain | Technology |
| --- | --- |
| **Frontend UI** | Streamlit |
| **Agent Framework** | LangGraph, LangChain |
| **LLM & Embeddings** | Google Gemini (1.5 Flash / Embeddings-v2) |
| **Vector Database** | ChromaDB (Persistent) |
| **Keyword Search** | rank_bm25 |
| **Reranking Model** | SentenceTransformers (CrossEncoder) |
| **External APIs** | ArXiv API, Tavily Search API |
| **Document Processing** | PyMuPDF (fitz), PyMuPDF4LLM |

---

## 💡 Usage Guide

1. **Ingest Data:** Navigate to the **Search Papers** or **ArXiv Downloader** tab to find and download relevant academic papers.
2. **Process Vectors:** Go to the **Ingest Papers** tab and click "Start Ingestion" to parse the PDFs, chunk the text, generate embeddings, and build the BM25 index.
3. **Chat & Research:** Return to the **Research Agent** tab to interact with your local knowledge base. Watch the terminal or UI spinners to observe the agent routing, retrieving, and grading in real-time.