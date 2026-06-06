import os
import re
import time
import requests
import fitz
import pickle
import chromadb
import streamlit as st
from rank_bm25 import BM25Okapi

from config import PDF_DIR, DB_PATH, BM25_PATH, tavily_client
from agent.graph import build_graph
from tools.arxiv_client import search_arxiv_by_query, search_arxiv_by_id, download_and_convert_paper
from rag.embeddings import GeminiEmbeddingFunction

# --- Initialization ---
app_workflow = build_graph()
st.set_page_config(page_title="Agentic RAG Research Assistant", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar UI ---
with st.sidebar:
    st.subheader("📊 Token Statistics")
    total_tokens = sum(len(msg["content"]) // 4 for msg in st.session_state.messages)
    st.metric("Estimated Total Tokens", total_tokens)

    st.session_state["n_results"] = st.number_input(
        "Number of retrieved documents", min_value=1, max_value=20, value=st.session_state.get("n_results", 2), step=1
    )

    if st.button("Undo Last Turn") and len(st.session_state.messages) >= 2:
        st.session_state.messages = st.session_state.messages[:-2]
        st.rerun()

    if st.button("Undo First Turn") and len(st.session_state.messages) >= 2:
        st.session_state.messages = st.session_state.messages[2:]
        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Layout & Tabs ---
tab1, tab_search, tab_tavily, tab_arxiv, tab_ingest = st.tabs([
    "Research Agent", "Search Papers", "Tavily Search", "ArXiv Downloader", "Ingest Papers"
])

with tab_tavily:
    st.subheader("🌐 Tavily Web Search")
    tavily_query = st.text_input("Enter research topic to find papers:")
    if st.button("Search Web") and tavily_query:
        with st.spinner("Searching web..."):
            tavily_results = tavily_client.search(f"arXiv paper {tavily_query}")
            match = re.search(r'(\d{4}\.\d{4,5})', str(tavily_results))
            if match:
                paper_id = match.group(1)
                st.success(f"Found potential arXiv ID: {paper_id}")
                st.session_state.tavily_results = search_arxiv_by_id(paper_id)
            else:
                st.warning("No arXiv ID found in search results.")

    if "tavily_results" in st.session_state:
        for paper in st.session_state.tavily_results:
            st.write(f"**{paper.title}**")
            if st.button("Download this paper", key=f"dl_{paper.entry_id}"):
                with requests.Session() as session:
                    download_and_convert_paper(paper, session)
                st.success("Finished processing paper.")

with tab_search:
    st.subheader("🔍 Search ArXiv")
    query = st.text_input("Enter search query:")
    if st.button("Search") and query:
        results = search_arxiv_by_query(query, max_results=5)
        if results:
            st.session_state.search_results = results
        else:
            st.warning("No results found.")
    
    if "search_results" in st.session_state:
        selected_papers = [
            paper for i, paper in enumerate(st.session_state.search_results)
            if st.checkbox(f"{paper.title} ({paper.published.year})", key=f"check_{i}")
        ]
        if st.button("Download Selected"):
            with requests.Session() as session:
                for paper in selected_papers:
                    download_and_convert_paper(paper, session)
            st.success("Finished processing selected papers.")

with tab_arxiv:
    st.subheader("📚 ArXiv Paper Downloader")
    paper_ids = st.text_area("Enter ArXiv IDs (comma-separated):", placeholder="1706.03762, 2605.19577")
    if st.button("Download Papers"):
        if not paper_ids.strip():
            st.warning("Please enter at least one paper ID.")
            st.stop()
        
        ids = [i.strip() for i in paper_ids.split(",") if i.strip()]
        progress = st.progress(0)
        
        with requests.Session() as session:
            for idx, pid in enumerate(ids):
                try:
                    st.write(f"🔄 Processing `{pid}`...")
                    results = search_arxiv_by_id(pid)
                    if not results:
                        st.error(f"No paper found: {pid}")
                        continue
                    download_and_convert_paper(results[0], session)
                    progress.progress((idx + 1) / len(ids))
                except Exception as e:
                    st.error(f"Error processing {pid}: {e}")
        st.success("✅ Finished processing papers")

with tab_ingest:
    st.subheader("📥 Ingest Papers into Vector DB")
    if st.button("Start Ingestion"):
        if not os.path.exists(PDF_DIR):
            st.error(f"Directory not found: {PDF_DIR}")
        else:
            client = chromadb.PersistentClient(path=DB_PATH)
            collection = client.get_or_create_collection(name="papers_all_pages", embedding_function=GeminiEmbeddingFunction())
            files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
            
            progress_bar = st.progress(0)
            all_texts_for_bm25, all_metas_for_bm25 = [], []
            
            for i, filename in enumerate(files):
                filepath = os.path.join(PDF_DIR, filename)
                st.write(f"Processing: {filename}")
                try:
                    doc = fitz.open(filepath)
                    for page_num, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            page_id = f"{filename}_page_{page_num}"
                            meta = {"source": filename, "page": page_num}
                            collection.upsert(documents=[text], metadatas=[meta], ids=[page_id])
                            all_texts_for_bm25.append(text)
                            all_metas_for_bm25.append(meta)
                            time.sleep(0.5) 
                    doc.close()
                    st.success(f"Ingested: {filename}")
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                progress_bar.progress((i + 1) / len(files))

            st.write("Building BM25 Keyword Index...")
            tokenized_corpus = [doc.lower().split(" ") for doc in all_texts_for_bm25]
            if tokenized_corpus:
                bm25 = BM25Okapi(tokenized_corpus)
                with open(BM25_PATH, "wb") as f:
                    pickle.dump({"bm25": bm25, "docs": all_texts_for_bm25, "metas": all_metas_for_bm25}, f)
            st.success("✅ Ingestion & Hybrid Indexing complete!")

with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Research Agent is thinking (Routing -> Grading -> Fallbacks)..."):
            try:
                result = app_workflow.invoke({
                    "query": prompt,
                    "chat_history": st.session_state.messages[:-1],
                    "documents": [],
                    "loop_count": 0,
                    "web_fallback": False
                })
                response = result.get('answer', 'No answer generated.')
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")