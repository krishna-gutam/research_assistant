import os
import pickle
import chromadb
import numpy as np
from config import DB_PATH, BM25_PATH
from rag.embeddings import GeminiEmbeddingFunction
from rag.reranker import reranker

def hybrid_search(query: str, top_k: int = 5):
    """Executes a hybrid search using ChromaDB (Vector) and BM25 (Keyword), then reranks."""
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection("papers_all_pages", embedding_function=GeminiEmbeddingFunction())
        
        # 1. Vector Search
        vector_results = collection.query(query_texts=[query], n_results=10)
        vector_docs = vector_results.get("documents", [[]])[0] if vector_results.get("documents") else []
        vector_metas = vector_results.get("metadatas", [[]])[0] if vector_results.get("metadatas") else []

        # 2. BM25 Search
        bm25_docs, bm25_metas = [], []
        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, "rb") as f:
                bm25_data = pickle.load(f)
            bm25 = bm25_data["bm25"]
            doc_scores = bm25.get_scores(query.lower().split(" "))
            top_bm25_indices = np.argsort(doc_scores)[::-1][:10]
            bm25_docs = [bm25_data["docs"][i] for i in top_bm25_indices]
            bm25_metas = [bm25_data["metas"][i] for i in top_bm25_indices]

        # 3. Combine and Deduplicate
        combined_docs, combined_metas, seen_texts = [], [], set()
        for doc, meta in zip(vector_docs + bm25_docs, vector_metas + bm25_metas):
            if doc not in seen_texts:
                seen_texts.add(doc)
                combined_docs.append(doc)
                combined_metas.append(meta)

        if not combined_docs:
            return []
            
        # 4. Rerank
        pairs = [(query, doc) for doc in combined_docs]
        scores = reranker.predict(pairs)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [{"content": combined_docs[i], "meta": combined_metas[i]} for i in top_indices]
        
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return []