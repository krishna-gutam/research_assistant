import os
import re
import arxiv
import requests
import pymupdf4llm
import streamlit as st
from config import PDF_DIR, MD_DIR

def search_arxiv_by_query(query: str, max_results: int = 1):
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        return list(client.results(search))
    except Exception as e:
        print(f"Error searching arXiv by query: {e}")
        return []

def search_arxiv_by_id(paper_id: str):
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        return list(client.results(search))
    except Exception as e:
        print(f"Error searching arXiv by ID {paper_id}: {e}")
        return []

def download_and_convert_paper(paper, session):
    safe_title = re.sub(r'[<>:"/\\|?*]', '', paper.title).strip()[:80]
    pid = paper.entry_id.split('/')[-1]
    
    pdf_filename = f"{safe_title}_{pid}.pdf"
    md_filename = f"{safe_title}_{pid}.md"
    
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    md_path = os.path.join(MD_DIR, md_filename)

    if not os.path.exists(pdf_path):
        response = session.get(paper.pdf_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
        st.success(f"Downloaded PDF: {safe_title}")
    else:
        st.info(f"PDF exists: {safe_title}")

    if not os.path.exists(md_path):
        md_text = pymupdf4llm.to_markdown(pdf_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        st.success(f"Converted: {safe_title}")
        
    return True