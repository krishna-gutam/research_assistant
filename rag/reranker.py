import streamlit as st
from sentence_transformers import CrossEncoder

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

reranker = load_reranker()