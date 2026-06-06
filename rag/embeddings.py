from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import API_KEY

class GeminiEmbeddingFunction:
    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=API_KEY)
    def __call__(self, input):
        return self.embedding_model.embed_documents(input)
    def embed_query(self, input):
        return [self.embedding_model.embed_query(input)]
    def name(self):
        return "gemini_embeddings"