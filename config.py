import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Constants
MODEL_ID = "gemini-3.1-flash-lite"
MAX_RESULTS = 3

# Directory Paths
PDF_DIR = "papers/pdf"
MD_DIR = "papers/md"
DB_PATH = "workspace_data/chroma_db"
BM25_PATH = "workspace_data/bm25_index.pkl"

# Ensure directories exist
os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/markdowns", exist_ok=True)
os.makedirs("workspace_data", exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)

# Global Clients
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)