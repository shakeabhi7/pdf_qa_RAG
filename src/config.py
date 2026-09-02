import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API Key not found")

#paths
DATA_DIR = "data"
FAISS_INDEX_DIR = "faiss_index"

#chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

#models settings
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

#Retrieval settings
RETRIEVAL_K = 4


