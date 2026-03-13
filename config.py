import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
RESUMES_DIR = BASE_DIR / "resumes"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
RESUMES_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# ChromaDB settings
COLLECTION_NAME = "resumes"

# Ollama settings
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 3