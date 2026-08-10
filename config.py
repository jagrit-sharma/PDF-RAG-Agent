import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBED_MODEL = "all-MiniLM-L6-v2"

CHAT_MODEL = "claude-opus-5"
MAX_TOKENS = 4096
TOP_K = 5

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

COLLECTION = f"mas_docs_{CHUNK_SIZE}"