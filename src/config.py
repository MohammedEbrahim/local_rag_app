from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DOCS_DIR = BASE_DIR / "data" / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"

OLLAMA_BASE_URL = "http://localhost:11434"

DEFAULT_LLM_MODEL = "llama3:latest"
DEFAULT_JUDGE_MODEL = "mistral:latest"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4

SUPPORTED_EXT = [".pdf", ".txt", ".md"]

SYSTEM_PROMPT = """You are a helpful assistant.
You must answer using ONLY the provided context.
If the answer is not in the context, say:
"I don't know based on the provided documents."
"""
