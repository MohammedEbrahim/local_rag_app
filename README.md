# Local RAG Assistant

A small Streamlit app for local Retrieval-Augmented Generation (RAG) using Ollama for embeddings/LLM and FAISS for vector search.

Features
- Upload PDF / TXT / MD documents.
- Build a FAISS vectorstore from document chunks.
- Chat interface that retrieves context from your documents and streams answers.
- Optional Mistral-based "judge" to evaluate answers.

Quick start

1) Prerequisites
- Python 3.11+ recommended
- Ollama installed and running locally (default API: http://localhost:11434)
- Git (optional)

2) Install Python dependencies

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

3) Start Ollama (if not already running)

Follow Ollama's docs to install and run the Ollama server on your machine. The app expects Ollama at http://localhost:11434 by default. You can change the base URL in `src/config.py`.

4) Run the Streamlit app

```powershell
streamlit run app.py
```

5) Usage
- Open the app in your browser (Streamlit will show the local URL).
- Go to the "Documents" tab to upload files (.pdf, .txt, .md).
- Use "Rebuild Index" to create the FAISS index from uploaded docs.
- Switch to the "Chat" tab and ask questions. The assistant will answer using only the retrieved document context.
- Optionally enable the Judge in settings to evaluate answers using a Mistral model.

Project layout
- app.py — Streamlit entrypoint
- src/config.py — configuration and defaults
- src/rag/ingest.py — document loading, splitting, and FAISS ingestion
- src/rag/retrieve.py — retrieval utilities (vector search)
- src/rag/llm.py — streaming answers from Ollama
- src/judge — Mistral judge integration
- src/ui — Streamlit UI components
- data/docs — where uploaded documents are stored
- vectorstore/faiss_index — FAISS index files

Configuration
- Adjust defaults in `src/config.py`:
  - `OLLAMA_BASE_URL` — Ollama server URL
  - `DEFAULT_LLM_MODEL`, `DEFAULT_EMBEDDING_MODEL`, `DEFAULT_JUDGE_MODEL`
  - `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`, `DEFAULT_TOP_K`

Troubleshooting
- If uploads are not recognized, make sure file extensions are one of: .pdf, .txt, .md (see `src/config.py`).
- If FAISS index build fails, ensure Ollama embeddings are reachable and working.
- If Streamlit fails to start, confirm dependencies are installed from `requirements.txt`.

License
MIT-style (refer to repository owner)

