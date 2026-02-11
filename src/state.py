import streamlit as st

from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_JUDGE_MODEL,
)


def init_session_state():
    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Models
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = DEFAULT_LLM_MODEL  # llama3

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL

    # Chunking
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE

    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP

    # Retrieval
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K

    # Judge (local Ollama)
    if "use_judge" not in st.session_state:
        st.session_state.use_judge = True

    if "judge_model" not in st.session_state:
        st.session_state.judge_model = DEFAULT_JUDGE_MODEL  # mistral
