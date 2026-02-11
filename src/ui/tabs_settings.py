import streamlit as st


def render_settings_tab():
    st.subheader("⚙️ RAG Settings")

    # Models
    st.write("### Models")

    st.session_state.llm_model = st.text_input(
        "LLM model (Ollama) — Answer Generator",
        value=st.session_state.llm_model,
        help="Example: llama3:latest, llama3.2:1b, mistral:latest",
    )

    st.session_state.embedding_model = st.text_input(
        "Embedding model (Ollama)",
        value=st.session_state.embedding_model,
        help="Recommended: nomic-embed-text",
    )

    # Chunking
    st.divider()
    st.write("### Chunking")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.chunk_size = st.number_input(
            "Chunk size",
            min_value=200,
            max_value=2000,
            value=int(st.session_state.chunk_size),
            step=50,
        )
    with col2:
        st.session_state.chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=int(st.session_state.chunk_overlap),
            step=10,
        )

    # Retrieval
    st.divider()
    st.write("### Retrieval")

    st.session_state.top_k = st.slider(
        "Top-K chunks retrieved",
        min_value=1,
        max_value=12,
        value=int(st.session_state.top_k),
    )

    st.info("After changing embedding model or chunk settings, rebuild the FAISS index.")

    # Judge
    st.divider()
    st.subheader("⚖️ Judge Settings (Local Ollama)")

    st.session_state.use_judge = st.checkbox(
        "Enable Mistral Judge",
        value=st.session_state.use_judge,
    )

    st.session_state.judge_model = st.text_input(
        "Judge model (Ollama)",
        value=st.session_state.judge_model,
        help="Recommended: mistral:latest",
    )
