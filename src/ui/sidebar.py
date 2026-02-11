import streamlit as st

from src.utils.files import (
    list_docs,
    vectorstore_exists,
    save_uploaded_files,
    clear_docs_folder,
    clear_vectorstore,
)
from src.rag.ingest import ingest_docs


def render_sidebar():
    with st.sidebar:
        st.subheader("📌 Status")

        docs = list_docs()
        vs_ready = vectorstore_exists()

        st.write(f"**Docs:** {len(docs)} file(s)")
        st.write(f"**Vectorstore:** {'✅ Ready' if vs_ready else '❌ Not built'}")

        st.divider()

        st.subheader("📤 Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF / TXT / MD",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        colA, colB = st.columns(2)
        with colA:
            if st.button("💾 Save", use_container_width=True, disabled=not uploaded_files):
                saved = save_uploaded_files(uploaded_files)
                st.success(f"Saved {len(saved)} file(s).")
                st.rerun()

        with colB:
            if st.button("🧹 Clear Docs", use_container_width=True, disabled=len(docs) == 0):
                clear_docs_folder()
                st.success("Documents cleared.")
                st.rerun()

        st.divider()

        st.subheader("🧠 Index")
        if st.button("📥 Build / Rebuild FAISS", use_container_width=True, disabled=len(docs) == 0):
            with st.spinner("Building vectorstore..."):
                ok = ingest_docs(
                    embedding_model=st.session_state.embedding_model,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                )
            if ok:
                st.success("Vectorstore built successfully.")
            else:
                st.error("Failed to build vectorstore.")
            st.rerun()

        if st.button("🗑️ Delete Vectorstore", use_container_width=True, disabled=not vs_ready):
            clear_vectorstore()
            st.success("Vectorstore deleted.")
            st.rerun()

        st.divider()

        st.subheader("🧽 Chat")
        if st.button("Clear Chat", use_container_width=True, disabled=len(st.session_state.messages) == 0):
            st.session_state.messages = []
            st.success("Chat cleared.")
            st.rerun()
