import streamlit as st

from src.utils.files import list_docs, delete_doc
from src.rag.ingest import ingest_docs


def render_docs_tab():
    docs = list_docs()

    if not docs:
        st.info("No documents uploaded yet.")
        return

    left, right = st.columns([0.38, 0.62], gap="large")

    with left:
        st.subheader("📚 Files")

        selected = st.selectbox(
            "Select a document",
            options=docs,
            format_func=lambda p: p.name,
        )

        st.caption(f"Path: `{selected}`")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Delete File", use_container_width=True):
                ok = delete_doc(selected)
                if ok:
                    st.success("Deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete file.")

        with c2:
            if st.button("🔁 Rebuild Index", use_container_width=True):
                with st.spinner("Rebuilding FAISS..."):
                    ok = ingest_docs(
                        embedding_model=st.session_state.embedding_model,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                    )
                if ok:
                    st.success("Rebuilt successfully.")
                else:
                    st.error("Rebuild failed.")
                st.rerun()

        st.divider()
        st.write("### All uploaded files")
        for f in docs:
            st.write(f"• {f.name}")

