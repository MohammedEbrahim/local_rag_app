import streamlit as st

from src.state import init_session_state
from src.ui.sidebar import render_sidebar
from src.ui.tabs_chat import render_chat_tab
from src.ui.tabs_docs import render_docs_tab
from src.ui.tabs_settings import render_settings_tab
from src.utils.files import ensure_dir
from src.config import DOCS_DIR


def main():
    # Ensure folders
    ensure_dir(DOCS_DIR)

    # Streamlit config
    st.set_page_config(page_title="Local RAG (Ollama + FAISS)", layout="wide")

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
          .stTabs [data-baseweb="tab-list"] button { font-size: 16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🦙 Local RAG Assistant")
    st.caption("Ollama + FAISS — Upload docs, build index, chat with streaming output + LLM Judge.")

    init_session_state()

    # Sidebar
    render_sidebar()

    # Tabs
    tab_chat, tab_docs, tab_settings = st.tabs(["💬 Chat", "📄 Documents", "⚙️ Settings"])

    with tab_chat:
        render_chat_tab()

    with tab_docs:
        render_docs_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
