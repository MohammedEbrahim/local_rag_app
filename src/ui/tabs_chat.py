import streamlit as st

from src.utils.files import vectorstore_exists
from src.rag.retrieve import rag_retrieve
from src.rag.llm import stream_ollama_answer
from src.judge.mistral import mistral_judge


def render_chat_tab():
    if not vectorstore_exists():
        st.warning("Upload documents and build the FAISS index before chatting.")

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask something from your documents...",
        disabled=not vectorstore_exists(),
    )

    if not question:
        return

    # Store user question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Retrieve
        with st.spinner("Retrieving..."):
            retrieved = rag_retrieve(
                question=question,
                embedding_model=st.session_state.embedding_model,
                top_k=st.session_state.top_k,
            )

        if not retrieved["context"].strip():
            msg = "No context retrieved. Rebuild your index."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        # Stream answer from llama3
        placeholder = st.empty()
        full_answer = ""

        try:
            for token in stream_ollama_answer(
                    llm_model=st.session_state.llm_model,  # llama3
                    question=question,
                    context=retrieved["context"],
            ):
                full_answer += token
                placeholder.markdown(full_answer)

        except Exception as e:
            full_answer = f"❌ Ollama error: {e}"
            placeholder.markdown(full_answer)

        # Sources
        if retrieved["sources"]:
            with st.expander("📌 Sources"):
                for i, s in enumerate(retrieved["sources"], start=1):
                    st.markdown(
                        f"**[{i}]** `{s['source']}` (page: {s['page']})\n\n"
                        f"> {s['snippet']}..."
                    )

        # Mistral Judge
        if st.session_state.use_judge:
            with st.expander(f"⚖️ {st.session_state.judge_model} Judge (Evaluation)"):
                with st.spinner(f"Judging answer with {st.session_state.judge_model}..."):
                    judge = mistral_judge(
                        question=question,
                        context=retrieved["context"],
                        answer=full_answer,
                        judge_model=st.session_state.judge_model,
                    )
                st.json(judge)

        # Store assistant answer
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
