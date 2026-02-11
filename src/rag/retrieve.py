from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_DIR
from src.utils.files import vectorstore_exists
from src.rag.ingest import get_embeddings


def load_vectorstore(embedding_model: str):
    if not vectorstore_exists():
        return None

    embeddings = get_embeddings(embedding_model)

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def format_context(docs: List[Document]) -> str:
    out = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "N/A")
        out.append(f"[{i}] SOURCE={source} PAGE={page}\n{d.page_content}")
    return "\n\n".join(out)


def rag_retrieve(question: str, embedding_model: str, top_k: int) -> dict:
    db = load_vectorstore(embedding_model)
    if db is None:
        return {"context": "", "sources": []}

    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)

    context = format_context(docs)

    sources = []
    for d in docs:
        sources.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", None),
                "snippet": d.page_content[:350],
            }
        )

    return {"context": context, "sources": sources}
