from typing import List

from tqdm import tqdm
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.config import DOCS_DIR, VECTORSTORE_DIR, SUPPORTED_EXT, OLLAMA_BASE_URL
from src.utils.files import ensure_dir


def get_embeddings(model_name: str):
    return OllamaEmbeddings(model=model_name, base_url=OLLAMA_BASE_URL)


def load_documents(docs_dir: Path) -> List[Document]:
    docs = []
    if not docs_dir.exists():
        return docs

    for file in docs_dir.rglob("*"):
        if file.suffix.lower() not in SUPPORTED_EXT:
            continue

        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())
        else:
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())

    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def ingest_docs(embedding_model: str, chunk_size: int, chunk_overlap: int):
    ensure_dir(VECTORSTORE_DIR)

    docs = load_documents(DOCS_DIR)
    if not docs:
        return False

    chunks = split_documents(docs, chunk_size, chunk_overlap)

    embeddings = get_embeddings(embedding_model)

    db = FAISS.from_documents(tqdm(chunks), embeddings)
    db.save_local(str(VECTORSTORE_DIR))

    return True
