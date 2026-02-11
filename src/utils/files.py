import shutil
from pathlib import Path
from typing import List

import streamlit as st

from src.config import DOCS_DIR, VECTORSTORE_DIR, SUPPORTED_EXT


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def vectorstore_exists() -> bool:
    return VECTORSTORE_DIR.exists() and (VECTORSTORE_DIR / "index.faiss").exists()


def list_docs() -> List[Path]:
    ensure_dir(DOCS_DIR)
    files = []
    for f in DOCS_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
            files.append(f)
    return sorted(files)


def save_uploaded_files(uploaded_files) -> List[Path]:
    ensure_dir(DOCS_DIR)
    saved_paths = []

    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower()

        if suffix not in SUPPORTED_EXT:
            st.warning(f"Skipping unsupported file: {uf.name}")
            continue

        out_path = DOCS_DIR / uf.name
        with open(out_path, "wb") as f:
            f.write(uf.getbuffer())

        saved_paths.append(out_path)

    return saved_paths


def delete_doc(path: Path):
    try:
        path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def clear_docs_folder():
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    ensure_dir(DOCS_DIR)


def clear_vectorstore():
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
