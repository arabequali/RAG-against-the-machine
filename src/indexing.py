"""Corpus loading, chunking and tokenisation (index construction)."""
import pickle
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.retriever import tokenize

RAW_DIR = "data/raw"
CHUNKS_PATH = "data/processed/chunks.pkl"
MAX_ALLOWED_CHUNK_SIZE = 2000
OVERLAP_RATIO = 0.05
# data/raw/vllm-0.10.1/<relative path>: the first 3 parts are dropped
RAW_PREFIX_PARTS = 3


def save_chunks(chunks: list[dict[str, Any]], path: str = CHUNKS_PATH) -> None:
    """Persist the indexed chunks with pickle.

    Args:
        chunks: Indexed chunks to save.
        path: Destination file; parent directories are created.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(path: str = CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load indexed chunks previously saved by ``save_chunks``.

    Args:
        path: Source file.

    Returns:
        The list of indexed chunks.

    Raises:
        FileNotFoundError: If the index does not exist.
    """
    with open(path, "rb") as f:
        chunks: list[dict[str, Any]] = pickle.load(f)
    return chunks


def _load_documents(raw_dir: str) -> list[Document]:
    """Load every .py, .md and .txt file found under ``raw_dir``."""
    docs: list[Document] = []
    for pattern in ("**/*.py", "**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            raw_dir,
            glob=pattern,
            recursive=True,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        docs.extend(loader.load())
    return docs


def _split_documents(
        docs: list[Document], max_chunk_size: int) -> list[Document]:
    """Chunk documents with a strategy depending on the file type."""
    overlap = int(max_chunk_size * OVERLAP_RATIO)
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for doc in tqdm(docs, desc="Chunking data"):
        source = str(doc.metadata.get("source", "")).removeprefix("./")
        if source.endswith(".py"):
            chunks.extend(py_splitter.split_documents([doc]))
        elif source.endswith(".md"):
            chunks.extend(md_splitter.split_documents([doc]))
        else:
            chunks.extend(txt_splitter.split_documents([doc]))
    return chunks


def indexing(max_chunk_size: int) -> list[dict[str, Any]]:
    """Build the index of the corpus located in ``data/raw``.

    Args:
        max_chunk_size: Maximum chunk size in characters (1 to 2000).

    Returns:
        Chunks with their tokens, file path and character range.

    Raises:
        ValueError: If ``max_chunk_size`` is out of range.
        FileNotFoundError: If no indexable file is found.
    """
    if not 0 < max_chunk_size <= MAX_ALLOWED_CHUNK_SIZE:
        raise ValueError(
            f"max_chunk_size must be between 1 and {MAX_ALLOWED_CHUNK_SIZE}")
    if not Path(RAW_DIR).is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {RAW_DIR}")

    docs = _load_documents(RAW_DIR)
    if not docs:
        raise FileNotFoundError(f"No .py/.md/.txt file found in {RAW_DIR}")

    indexed_chunks: list[dict[str, Any]] = []
    for chunk in tqdm(_split_documents(docs, max_chunk_size),
                      desc="Tokenizing"):
        start = int(chunk.metadata["start_index"])
        source = str(chunk.metadata.get("source", "")).removeprefix("./")
        rel_path = "/".join(Path(source).parts[RAW_PREFIX_PARTS:])
        indexed_chunks.append({
            "tokens": tokenize(chunk.page_content,
                               is_code=source.endswith(".py"))
            + tokenize(rel_path, is_code=True),
            "file_path": source,
            "first_character_index": start,
            "last_character_index": start + len(chunk.page_content),
        })
    return indexed_chunks
