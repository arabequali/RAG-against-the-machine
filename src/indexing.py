from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.retriever import tokenize
import pickle
from pathlib import Path


def save_chunks(chunks: list, path: str = "data/processed/chunks.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(path: str = "data/processed/chunks.pkl") -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def indexing(max_chunk_size: int) -> None:
    dir_py = DirectoryLoader(
        "data/raw",
        glob="**/*.py",
        recursive=True,
        loader_cls=TextLoader
    )
    dir_md = DirectoryLoader(
        "data/raw",
        glob="**/*.md",
        recursive=True,
        loader_cls=TextLoader
    )
    dir_txt = DirectoryLoader(
        "data/raw",
        glob="**/*.txt",
        recursive=True,
        loader_cls=TextLoader
    )

    docs = dir_py.load() + dir_md.load() + dir_txt.load()

    py_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=50,
        add_start_index=True
    )
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=max_chunk_size,
        chunk_overlap=50,
        add_start_index=True
    )
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=50, add_start_index=True
    )

    all_chunks = []
    for doc in tqdm(docs, desc="Chunking data"):
        source = doc.metadata.get("source", "").removeprefix("./")
        if source.endswith(".py"):
            all_chunks.extend(py_splitter.split_documents([doc]))
        elif source.endswith(".md"):
            all_chunks.extend(md_splitter.split_documents([doc]))
        else:
            all_chunks.extend(txt_splitter.split_documents([doc]))

    indexed_chunks = []
    for chunk in tqdm(all_chunks, desc="Tokenizing"):
        start = chunk.metadata["start_index"]
        indexed_chunks.append({
            "tokens": tokenize(chunk.page_content),
            "file_path": chunk.metadata.get("source", "").removeprefix("./"),
            "first_character_index": start,
            "last_character_index": start + len(chunk.page_content),
        })

    save_chunks(indexed_chunks)
    return (indexed_chunks)
