"""Command-line interface of the RAG pipeline (``python -m src``)."""
import functools
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, ParamSpec

import fire
from tqdm import tqdm

from src.generator import Generator
from src.indexing import indexing, load_chunks, save_chunks
from src.models import (
    MinimalAnswer, MinimalSearchResults, RagDataset,
    StudentSearchResults, StudentSearchResultsAndAnswer
)
from src.recall import recall_at_k
from src.retriever import Retriever

DEFAULT_CHUNK_SIZE = 2000
EVAL_KS = (1, 3, 5, 10)

P = ParamSpec("P")


def _safe(func: Callable[P, None]) -> Callable[P, None]:
    """Turn any exception of a command into a clean error message."""
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - CLI must never crash
            print(f"Error: {exc}", file=sys.stderr)
    return wrapper


def _check_k(k: int) -> int:
    """Validate that ``k`` is a positive integer."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return k


def _check_query(query: Any) -> str:
    """Convert a query to a non-empty string."""
    text = str(query).strip()
    if not text:
        raise ValueError("The query must not be empty")
    return text


def _get_chunks() -> list[dict[str, Any]]:
    """Load the index, building it automatically if it is missing."""
    try:
        return load_chunks()
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("Indexing has not been done yet (or the index is corrupt).")
        print("Automatic indexing will be done with a max chunk size of "
              f"{DEFAULT_CHUNK_SIZE}.")
        time.sleep(1.5)
        chunks = indexing(DEFAULT_CHUNK_SIZE)
        save_chunks(chunks)
        return chunks


def _write_json(directory: str, name: str, content: str) -> Path:
    """Write ``content`` to ``directory/name`` and return the path."""
    out_path = Path(directory) / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


@_safe
def index(max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    """Index ``data/raw`` and save the index under ``data/processed``.

    Args:
        max_chunk_size: Maximum chunk size in characters (1 to 2000).
    """
    chunks = indexing(max_chunk_size)
    save_chunks(chunks)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks "
          "under data/processed/")


@_safe
def search(query: str, k: int = 5) -> None:
    """Print the top-k sources for a single query.

    Args:
        query: The question.
        k: Number of sources to return.
    """
    text, k = _check_query(query), _check_k(k)
    retriever = Retriever(_get_chunks())
    for s in retriever.search(text, k):
        print(f"{s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")


@_safe
def answer(query: str, k: int = 5) -> None:
    """Answer a single query from the retrieved context.

    Args:
        query: The question.
        k: Number of sources given to the model.
    """
    text, k = _check_query(query), _check_k(k)
    retriever = Retriever(_get_chunks())
    sources = retriever.search(text, k)
    print(Generator().generate_answer(text, sources))


@_safe
def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    """Search every question of a dataset and save the results as JSON.

    Args:
        dataset_path: Path to a RagDataset JSON file.
        k: Number of sources per question.
        save_directory: Directory receiving the output file.
    """
    k = _check_k(k)
    dataset = RagDataset.model_validate_json(
        Path(dataset_path).read_text(encoding="utf-8"))
    retriever = Retriever(_get_chunks())

    results = [
        MinimalSearchResults(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=retriever.search(q.question, k),
        )
        for q in tqdm(dataset.rag_questions, desc="Searching dataset")
    ]
    output = StudentSearchResults(search_results=results, k=k)
    out_path = _write_json(
        save_directory, Path(dataset_path).name,
        output.model_dump_json(indent=2))
    print(f"Saved student_search_results to {out_path}")


@_safe
def answer_dataset(
        student_search_results_path: str,
        save_directory: str) -> None:
    """Generate an answer for every question of a search-results file.

    Args:
        student_search_results_path: Output of ``search_dataset``.
        save_directory: Directory receiving the output file.
    """
    data = StudentSearchResults.model_validate_json(
        Path(student_search_results_path).read_text(encoding="utf-8"))
    generator = Generator()

    answers = [
        MinimalAnswer(
            question_id=r.question_id,
            question=r.question,
            retrieved_sources=r.retrieved_sources,
            answer=generator.generate_answer(r.question, r.retrieved_sources),
        )
        for r in tqdm(data.search_results, desc="Generating answers")
    ]
    output = StudentSearchResultsAndAnswer(search_results=answers, k=data.k)
    out_path = _write_json(
        save_directory, Path(student_search_results_path).name,
        output.model_dump_json(indent=2))
    print(f"Saved student_search_results_and_answer to {out_path}")


@_safe
def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Print recall@k of search results against a ground-truth dataset.

    Args:
        student_search_results_path: Output of ``search_dataset``.
        dataset_path: Ground-truth (answered questions) dataset.
    """
    results = StudentSearchResults.model_validate_json(
        Path(student_search_results_path).read_text(encoding="utf-8"))
    truth = RagDataset.model_validate_json(
        Path(dataset_path).read_text(encoding="utf-8"))
    for k in EVAL_KS:
        value, n = recall_at_k(results, truth, k)
        print(f"Recall@{k}: {value:.3f} ({value * 100:.1f}%) on {n} questions")


if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
        "answer": answer,
        "search_dataset": search_dataset,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate,
    })
