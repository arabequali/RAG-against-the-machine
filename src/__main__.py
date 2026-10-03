from pathlib import Path
import fire
import time
from tqdm import tqdm

from src.indexing import indexing, save_chunks, load_chunks
from src.retriever import Retriever
from src.generator import Generator
from src.models import (
    RagDataset, MinimalSearchResults, StudentSearchResults,
    MinimalAnswer, StudentSearchResultsAndAnswer,
)


def index(max_chunk_size: int) -> None:
    chunks = indexing(max_chunk_size)
    save_chunks(chunks)
    print("Ingestion complete! Indices saved under data/processed/")


def search(query: str, k: int=5) -> None:
    try:
        chunks = load_chunks()
    except FileNotFoundError:
        print("Indexing has not been done yet.\nAutomatic indexing will be done with a max chunk size of 2000.")
        time.sleep(1.5)
        chunks = indexing(2000)
    retriever = Retriever(chunks)
    sources = retriever.search(query, k)
    for s in sources:
        print(s)


def answer(query: str, k: int=5) -> None:
    chunks = load_chunks()
    retriever = Retriever(chunks)
    sources = retriever.search(query, k)
    generator = Generator()
    text = generator.generate_answer(query, sources)
    print('\033[00m' + text)


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    dataset = RagDataset.model_validate_json(Path(dataset_path).read_text())
    try:
        chunks = load_chunks()
    except FileNotFoundError:
        print("Indexing has not been done yet.\nAutomatic indexing will be done with a max chunk size of 2000.")
        time.sleep(1.5)
        chunks = indexing(2000)
    retriever = Retriever(chunks)

    results = []
    for q in tqdm(dataset.rag_questions, desc="Searching dataset"):
        sources = retriever.search(q.question, k)
        results.append(MinimalSearchResults(
            question_id=q.question_id, question=q.question, retrieved_sources=sources
        ))

    output = StudentSearchResults(search_results=results, k=k)
    out_path = Path(save_directory) / Path(dataset_path).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output.model_dump_json(indent=2))
    print(f"Saved student_search_results to {out_path}")


def answer_dataset(student_search_results_path: str, save_directory: str) -> None:
    data = StudentSearchResults.model_validate_json(Path(student_search_results_path).read_text())
    generator = Generator()

    answers = []
    for r in tqdm(data.search_results, desc="Generating answers"):
        text = generator.generate_answer(r.question, r.retrieved_sources)
        answers.append(MinimalAnswer(
            question_id=r.question_id, question=r.question,
            retrieved_sources=r.retrieved_sources, answer=text,
        ))

    output = StudentSearchResultsAndAnswer(search_results=answers, k=data.k)
    out_path = Path(save_directory) / Path(student_search_results_path).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output.model_dump_json(indent=2))
    print(f"Saved student_search_results_and_answer to {out_path}")


if __name__ == '__main__':
    fire.Fire()
