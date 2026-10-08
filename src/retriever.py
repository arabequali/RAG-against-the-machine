import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.models import MinimalSource


def tokenize(text: str) -> list:
    return re.sub(r"[^\w\s]", " ", text).lower().split()


class Retriever:
    def __init__(self, indexed_chunks: list[dict]) -> None:
        self.indexed_chunks = indexed_chunks
        tokenized_corpus = [c["tokens"] for c in indexed_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)  # rapide : déjà tokenisé

    def search(self, query: str, k: int) -> list[MinimalSource]:
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            MinimalSource(
                file_path=self.indexed_chunks[i]["file_path"],
                first_character_index=self.indexed_chunks[i]["first_character_index"],
                last_character_index=self.indexed_chunks[i]["last_character_index"],
            )
            for i in top_idx
        ]
