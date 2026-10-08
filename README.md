*This project has been created as part of the 42 curriculum by avally.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about a
codebase (the vLLM 0.10.1 repository). Given a question, the system:

1. retrieves the most relevant source locations (`file_path` + character
   range) from a pre-built lexical index,
2. feeds those snippets to a small local model (`Qwen/Qwen3-0.6B`),
3. returns an answer grounded in the retrieved context.

Retrieval quality is measured with **recall@k** (a source is found when a
result is in the same file with an IoU > 0.05 on the character range).

## Instructions

Requirements: Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
make run            # index the corpus (max_chunk_size 2000)
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make clean          # remove caches
make debug          # run under pdb
```

The corpus must be placed in `data/raw/` (e.g. `data/raw/vllm-0.10.1/`).
The index is written to `data/processed/chunks.pkl`.

### CLI

All commands are run with `uv run python -m src <command> [options]`.

| Command          | Options                                                        | Purpose                                   |
|------------------|----------------------------------------------------------------|-------------------------------------------|
| `index`          | `--max_chunk_size` (1–2000, default 2000)                      | Build the index                           |
| `search`         | `<query> --k`                                                  | Top-k sources for one query               |
| `answer`         | `<query> --k`                                                  | Answer one query                          |
| `search_dataset` | `--dataset_path --k --save_directory`                          | Batch search, writes `StudentSearchResults` |
| `answer_dataset` | `--student_search_results_path --save_directory`               | Batch answers, writes `StudentSearchResultsAndAnswer` |
| `evaluate`       | `--student_search_results_path --dataset_path`                 | Recall@1/3/5/10 against ground truth      |

Degenerate inputs (empty query, `k=0`, missing file, malformed JSON, missing
index) print an error message instead of a traceback. If the index is missing,
`search`, `answer` and `search_dataset` build it automatically.

## System architecture

```
data/raw ──► indexing.py ──► data/processed/chunks.pkl
                                   │
question ──► retriever.py ◄────────┘      (2 BM25 indices + weighted RRF)
                 │
                 ▼
            top-k MinimalSource ──► generator.py (Qwen3-0.6B) ──► answer
                 │
                 └──► recall.py (evaluate)
```

- `src/models.py`: pydantic data models exchanged between the stages.
- `src/indexing.py`: loading, chunking and tokenisation of the corpus.
- `src/retriever.py`: tokenizer, BM25 indices, rank fusion.
- `src/generator.py`: prompt construction and generation with Qwen3-0.6B.
- `src/recall.py`: recall@k with the IoU rule of the subject.
- `src/__main__.py`: Python Fire CLI.

## Chunking strategy

Two distinct strategies from `langchain-text-splitters`
(`RecursiveCharacterTextSplitter`):

- **Python files**: `Language.PYTHON` separators, which prefer to cut at
  class / function boundaries before falling back to blank lines and lines.
- **Markdown / text files**: `Language.MARKDOWN` for `.md` (cuts on headers
  first), a plain recursive splitter for `.txt`.

Chunks are at most `max_chunk_size` characters (default and hard maximum:
2000, the limit enforced by the moulinette) with a 5 % overlap. Each chunk
stores its exact `first_character_index` / `last_character_index` in the
original file, so results can be compared with the reference spans.

## Retrieval method

- **Tokenisation**: lowercase, stopword removal, Snowball English stemming.
  For code, `snake_case` and `CamelCase` identifiers are kept whole (unstemmed,
  so verbatim identifier queries match) **and** split into sub-words. The
  file path (without `data/raw/vllm-0.10.1/`) is added to each chunk's tokens,
  which helps when a question mentions a file or module name.
- **Two BM25 indices** (`rank_bm25.BM25Okapi`, `k1=1.5`, `b=0.75`): one for
  `.py` chunks (with de-duplicated tokens, i.e. binary term frequency, to avoid
  long repetitive code dominating) and one for docs (`.md`, `.txt`).
- **Query routing**: a heuristic (`looks_like_code`) detects identifiers or
  words such as "function", "class", "method" in the question and slightly
  favours the code index (weights 1.0 / 0.95), otherwise the docs index.
- **Fusion**: Weighted Reciprocal Rank Fusion
  (`score = Σ weight / (60 + rank)`) over the top 50 of each index.

## Performance analysis

> Fill in with the numbers obtained on your machine
> (`uv run python -m src evaluate ...` or the moulinette).

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Target @5 |
|---------|----------|----------|----------|-----------|-----------|
| Docs    | _TBD_    | _TBD_    | _TBD_    | _TBD_     | ≥ 80 %    |
| Code    | _TBD_    | _TBD_    | _TBD_    | _TBD_     | ≥ 50 %    |

| Constraint            | Target      | Measured |
|-----------------------|-------------|----------|
| Indexing time         | ≤ 5 min     | _TBD_    |
| 200 questions search  | ≤ 90 s      | _TBD_    |

Effect of `max_chunk_size` on recall: _TBD (e.g. 500 / 1000 / 2000)_.

## Design decisions

- **Lexical BM25 only**, with careful tokenisation: fast to index, fast to
  query (well under the throughput limit) and strong on identifier-heavy
  questions.
- **Separate indices for code and docs** so that IDF statistics of prose are
  not polluted by code, and fused with RRF, which needs no score
  normalisation.
- **Index stored as a pickle of plain dicts** (tokens + location): simple,
  quick to load, and the retriever is rebuilt in memory at start-up.
- **Chunk size capped at 2000** by the CLI to guarantee valid output.
- **Qwen3-0.6B with `enable_thinking=False`** and a strict system prompt
  ("answer only using the sources, otherwise say so") to limit hallucinations
  and keep generation short.
- **Pydantic** models for every exchanged structure; all commands share a
  single error-handling wrapper so the CLI never crashes with a traceback.

## Challenges faced

- Questions paraphrase ideas while code uses identifiers: solved by indexing
  both the whole identifier and its sub-words, plus the file path.
- Keeping character offsets accurate after splitting: `add_start_index=True`
  gives the start of each chunk in the original document.
- A 0.6B model reasons poorly: the quality of the retrieved context and the
  prompt matter more than the model, so effort went into retrieval.
- _Add your own difficulties and solutions here._

## Example usage

```bash
uv run python -m src index --max_chunk_size 2000

uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src answer "What does max_num_seqs control?" --k 5

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## Resources

- [vLLM documentation](https://docs.vllm.ai/)
- [BM25 (Robertson & Zaragoza, *The Probabilistic Relevance Framework*)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [rank_bm25](https://github.com/dorianbrown/rank_bm25),
  [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/),
  [Snowball stemmer](https://snowballstem.org/),
  [Qwen3 on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B),
  [Python Fire](https://github.com/google/python-fire),
  [pydantic](https://docs.pydantic.dev/)

### Use of AI

- cleaning the code to flake8/mypy
- writing docstrings
- structuring the README
Every generated part was reviewed, tested and is understood by the authors.
