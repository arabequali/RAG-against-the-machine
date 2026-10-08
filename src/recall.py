"""Recall@k computation matching the subject's definition."""
from src.models import (
    AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults
)

IOU_THRESHOLD = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Compute the intersection over union of two character ranges.

    Args:
        a: First source.
        b: Second source.

    Returns:
        The IoU in [0, 1], or 0.0 if the union is empty.
    """
    inter = max(0, min(a.last_character_index, b.last_character_index)
                - max(a.first_character_index, b.first_character_index))
    union = ((a.last_character_index - a.first_character_index)
             + (b.last_character_index - b.first_character_index)
             - inter)
    return inter / union if union > 0 else 0.0


def is_found(truth: MinimalSource, retrieved: list[MinimalSource]) -> bool:
    """Tell whether a true source is covered by a retrieved result.

    Args:
        truth: Ground-truth source.
        retrieved: Retrieved sources.

    Returns:
        True if a result is in the same file with IoU above the threshold.
    """
    return any(
        r.file_path == truth.file_path and iou(truth, r) > IOU_THRESHOLD
        for r in retrieved
    )


def recall_at_k(
        results: StudentSearchResults,
        dataset: RagDataset,
        k: int) -> tuple[float, int]:
    """Compute the mean recall@k over the answered questions.

    Args:
        results: Search results to score.
        dataset: Ground-truth dataset.
        k: Number of top results considered per question.

    Returns:
        A tuple (mean recall@k, number of evaluated questions).
    """
    truth = {
        q.question_id: q for q in dataset.rag_questions
        if isinstance(q, AnsweredQuestion)
    }
    total = 0.0
    count = 0
    for res in results.search_results:
        question = truth.get(res.question_id)
        if question is None or not question.sources:
            continue
        retrieved = res.retrieved_sources[:k]
        found = sum(1 for s in question.sources if is_found(s, retrieved))
        total += found / len(question.sources)
        count += 1
    return (total / count if count else 0.0), count
