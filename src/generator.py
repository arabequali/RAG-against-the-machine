"""Answer generation with a small local language model."""
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 512


def read_source_text(source: MinimalSource) -> str:
    """Read the text covered by a source.

    Args:
        source: File path and character range.

    Returns:
        The corresponding text, or an empty string if the file is unreadable.
    """
    try:
        text = Path(source.file_path).read_text(
            encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return text[source.first_character_index:source.last_character_index]


class Generator:
    """Grounded answer generator based on a causal language model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and the model.

        Args:
            model_name: Hugging Face identifier of the model.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=(torch.float16 if self.device.type == "cuda"
                   else torch.float32),
            ).to(self.device)  # type: ignore

    def generate_answer(
            self,
            question: str,
            sources: list[MinimalSource]) -> str:
        """Answer a question using only the given sources.

        Args:
            question: The user question.
            sources: Retrieved sources used as context.

        Returns:
            The generated answer.
        """
        context = "\n\n".join(
            f"[Source: {s.file_path}]\n{read_source_text(s)}"
            for s in sources
        )
        system = ("Answer only using the sources below. "
                  "If the answer isn't there, say so.\n\n" + context)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = cast(
                torch.Tensor,
                self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS),
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return str(self.tokenizer.decode(
            new_tokens, skip_special_tokens=True)).strip()
