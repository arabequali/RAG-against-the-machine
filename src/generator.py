from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"


def read_source_text(source: MinimalSource) -> str:
    text = Path(source.file_path).read_text(encoding="utf-8", errors="ignore")
    return text[source.first_character_index:source.last_character_index]


class Generator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_answer(self, question: str, sources: list[MinimalSource]) -> str:
        context = "\n\n".join(
            f"[Source: {s.file_path}]\n{read_source_text(s)}" for s in sources
        )
        system = ("Answer only using the sources below. "
                   "If the answer isn't there, say so.\n\n" + context)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        new_tokens = output_ids[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
