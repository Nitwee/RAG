from typing import Any

from student.indexing.chunk import Chunk


class LLMAnswererError(Exception):
    pass


class LLMAnswerer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
        max_context_chars: int = 12000,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.device = "cpu"

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        self.load_model()
        prompt = self.build_prompt(question, chunks)
        return self.generate(prompt)

    def load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise LLMAnswererError(
                "Missing LLM dependencies. Install PyTorch before using "
                "answer."
            ) from e

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)
            model: Any = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
        except OSError as e:
            raise LLMAnswererError(
                f"Cannot load model {self.model_name}: {e}"
            ) from e
        except Exception as e:
            raise LLMAnswererError(
                f"Cannot initialize model {self.model_name}: {e}"
            ) from e

    def build_prompt(self, question: str, chunks: list[Chunk]) -> str:
        context = self.build_context(chunks)
        return (
            "Answer the question using only the sources below.\n"
            "If the sources do not contain the answer, say that the indexed "
            "sources do not provide enough information.\n"
            "Cite sources with labels like [S1] or [S2].\n\n"
            f"Sources:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    def build_context(self, chunks: list[Chunk]) -> str:
        parts: list[str] = []
        used_chars = 0

        for index, chunk in enumerate(chunks, start=1):
            header = (
                f"[S{index}] {chunk.filepath}:"
                f"{chunk.first_character_index}-"
                f"{chunk.last_character_index}\n"
            )
            chunk_size = len(header) + len(chunk.content)

            if used_chars + chunk_size > self.max_context_chars:
                break

            parts.append(f"{header}{chunk.content}")
            used_chars += chunk_size

        return "\n\n".join(parts)

    def generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise LLMAnswererError("Model is not loaded")

        try:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0][prompt_length:]
            answer = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
        except Exception as e:
            raise LLMAnswererError(f"Cannot generate answer: {e}") from e

        return str(answer).strip()
