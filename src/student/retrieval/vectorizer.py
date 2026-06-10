import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from student.indexing.chunk import Chunk


class VectorizerError(Exception):
    pass


class Vectorizer:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings: NDArray[np.float32] | None = None
        self.embeddings_path = Path("data/processed/embeddings/embeddings.npy")
        self.chunks_path = Path("data/processed/chunks/chunks.json")

    def build(self) -> None:
        documents = [
            f"{chunk.filepath}\n{chunk.content}"
            for chunk in self.chunks
        ]

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.save()

    def save(self) -> None:
        if self.embeddings is None:
            raise VectorizerError("No embeddings to save")

        try:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.embeddings_path, self.embeddings)
        except OSError as e:
            raise VectorizerError(f"Cannot save embeddings: {e}") from e

    def search(self, query: str, k: int) -> list[Chunk]:
        if k <= 0 or not self.chunks:
            return []
        if self.embeddings is None:
            raise VectorizerError("Embeddings are not loaded")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        query_vector = np.asarray(query_embedding, dtype=np.float32)

        scores = self.embeddings @ query_vector
        sorted_chunks_indexes = np.argsort(scores)
        top_chunks_indexes = sorted_chunks_indexes[-k:][::-1]

        return [self.chunks[int(chunk_id)] for chunk_id in top_chunks_indexes]

    @classmethod
    def load(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "Vectorizer":
        chunks_path = Path("data/processed/chunks/chunks.json")
        embeddings_path = Path("data/processed/embeddings/embeddings.npy")

        if not chunks_path.exists():
            raise VectorizerError(f"Chunks file not found: {chunks_path}")
        if not embeddings_path.exists():
            raise VectorizerError(
                f"Embeddings file not found: {embeddings_path}"
            )

        try:
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = [Chunk(**item) for item in chunks_data]
            embeddings = np.load(embeddings_path)
        except json.JSONDecodeError as e:
            raise VectorizerError(f"Invalid chunks JSON: {chunks_path}") from e
        except (OSError, ValueError) as e:
            raise VectorizerError(f"Cannot load embeddings data: {e}") from e
        except (TypeError, ValidationError) as e:
            raise VectorizerError(f"Invalid chunk data: {chunks_path}") from e

        retriever = cls(chunks, model_name=model_name)
        retriever.embeddings = np.asarray(embeddings, dtype=np.float32)
        return retriever
