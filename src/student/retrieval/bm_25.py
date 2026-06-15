"""BM25 retrieval backend backed by bm25s."""

import json
from pathlib import Path

import bm25s
from pydantic import ValidationError

from student.indexing.chunk import Chunk


class BM25RetrieverError(Exception):
    """Raised when BM25 indexing, loading, or searching fails."""

    pass


class BM25Retriever:
    """Build, persist, load, and query a BM25 index."""

    def __init__(self, chunks: list[Chunk]) -> None:
        """Create a retriever for a list of chunks."""

        self.chunks = chunks
        self.model = bm25s.BM25(corpus=self._corpus_ids())
        self.index_path = Path("data/processed/bm25_index")
        self.chunks_path = Path("data/processed/chunks/chunks.json")

    def build(self) -> None:
        """Tokenize chunks, build the BM25 index, and save it."""

        documents = [
            f"{chunk.filepath}\n{chunk.content}"
            for chunk in self.chunks
        ]
        tokens = bm25s.tokenize(documents, show_progress=True)
        self.model.index(tokens, show_progress=True)
        self.save()

    def save(self) -> None:
        """Persist the BM25 index and chunk metadata to disk."""

        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.index_path, corpus=self._corpus_ids())
            data = [chunk.model_dump(mode="json") for chunk in self.chunks]
            self.chunks_path.write_text(
                json.dumps(data),
                encoding="utf-8",
            )
        except PermissionError as e:
            raise BM25RetrieverError(
                "Permission denied while saving BM25"
            ) from e
        except OSError as e:
            raise BM25RetrieverError(f"Cannot save BM25 data: {e}") from e

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return the top-k chunks for a query."""

        if k <= 0 or not self.chunks:
            return []

        query_tokens = bm25s.tokenize([query], show_progress=False)
        chunk_ids = self.model.retrieve(
            query_tokens,
            corpus=self._corpus_ids(),
            k=min(k, len(self.chunks)),
            return_as="documents",
            show_progress=False,
        )[0]

        return [self.chunks[int(chunk_id)] for chunk_id in chunk_ids]

    @classmethod
    def load(cls) -> "BM25Retriever":
        """Load a persisted BM25 retriever from processed data."""

        index_path = Path("data/processed/bm25_index")
        chunks_path = Path("data/processed/chunks/chunks.json")

        if not chunks_path.exists():
            raise BM25RetrieverError(f"Chunks file not found: {chunks_path}")
        if not index_path.exists():
            raise BM25RetrieverError(f"BM25 index not found: {index_path}")
        if not chunks_path.is_file():
            raise BM25RetrieverError(
                f"Chunks path is not a file: {chunks_path}"
            )
        if not index_path.is_dir():
            raise BM25RetrieverError(
                f"BM25 index path is not a directory: {index_path}"
            )

        try:
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        except PermissionError as e:
            raise BM25RetrieverError(
                f"Permission denied reading {chunks_path}"
            ) from e
        except json.JSONDecodeError as e:
            raise BM25RetrieverError(
                f"Invalid chunks JSON: {chunks_path}"
            ) from e
        except OSError as e:
            raise BM25RetrieverError(f"Cannot read chunks file: {e}") from e

        try:
            chunks = [Chunk(**item) for item in chunks_data]
        except (TypeError, ValidationError) as e:
            raise BM25RetrieverError(
                f"Invalid chunk data: {chunks_path}"
            ) from e

        retriever = cls(chunks)
        try:
            retriever.model = bm25s.BM25.load(
                index_path,
                load_corpus=False,
            )
        except PermissionError as e:
            raise BM25RetrieverError(
                f"Permission denied reading {index_path}"
            ) from e
        except OSError as e:
            raise BM25RetrieverError(f"Cannot load BM25 index: {e}") from e
        except Exception as e:
            raise BM25RetrieverError(
                f"Invalid BM25 index: {index_path}"
            ) from e

        return retriever

    def _corpus_ids(self) -> list[str]:
        """Return stable string ids matching chunk positions."""

        return [str(i) for i in range(len(self.chunks))]
