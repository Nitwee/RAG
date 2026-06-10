from student.indexing.chunk import Chunk
from student.retrieval.bm_25 import BM25Retriever, BM25RetrieverError
from student.retrieval.vectorizer import Vectorizer, VectorizerError


class HybridRetrieverError(Exception):
    pass


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        vectorizer: Vectorizer,
    ) -> None:
        self.bm25 = bm25
        self.vectorizer = vectorizer

    def search(self, query: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []

        bm25_candidates = self.bm25.search(query, k)
        vector_candidates = self.vectorizer.search(query, k)
        pattern = self._build_pattern(k)

        chunks: list[Chunk] = []
        for source in pattern:
            if source == "bm25":
                self._add_next_unique(chunks, bm25_candidates)
            else:
                self._add_next_unique(chunks, vector_candidates)

        while len(chunks) < k:
            added = self._add_next_unique(chunks, bm25_candidates)
            if not added:
                added = self._add_next_unique(chunks, vector_candidates)
            if not added:
                break

        return chunks[:k]

    @classmethod
    def load(cls) -> "HybridRetriever":
        try:
            return cls(
                bm25=BM25Retriever.load(),
                vectorizer=Vectorizer.load(),
            )
        except (BM25RetrieverError, VectorizerError) as e:
            raise HybridRetrieverError(f"Cannot load hybrid retriever: {e}")

    def _build_pattern(self, k: int) -> list[str]:
        pattern = ["bm25", "bm25", "bm25", "bm25", "vector"]
        if k > 5:
            pattern += ["bm25", "bm25", "bm25", "vector", "vector"]
        return pattern[:k]

    def _add_next_unique(
        self,
        chunks: list[Chunk],
        candidates: list[Chunk],
    ) -> bool:
        seen = {
            (
                chunk.filepath,
                chunk.first_character_index,
                chunk.last_character_index,
            )
            for chunk in chunks
        }

        for chunk in candidates:
            key = (
                chunk.filepath,
                chunk.first_character_index,
                chunk.last_character_index,
            )
            if key in seen:
                continue
            chunks.append(chunk)
            return True
        return False
