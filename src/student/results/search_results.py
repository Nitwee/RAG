from student.reader.reader import Reader, ReaderError
from student.retrieval.bm_25 import BM25Retriever, BM25RetrieverError
from student.retrieval.hybrid import HybridRetriever, HybridRetrieverError
from student.retrieval.vectorizer import Vectorizer, VectorizerError
from student.models import (
    RagDataset,
    AnsweredQuestion,
    UnansweredQuestion,
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
)
from pydantic import ValidationError
from pathlib import Path
from tqdm import tqdm


class SearchResultsError(Exception):
    pass


class SearchResultsFinder:
    def __init__(
        self,
        dataset_path: str,
        save_directory: str = "data/output/search_results",
        k: int = 10,
        method: str = "bm25",
    ) -> None:
        try:
            self.reader = Reader()
            self.retriever = self.load_retriever(method)

            dataset, input_path = self.validate_dataset(dataset_path)
        except (
            ReaderError,
            BM25RetrieverError,
            VectorizerError,
            HybridRetrieverError,
        ) as e:
            raise SearchResultsError(e)
        try:
            res = self.find_search_results(dataset.rag_questions, k)
            results = StudentSearchResults(search_results=res, k=k)
        except ValidationError as e:
            raise SearchResultsError(f"Invalid search result format: {e}")

        self.reader.write_output(
            save_directory,
            input_path.name,
            results.model_dump_json(indent=2),
        )

    def load_retriever(
        self,
        method: str,
    ) -> BM25Retriever | Vectorizer | HybridRetriever:
        method = method.lower()
        if method == "bm25":
            return BM25Retriever.load()
        if method == "vector":
            return Vectorizer.load()
        if method == "hybrid":
            return HybridRetriever.load()
        raise SearchResultsError(f"Unknown retrieval method: {method}")

    def validate_dataset(
            self,
            dataset_path: str,
            ) -> tuple[RagDataset, Path]:
        content, input_path = self.reader.validate_read(dataset_path)
        try:
            dataset = RagDataset.model_validate_json(content)
        except ValidationError as e:
            raise ReaderError(e)
        return (dataset, input_path)

    def find_search_results(
        self,
        dataset: list[AnsweredQuestion | UnansweredQuestion],
        k: int,
    ) -> list[MinimalSearchResults]:
        search_results: list[MinimalSearchResults] = []
        for question_data in tqdm(dataset, desc="Searching questions"):
            question_id = question_data.question_id
            question = question_data.question
            chunks = self.retriever.search(question, k)
            sources = [
                MinimalSource(
                    file_path=str(chunk.filepath),
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in chunks
            ]
            search_results.append(
                MinimalSearchResults(
                    question_id=question_id,
                    question_str=question,
                    retrieved_sources=sources,
                )
            )
        return search_results
