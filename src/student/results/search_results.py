from student.retrieval.reader import Reader, ReaderError
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


class SearchResultsError(Exception):
    pass


class SearchResultsFinder:
    def __init__(
        self,
        dataset_path: str,
        save_directory: str = "data/output/search_results",
        k: int = 10,
    ) -> None:
        try:
            self.reader = Reader()
            self.reader.load_bm25()
            dataset, input_path = self.validate_dataset(dataset_path)
        except ReaderError as e:
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
        for question_data in dataset:
            question_id = question_data.question_id
            question = question_data.question
            chunks = self.reader.retriever.search(question, k)
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
