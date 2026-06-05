from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from student.generation.answerer import LLMAnswerer, LLMAnswererError
from student.indexing.chunk import Chunk
from student.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from student.retrieval.reader import Reader, ReaderError


class AnswerResultsError(Exception):
    pass


class AnswerResultsFinder:
    def __init__(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
        max_context_chars: int = 12000,
    ) -> None:
        self.reader = Reader()
        search_results, input_path = self.load_search_results(
            student_search_results_path
        )
        self.answerer = LLMAnswerer(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            max_context_chars=max_context_chars,
        )

        try:
            answers = self.find_answer_results(search_results.search_results)
            results = StudentSearchResultsAndAnswer(
                search_results=answers,
                k=search_results.k,
            )
        except (ValidationError, LLMAnswererError, ReaderError) as e:
            raise AnswerResultsError(
                f"Invalid answer result format: {e}"
            ) from e

        try:
            self.reader.write_output(
                save_directory,
                input_path.name,
                results.model_dump_json(indent=2),
            )
        except ReaderError as e:
            raise AnswerResultsError(e) from e

    def load_search_results(
        self,
        search_results_path: str,
    ) -> tuple[StudentSearchResults, Path]:
        try:
            content, input_path = self.reader.validate_read(
                search_results_path
            )
        except ReaderError as e:
            raise AnswerResultsError(e) from e

        try:
            return (
                StudentSearchResults.model_validate_json(content),
                input_path,
            )
        except ValidationError as e:
            raise AnswerResultsError(
                f"Invalid search results JSON: {input_path}"
            ) from e

    def find_answer_results(
        self,
        search_results: Sequence[MinimalSearchResults],
    ) -> list[MinimalAnswer]:
        answer_results: list[MinimalAnswer] = []

        for search_result in search_results:
            chunks = self.sources_to_chunks(search_result.retrieved_sources)
            answer = self.answerer.answer(search_result.question_str, chunks)
            answer_results.append(
                MinimalAnswer(
                    question_id=search_result.question_id,
                    question_str=search_result.question_str,
                    retrieved_sources=search_result.retrieved_sources,
                    answer=answer,
                )
            )

        return answer_results

    def sources_to_chunks(self, sources: list[MinimalSource]) -> list[Chunk]:
        return [self.source_to_chunk(source) for source in sources]

    def source_to_chunk(self, source: MinimalSource) -> Chunk:
        filepath = self.resolve_source_path(source.file_path)
        content = self.reader.read_file(filepath)
        start = source.first_character_index
        end = source.last_character_index

        if start < 0 or end < start or end > len(content):
            raise AnswerResultsError(
                f"Invalid source offsets for {source.file_path}: "
                f"{start}-{end}"
            )

        try:
            return Chunk(
                filepath=filepath,
                content=content[start:end],
                first_character_index=start,
                last_character_index=end,
            )
        except ValidationError as e:
            raise AnswerResultsError(
                f"Invalid chunk rebuilt from {source.file_path}"
            ) from e

    def resolve_source_path(self, file_path: str) -> Path:
        filepath = Path(file_path)
        if not filepath.exists():
            raise AnswerResultsError(f"Source file not found: {file_path}")
        return filepath
