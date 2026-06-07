"""Command-line interface for the RAG project."""

import sys
import uuid

import fire

from student.generation.answerer import LLMAnswerer, LLMAnswererError
from student.indexing.index_manager import IndexManager, IndexManagerError
from student.models import MinimalAnswer, MinimalSource
from student.retrieval.bm_25 import BM25Retriever, BM25RetrieverError
from student.results.search_results import (
    SearchResultsFinder,
    SearchResultsError,
)
from student.results.answer_results import (
    AnswerResultsFinder,
    AnswerResultsError,
)
from student.evaluator.eval import Evaluator, EvaluatorError

class AnalyzerCLI:
    """CLI commands expected by the project subject."""

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
    ) -> None:
        """Index the vLLM repository."""
        try:
            self.manager = IndexManager(repo_path, max_chunk_size)
            retriever = BM25Retriever(self.manager.chunks)
            retriever.build()
        except (IndexManagerError, BM25RetrieverError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return
        print(f"Indexing {repo_path} with max_chunk_size={max_chunk_size}")
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        """Search relevant chunks for a single query."""
        try:
            retriever = BM25Retriever.load()
            chunks = retriever.search(query, k)
        except BM25RetrieverError as e:
            print(f"Error: {e}", file=sys.stderr)
            return
        for chunk in chunks:
            print(
                f"{chunk.filepath}:"
                f"{chunk.first_character_index}-"
                f"{chunk.last_character_index}"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search relevant chunks for every question in a dataset."""
        try:
            SearchResultsFinder(
                dataset_path=dataset_path,
                save_directory=save_directory,
                k=k,
            )
        except SearchResultsError as e:
            print(e)
            return

    def answer(
        self,
        question: str,
        k: int = 10,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
        max_context_chars: int = 12000,
    ) -> None:
        """Answer one question using retrieved context."""
        try:
            retriever = BM25Retriever.load()
            chunks = retriever.search(question, k)
            answerer = LLMAnswerer(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_context_chars=max_context_chars,
            )
            answer = answerer.answer(question, chunks)
            sources = [
                MinimalSource(
                    file_path=str(chunk.filepath),
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in chunks
            ]
            result = MinimalAnswer(
                question_id=str(uuid.uuid4()),
                question_str=question,
                retrieved_sources=sources,
                answer=answer,
            )
        except (BM25RetrieverError, LLMAnswererError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return

        print(result.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
        max_context_chars: int = 12000,
    ) -> None:
        """Generate answers from a search results file."""
        try:
            AnswerResultsFinder(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                max_context_chars=max_context_chars,
            )
        except AnswerResultsError as e:
            print(e)
            return

    def evaluate(
        self,
        answers_path: str = "datasets_public/public/AnsweredQuestions/dataset_docs_public.json",
        stud_results_path: str = "data/output/search_results/dataset_docs_public.json",
        k: int = 10,
    ) -> None:
        """Evaluate search results against a ground-truth dataset."""
        try:
            evaluator = Evaluator(
                answers_path,
                stud_results_path,
                k
            )
        except EvaluatorError as e:
            print(e)
            return
        print(evaluator)



def main() -> None:
    """Run the Fire CLI."""
    fire.Fire(AnalyzerCLI)
