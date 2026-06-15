"""Command-line interface for the RAG project."""

import sys
import uuid

import fire

from student.generation.answerer import LLMAnswerer, LLMAnswererError
from student.indexing.index_manager import IndexManager, IndexManagerError
from student.models import MinimalAnswer, MinimalSource
from student.retrieval.bm_25 import BM25Retriever, BM25RetrieverError
from student.retrieval.hybrid import HybridRetriever, HybridRetrieverError
from student.retrieval.vectorizer import Vectorizer, VectorizerError
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
        method: str = "bm25",
    ) -> None:
        """Index the vLLM repository."""
        try:
            self.manager = IndexManager(repo_path, max_chunk_size)
            retriever = BM25Retriever(self.manager.chunks)
            retriever.build()
            if method in ("hybrid", "vector"):
                vectorizer = Vectorizer(retriever.chunks)
                vectorizer.build()
        except (IndexManagerError, BM25RetrieverError, VectorizerError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return
        print(f"Indexing {repo_path} with max_chunk_size={max_chunk_size}")
        print("Ingestion complete! Indices saved under data/processed/")

    def search(
        self,
        query: str,
        k: int = 10,
        method: str = "bm25",
    ) -> None:
        """Search relevant chunks for a single query."""
        try:
            retriever = self.load_retriever(method)
            chunks = retriever.search(query, k)

        except (
            BM25RetrieverError,
            VectorizerError,
            HybridRetrieverError,
        ) as e:
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
        dataset_path: str = (
            "datasets_public/public/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        k: int = 10,
        save_directory: str = "data/output/search_results",
        method: str = "bm25",
    ) -> None:
        """Search relevant chunks for every question in a dataset."""
        try:
            SearchResultsFinder(
                dataset_path=dataset_path,
                save_directory=save_directory,
                k=k,
                method=method,
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
        method: str = "bm25",
    ) -> None:
        """Answer one question using retrieved context."""
        try:
            retriever = self.load_retriever(method)
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
        except (
            BM25RetrieverError,
            VectorizerError,
            HybridRetrieverError,
            LLMAnswererError,
        ) as e:
            print(f"Error: {e}", file=sys.stderr)
            return

        print(result.model_dump_json(indent=2))

    def load_retriever(
        self,
        method: str,
    ) -> BM25Retriever | Vectorizer | HybridRetriever:
        """Load a retriever by name."""
        method = method.lower()
        if method == "bm25":
            return BM25Retriever.load()
        if method == "vector":
            return Vectorizer.load()
        if method == "hybrid":
            return HybridRetriever.load()
        raise HybridRetrieverError(f"Unknown retrieval method: {method}")

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
        answers_path: str = (
            "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
        ),
        stud_results_path: str = (
            "data/output/search_results/dataset_docs_public.json"
        ),
        k: int = 10,
    ) -> None:
        """Evaluate search results against a ground-truth dataset."""
        try:
            evaluator = Evaluator(
                answers_path,
                stud_results_path,
            )
            score = evaluator.analyze(k)
            print(f"Recall@{k}: {score:.3f}")

        except EvaluatorError as e:
            print(e)
            return


def main() -> None:
    """Run the Fire CLI."""
    fire.Fire(AnalyzerCLI)
