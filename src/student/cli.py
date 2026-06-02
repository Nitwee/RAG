"""Command-line interface for the RAG project."""

from pathlib import Path
from student.indexing.index_manager import IndexManager

import fire


class StudentCLI:
    """CLI commands expected by the project subject."""

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
    ) -> None:
        """Index the vLLM repository."""
        self.chunks = IndexManager("vllm-0.10.1", max_chunk_size)
        print(f"Indexing {repo_path} with max_chunk_size={max_chunk_size}")

    def search(self, query: str, k: int = 10) -> None:
        """Search relevant chunks for a single query."""
        print(f"Searching query={query!r} with k={k}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search relevant chunks for every question in a dataset."""
        output_dir = Path(save_directory)
        print(f"Searching dataset={dataset_path} with k={k}")
        print(f"Results will be saved under {output_dir}")

    def answer(self, question: str, k: int = 10) -> None:
        """Answer one question using retrieved context."""
        print(f"Answering question={question!r} with k={k}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers from a search results file."""
        output_dir = Path(save_directory)
        print(f"Answering search results={student_search_results_path}")
        print(f"Answers will be saved under {output_dir}")

    def evaluate(
        self,
        student_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Evaluate search results against a ground-truth dataset."""
        print(f"Evaluating results={student_results_path}")
        print(f"Ground truth dataset={dataset_path}")
        print(f"k={k}")


def main() -> None:
    """Run the Fire CLI."""
    fire.Fire(StudentCLI)
