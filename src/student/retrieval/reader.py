from pathlib import Path
from student.retrieval.bm_25 import BM25Retriever, BM25RetrieverError


class ReaderError(Exception):
    pass


class Reader:
    def load_bm25(self) -> None:
        try:
            self.retriever = BM25Retriever.load()
        except BM25RetrieverError as e:
            raise ReaderError(f"Cannot load retriever: {e}")

    def validate_read(
            self,
            dataset_path: str,
            ) -> tuple[str, Path]:
        input_path = Path(dataset_path)
        if not input_path.exists():
            raise ReaderError(f"Input_path {dataset_path} doesnt exist")
        if not input_path.is_file():
            raise ReaderError(f"Input_path {dataset_path} isnt a file")
        content = self.read_file(input_path)
        return (content, input_path)

    def read_file(self, filepath: Path) -> str:
        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ReaderError(f"Cannot decode {filepath} as UTF-8 {e}")
        except FileNotFoundError as e:
            raise ReaderError(f"File {filepath} not found. {e}")
        except PermissionError as e:
            raise ReaderError(f"Permission denied on {filepath}. {e}")
        except IsADirectoryError as e:
            raise ReaderError(f"{filepath} is a directory. {e}")
        except OSError as e:
            raise ReaderError(f"OS error reading {filepath}: {e}")
        return content

    def write_output(
        self,
        save_directory: str,
        filename: str,
        result: str,
    ) -> None:
        output_dir = Path(save_directory)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / filename
            output_file.write_text(result, encoding="utf-8")
        except PermissionError as e:
            raise ReaderError(
                f"Permission denied writing to {output_dir}"
            ) from e
        except OSError as e:
            raise ReaderError(
                f"Cannot write answer results: {e}"
            ) from e
