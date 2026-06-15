"""Read source files from the indexed repository."""

from pathlib import Path
from pydantic import BaseModel
from typing import Literal


class FilesReaderErr(Exception):
    """Raised when repository files cannot be read safely."""

    pass


class SourceFile(BaseModel):
    """In-memory representation of one source file."""

    filepath: Path
    content: str
    kind: Literal["python", "text"]


class FilesReader:
    """Collect Python and text files from a repository folder."""

    def __init__(self, src: str) -> None:
        """Validate the source folder and read supported files."""

        self.py_files: list[SourceFile] = []
        self.txt_files: list[SourceFile] = []
        folder = self.check_folder(src)
        self.read_files(folder)

    def check_folder(self, src: str) -> Path:
        """Return a valid repository folder path."""

        folder = Path(src)
        if not folder.exists():
            raise FilesReaderErr(f"Folder {src} does not exists.")
        if not folder.is_dir():
            raise FilesReaderErr(f"{src} is not a folder.")
        return folder

    def read_files(self, folder: Path) -> None:
        """Populate Python and text file lists from a folder tree."""

        all_files = folder.rglob("*")
        for filepath in all_files:
            if not filepath.is_file():
                continue
            filetype = self.file_type_sorter(filepath)
            if filetype == "python":
                self.py_files.append(self.read_file(filepath, filetype))
            elif filetype == "text":
                self.txt_files.append(self.read_file(filepath, filetype))

    def file_type_sorter(self, file: Path) -> Literal["python", "text"] | None:
        """Classify a path as Python, supported text, or ignored."""

        authorized = {
            ".cmake",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".html",
            ".in",
            ".inl",
            ".jinja",
            ".js",
            ".md",
            ".rst",
            ".sh",
            ".toml",
            ".tpl",
            ".txt",
            ".yaml",
            ".yml",
        }
        if file.suffix == ".py":
            return "python"
        elif file.suffix in authorized:
            return "text"
        return None

    def read_file(
            self,
            filepath: Path,
            kind: Literal["python", "text"]
            ) -> SourceFile:
        """Read one UTF-8 file and return a source model."""

        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise FilesReaderErr(f"Cannot decode {filepath} as UTF-8") from e
        except FileNotFoundError as e:
            raise FilesReaderErr(f"File {filepath} not found.") from e
        except PermissionError as e:
            raise FilesReaderErr(f"Permission denied on {filepath}.") from e
        except IsADirectoryError as e:
            raise FilesReaderErr(f"{filepath} is a directory.") from e
        except OSError as e:
            raise FilesReaderErr(f"OS error reading {filepath}: {e}") from e

        return SourceFile(
            filepath=filepath,
            content=content,
            kind=kind,
        )
