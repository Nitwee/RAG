from pathlib import Path
from pydantic import BaseModel
from typing import Literal


class FilesReaderErr(Exception):
    pass


class SourceFile(BaseModel):
    filepath: Path
    content: str
    kind: Literal["python", "text"]


class FilesReader:
    def __init__(self, src: str) -> None:
        self.py_files: list[SourceFile] = []
        self.txt_files: list[SourceFile] = []
        folder = self.check_folder(src)
        self.read_files(folder)

    def check_folder(self, src: str) -> Path:
        folder = Path(src)
        if not folder.exists():
            raise FilesReaderErr(f"Folder {src} does not exists.")
        if not folder.is_dir():
            raise FilesReaderErr(f"{src} is not a folder.")
        return folder

    def read_files(self, folder: Path) -> None:
        all_files = folder.rglob("*")
        i = 0
        j = 0
        for filepath in all_files:
            i += 1
            if not filepath.is_file():
                continue

            filetype = self.file_type_sorter(filepath)
            if filetype == "python":
                j += 1
                self.py_files.append(self.read_source_file(filepath, filetype))
            elif filetype == "text":
                j += 1
                self.txt_files.append(self.read_source_file(filepath, filetype))
        print(i, j)


    def file_type_sorter(self, file: Path) -> Literal["python", "text"] | None:
        authorized = {".md", ".rst", ".txt", ".yaml", ".yml", ".toml"}
        if file.suffix == ".py":
            return "python"
        elif file.suffix in authorized:
            return "text"
        return None

    def read_source_file(
            self,
            filepath: Path,
            kind: Literal["python", "text"]
            ) -> SourceFile:
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
