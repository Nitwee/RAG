"""AST-based chunking for Python source files."""

import ast
from pydantic import ValidationError
from student.indexing.files_reader import SourceFile, FilesReader
from student.indexing.chunk import Chunk, Chunker, ChunkError


class PyChunker(Chunker):
    """Split Python files around classes and functions."""

    def __init__(self, max_chunk_size: int) -> None:
        """Initialize an empty Python chunk collection."""

        super().__init__(max_chunk_size)
        self.chunks: list[Chunk] = []
        self.max_chunk_size = max_chunk_size

    def parse_all_files(self, filereader: FilesReader) -> None:
        """Parse all Python files, falling back to text on syntax errors."""

        for file in filereader.py_files:
            if not self.parse_file(file):
                file.kind = "text"
                filereader.txt_files.append(file)

    def parse_file(self, file: SourceFile) -> bool:
        """Parse one Python file and append its chunks."""

        try:
            offsets = self.lines_to_offsets(file.content)
            tree = ast.parse(file.content)
        except SyntaxError:
            return False
        chunks = self.split_chunks(tree, file.content, offsets)
        for chunk in chunks:
            start = chunk[0]
            end = chunk[1]
            try:
                self.chunks.append(
                    Chunk(
                        filepath=file.filepath,
                        content=file.content[start:end],
                        first_character_index=start,
                        last_character_index=end,
                    )
                )
            except ValidationError as e:
                raise ChunkError(f"Invalid chunk for {file.filepath}") from e

        return True

    def split_chunks(
            self,
            tree: ast.Module,
            content: str,
            offsets: list[int]
            ) -> list[tuple[int, int]]:
        """Return source ranges for top-level Python code units."""

        chunks: list[tuple[int, int]] = []
        header_end = None

        for node in tree.body:
            if isinstance(node,
                          (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                          ):
                header_end, _ = self.node_range(node, offsets)
                break
        if header_end is not None and header_end > 0:
            self.add_range(content, 0, header_end, chunks)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                start, end = self.node_range(node, offsets)
                if end - start <= self.max_chunk_size:
                    # Class is not too long -> Add all class
                    chunks.append((start, end))
                else:
                    # Class is too long -> Add methods one by one
                    method_found = False
                    for met in node.body:
                        if isinstance(met, (ast.FunctionDef,
                                            ast.AsyncFunctionDef)):
                            method_found = True
                            self.add_func(met, content, offsets, chunks)
                    if not method_found:
                        self.add_range(content, start, end, chunks)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Add same logic as for methods
                self.add_func(node, content, offsets, chunks)
        return chunks

    def add_func(
        self,
        func: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        content: str,
        offsets: list[int],
        chunks: list[tuple[int, int]]
    ) -> None:
        """Add one function or method range, splitting if needed."""

        start, end = self.node_range(func, offsets)
        self.add_range(content, start, end, chunks)

    def node_range(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        offsets: list[int],
    ) -> tuple[int, int]:
        """Convert an AST node line span to character offsets."""

        start_line = node.lineno
        end_line = node.end_lineno

        for decorator in node.decorator_list:
            start_line = min(start_line, decorator.lineno)

        if end_line is None:
            end_line = start_line

        start = offsets[start_line - 1]
        end = offsets[end_line]

        return start, end

    def lines_to_offsets(self, content: str) -> list[int]:
        """Build a line-number to character-offset lookup table."""

        offsets: list[int] = [0]
        current = 0

        for line in content.splitlines(keepends=True):
            current += len(line)
            offsets.append(current)

        return offsets
