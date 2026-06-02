import ast
from student.indexing.files_reader import SourceFile, FilesReader
from student.indexing.chunk import Chunk


class PyChunkerErr(Exception):
    pass


class PyChunker:
    def __init__(self, max_chunk_size: int) -> None:
        self.chunks: list[Chunk] = []
        self.max_chunk_size = max_chunk_size

    def parse_all_files(self, filereader: FilesReader) -> None:
        for file in filereader.py_files:
            if not self.parse_file(file):
                file.kind = "text"

    def parse_file(self, file: SourceFile) -> bool:
        try:
            offsets = self.lines_to_offsets(file.content)
            print(offsets)
            tree = ast.parse(file.content)
        except SyntaxError:
            return False
        chunks = self.create_chunk(tree, offsets)
        for chunk in chunks:
            start = chunk[0]
            end = chunk[1]
            self.chunks.append(
                Chunk(
                    filepath=file.filepath,
                    content=file.content[start:end],
                    first_character_index=start,
                    last_character_index=end,
                )
            )
        return True

    def create_chunk(
            self,
            tree: ast.Module,
            offsets: list[int]
            ) -> list[tuple[int, int]]:
        chunks = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                start, end = self.node_range(node, offsets)
                if end - start <= self.max_chunk_size:
                    chunks.append((start, end))
                else:
                    for met in node.body:
                        if isinstance(met, (ast.FunctionDef,
                                              ast.AsyncFunctionDef)):
                            met_start, met_end = self.node_range(met, offsets)
                            if met_end - met_start <= self.max_chunk_size:
                                chunks.append((met_start, met_end))
                            else:
                                


            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start, end = self.node_range(node, offsets)
                chunks.append((start, end))
        return chunks
    
    def add_func(
        self,
        func: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        offsets: list[int],
    ) -> bool:
        method_start, method_end = self.node_range(
                                func,
                                offsets
                                )
        if method_end - method_start <= self.max_chunk_size:
            return True
        else:
            return False
        

    def node_range(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        offsets: list[int],
    ) -> tuple[int, int]:
        start_line = node.lineno
        end_line = node.end_lineno

        if end_line is None:
            end_line = start_line

        start = offsets[start_line - 1]
        end = offsets[end_line]

        return start, end

    def lines_to_offsets(self, content: str) -> list[int]:
        offsets: list[int] = [0]
        current = 0

        for line in content.splitlines(keepends=True):
            current += len(line)
            offsets.append(current)

        return offsets
