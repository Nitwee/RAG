from student.indexing.files_reader import SourceFile, FilesReader
from student.indexing.chunk import Chunk, Chunker, ChunkError
from pydantic import ValidationError


class TxtChunker(Chunker):
    def __init__(self, max_chunk_size: int) -> None:
        super().__init__(max_chunk_size)
        self.chunks: list[Chunk] = []
        self.max_chunk_size = max_chunk_size

    def parse_all_files(self, filereader: FilesReader) -> None:
        for file in filereader.txt_files:
            self.parse_file(file)

    def parse_file(self, file: SourceFile) -> None:
        chunks: list[tuple[int, int]] = []
        self.add_range(file.content, 0, len(file.content), chunks)
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
