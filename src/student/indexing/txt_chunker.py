"""Text fallback chunking for non-Python files."""

from student.indexing.files_reader import SourceFile, FilesReader
from student.indexing.chunk import Chunk, Chunker, ChunkError
from pydantic import ValidationError


class TxtChunker(Chunker):
    """Split text files into bounded character ranges."""

    def __init__(self, max_chunk_size: int) -> None:
        """Initialize an empty text chunk collection."""

        super().__init__(max_chunk_size)
        self.chunks: list[Chunk] = []
        self.max_chunk_size = max_chunk_size

    def parse_all_files(self, filereader: FilesReader) -> None:
        """Parse all text files collected by a file reader."""

        for file in filereader.txt_files:
            self.parse_file(file)

    def parse_file(self, file: SourceFile) -> None:
        """Split one text file and append validated chunks."""

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
