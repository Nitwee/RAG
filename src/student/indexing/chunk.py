from pydantic import BaseModel, Field, ValidationError
from pathlib import Path
from abc import ABC, abstractmethod
from student.indexing.files_reader import FilesReader


class ChunkError(Exception):
    pass


class Chunk(BaseModel):
    filepath: Path
    content: str
    first_character_index: int
    last_character_index: int


class ChunkSize(BaseModel):
    max_chunk_size: int = Field(default=2000, ge=1, le=2000)


class Chunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        try:
            ChunkSize(max_chunk_size=max_chunk_size)
        except ValidationError as e:
            raise ChunkError(e)
        self.max_chunk_size = max_chunk_size

    def add_range(
        self,
        content: str,
        start: int,
        end: int,
        chunks: list[tuple[int, int]],
    ) -> None:
        while start < end:
            chunk_end = min(start + self.max_chunk_size, end)

            if chunk_end < end:
                newline = content.rfind("\n", start, chunk_end)
                if newline > start:
                    chunk_end = newline + 1

            chunks.append((start, chunk_end))
            start = chunk_end

    @abstractmethod
    def parse_all_files(self, filereader: FilesReader) -> None:
        ...
