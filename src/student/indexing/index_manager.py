"""Coordinate file reading and chunking for indexing."""

from student.indexing.files_reader import FilesReader, FilesReaderErr
from student.indexing.py_chunker import PyChunker
from student.indexing.txt_chunker import TxtChunker
from student.indexing.chunk import ChunkError


class IndexManagerError(Exception):
    """Raised when repository indexing fails."""

    pass


class IndexManager:
    """Build the complete chunk collection for a repository."""

    def __init__(self, src: str, max_chunk_size: int) -> None:
        """Read files and run Python/text chunkers."""

        try:
            self.files = FilesReader(src)
            self.py_chunks = PyChunker(max_chunk_size)
            self.py_chunks.parse_all_files(self.files)
            self.txt_chunks = TxtChunker(max_chunk_size)
            self.txt_chunks.parse_all_files(self.files)
            self.chunks = self.py_chunks.chunks + self.txt_chunks.chunks

        except (FilesReaderErr, ChunkError) as e:
            raise IndexManagerError(f"Indexing Error: {e}")
