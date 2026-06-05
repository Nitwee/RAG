from student.indexing.files_reader import FilesReader, FilesReaderErr
from student.indexing.py_chunker import PyChunker
from student.indexing.txt_chunker import TxtChunker
from student.indexing.chunk import ChunkError


class IndexManagerError(Exception):
    pass


class IndexManager:
    def __init__(self, src: str, max_chunk_size: int) -> None:
        try:
            self.files = FilesReader(src)
            self.py_chunks = PyChunker(max_chunk_size)
            self.py_chunks.parse_all_files(self.files)
            self.txt_chunks = TxtChunker(max_chunk_size)
            self.txt_chunks.parse_all_files(self.files)
            self.chunks = self.py_chunks.chunks + self.txt_chunks.chunks

        except (FilesReaderErr, ChunkError) as e:
            raise IndexManagerError(f"Indexing Error: {e}")
