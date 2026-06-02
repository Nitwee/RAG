from student.indexing.files_reader import FilesReader, FilesReaderErr
from student.indexing.py_chunker import PyChunker, PyChunkerErr


class IndexError(Exception):
    pass


class IndexManager:
    def __init__(self, src: str, max_chunk_size: int) -> None:
        try:
            self.files = FilesReader(src)
            self.py_chunks = PyChunker(max_chunk_size)
            self.py_chunks.parse_all_files(self.files)
        except (FilesReaderErr, PyChunkerErr) as e:
            raise IndexError(f"Index Error: {e}")
