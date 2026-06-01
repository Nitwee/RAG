from pathlib import Path

class ChunkSplitter:
    def __init__(self, src: str) -> None:
        test = Path(src)
        all_files = test.rglob("*")
        for file in all_files:
            print(file)


    def read_files(self) -> bool:
        return True
    
