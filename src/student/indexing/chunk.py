from pydantic import BaseModel
from pathlib import Path


class Chunk(BaseModel):
    filepath: Path
    content: str
    first_character_index: int
    last_character_index: int
