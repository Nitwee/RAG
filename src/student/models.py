"""Pydantic data models used by the RAG pipeline."""

from __future__ import annotations

import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Minimal source location used by the evaluator."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without expected answer or sources."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with expected answer and source annotations."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset containing answered or unanswered RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Generated answer with its retrieved sources."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search result file produced by the student pipeline."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Answer result file produced by the student pipeline."""

    search_results: List[MinimalAnswer]
