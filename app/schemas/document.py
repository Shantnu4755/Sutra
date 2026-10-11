import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ..models.document import DocumentStatus


class DocumentFilter(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    sha256: str
    status: DocumentStatus
    pages: int | None
    error: str | None
    created_at: datetime
    chunks: int = 0


class AskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    chunk_id: int
    document_id: uuid.UUID
    title: str
    page: int
    text: str  # shown in the UI's side panel when a citation chip is clicked


class AnswerOut(BaseModel):
    answer: str
    found: bool
    citations: list[Citation]
    cached: bool = False
