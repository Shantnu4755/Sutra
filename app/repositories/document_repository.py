import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models.document import Chunk, Document, DocumentStatus


class DocumentRepository:
    def add(self, s: Session, document: Document) -> None:
        s.add(document)

    def get(self, s: Session, document_id: uuid.UUID) -> Document | None:
        return s.get(Document, document_id)

    def get_by_sha256(self, s: Session, sha256: str) -> Document | None:
        return s.execute(select(Document).where(Document.sha256 == sha256)).scalar_one_or_none()

    def recent(self, s: Session, limit: int, offset: int) -> list[Document]:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        return list(s.execute(stmt).scalars())

    def lock_next_pending(self, s: Session) -> Document | None:
        """Oldest PENDING document, row-locked for the whole ingest; other workers skip it."""
        stmt = (
            select(Document)
            .where(Document.status == DocumentStatus.PENDING.value)
            .order_by(Document.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        return s.execute(stmt).scalar_one_or_none()

    def chunk_counts(self, s: Session, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
        stmt = (
            select(Chunk.document_id, func.count())
            .where(Chunk.document_id.in_(document_ids))
            .group_by(Chunk.document_id)
        )
        return {doc_id: count for doc_id, count in s.execute(stmt).all()}
