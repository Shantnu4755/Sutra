"""Import every model module here so Alembic autogenerate and create_all see all tables."""

from .document import EMBEDDING_DIM, Chunk, Document, DocumentStatus

__all__ = ["EMBEDDING_DIM", "Chunk", "Document", "DocumentStatus"]
