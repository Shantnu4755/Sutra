import logging
import uuid
from itertools import batched
from pathlib import Path

from pypdf import PdfReader

from ..exceptions import AppError, ValidationError
from ..extensions import db
from ..models.document import DocumentStatus
from ..repositories.chunk_repository import ChunkRepository
from ..repositories.document_repository import DocumentRepository
from .document_service import DocumentService
from .text_pipeline import Chunker, Embedder

logger = logging.getLogger(__name__)

EMBED_BATCH = 32


class IngestService:
    """PDF -> pages -> chunks -> embeddings -> rows. One document per call, row-locked meanwhile."""

    def __init__(self, data_dir: Path, embedder: Embedder, chunker: Chunker | None = None) -> None:
        self.files = DocumentService(data_dir)
        self.embedder = embedder
        self.chunker = chunker or Chunker()
        self.documents = DocumentRepository()
        self.chunks = ChunkRepository()

    def process_next(self) -> bool:
        """Returns False when nothing is waiting."""
        with db.session() as s:
            doc = self.documents.lock_next_pending(s)
            if doc is None:
                return False
            path = self.files.pdf_path(doc.id)
            try:
                with s.begin_nested():  # a failure part-way leaves no half-ingested chunks
                    pieces, pages = self._pieces(path)
                    for batch in batched(pieces, EMBED_BATCH):
                        batch = list(batch)
                        self.chunks.add_many(
                            s, doc.id, batch, self.embedder.embed([text for _, text in batch])
                        )
                doc.status, doc.pages, doc.error = DocumentStatus.READY.value, pages, None
                logger.info("document_ready", extra={"code": str(doc.id)})
            except Exception as exc:
                reason = exc.message if isinstance(exc, AppError) else f"{type(exc).__name__}: {exc}"
                logger.warning("document_failed", extra={"code": str(doc.id), "error_message": reason[:300]})
                doc.status, doc.error = DocumentStatus.FAILED.value, reason[:1000]
            document_id = doc.id
        self._remove(document_id)
        return True

    def _pieces(self, path: Path) -> tuple[list[tuple[int, str]], int]:
        reader = PdfReader(path)
        pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
        pieces = [(number, piece) for number, text in pages for piece in self.chunker.split(text)]
        if not pieces:
            raise ValidationError("No text could be extracted from this PDF (is it a scanned image?)")
        return pieces, len(pages)

    def _remove(self, document_id: uuid.UUID) -> None:
        try:
            self.files.pdf_path(document_id).unlink(missing_ok=True)
        except OSError:
            logger.exception("remove_pdf_failed", extra={"code": str(document_id)})
