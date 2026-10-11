import hashlib
import logging
import shutil
import uuid
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader
from pypdf.errors import PyPdfError
from sqlalchemy.exc import IntegrityError

from ..exceptions import NotFoundError, ValidationError
from ..extensions import db
from ..models.document import Document, DocumentStatus
from ..repositories.document_repository import DocumentRepository
from ..schemas.document import DocumentFilter, DocumentOut

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 10 * 1024 * 1024


class Upload(Protocol):
    filename: str | None

    def save(self, dst: str | Path) -> None: ...


class DocumentService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.documents = DocumentRepository()

    def pdf_path(self, document_id: uuid.UUID) -> Path:
        return self.data_dir / "documents" / f"{document_id}.pdf"

    def create(self, upload: Upload, title: str | None = None) -> tuple[DocumentOut, bool]:
        """Returns (document, created). The same file again returns the existing record."""
        incoming = self.data_dir / "incoming" / f"{uuid.uuid4()}.pdf"
        incoming.parent.mkdir(parents=True, exist_ok=True)
        upload.save(incoming)
        try:
            return self._register(
                incoming, title or _pdf_title(incoming) or Path(upload.filename or "document").stem
            )
        finally:
            incoming.unlink(missing_ok=True)  # moved into place when accepted

    def create_from_path(self, path: Path, title: str | None = None) -> tuple[DocumentOut, bool]:
        """For seeding from disk (CLI): the file is copied, never moved."""
        incoming = self.data_dir / "incoming" / f"{uuid.uuid4()}.pdf"
        incoming.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, incoming)
        try:
            return self._register(incoming, title or _pdf_title(incoming) or path.stem)
        finally:
            incoming.unlink(missing_ok=True)

    def list(self, f: DocumentFilter) -> list[DocumentOut]:
        with db.session() as s:
            docs = self.documents.recent(s, f.limit, f.offset)
            counts = self.documents.chunk_counts(s, [d.id for d in docs])
            return [
                DocumentOut.model_validate(d).model_copy(update={"chunks": counts.get(d.id, 0)}) for d in docs
            ]

    def get(self, document_id: uuid.UUID) -> DocumentOut:
        with db.session() as s:
            doc = self.documents.get(s, document_id)
            if doc is None:
                raise NotFoundError("Document not found", {"document_id": str(document_id)})
            count = self.documents.chunk_counts(s, [doc.id]).get(doc.id, 0)
            return DocumentOut.model_validate(doc).model_copy(update={"chunks": count})

    def _register(self, path: Path, title: str) -> tuple[DocumentOut, bool]:
        if path.stat().st_size > MAX_PDF_BYTES:
            raise ValidationError("PDF is larger than 10 MB")
        with path.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise ValidationError("File is not a PDF")
        sha256 = _sha256(path)
        with db.session() as s:
            existing = self.documents.get_by_sha256(s, sha256)
            if existing is not None:
                if existing.status == DocumentStatus.FAILED.value:  # uploading it again retries it
                    self._move_into_place(path, existing.id)
                    existing.status, existing.error = DocumentStatus.PENDING.value, None
                    logger.info("document_requeued", extra={"code": str(existing.id)})
                return DocumentOut.model_validate(existing), False
            doc = Document(title=title.strip()[:200] or "Untitled", sha256=sha256)
            self.documents.add(s, doc)
            try:
                s.flush()
            except IntegrityError:  # the same file uploaded twice at once
                s.rollback()
                winner = self.documents.get_by_sha256(s, sha256)
                if winner is None:
                    raise
                return DocumentOut.model_validate(winner), False
            self._move_into_place(path, doc.id)
            logger.info("document_queued", extra={"code": str(doc.id)})
            return DocumentOut.model_validate(doc), True

    def _move_into_place(self, path: Path, document_id: uuid.UUID) -> None:
        target = self.pdf_path(document_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pdf_title(path: Path) -> str | None:
    """The title stored in the PDF itself, if any. A broken file is left for the worker to report."""
    try:
        title = PdfReader(path).metadata.title if path.stat().st_size else None
    except (PyPdfError, OSError, ValueError, AttributeError):
        return None
    return title.strip() if title and title.strip() else None
