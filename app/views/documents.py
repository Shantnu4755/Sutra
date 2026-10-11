import uuid
from pathlib import Path

from flask import request
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import ValidationError
from ..schemas.document import AskIn, DocumentFilter
from ..services.answer_service import AnswerService
from ..services.document_service import DocumentService
from .base import BaseView


class UploadForm(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str | None = Field(default=None, min_length=1, max_length=200)


class DocumentListView(BaseView):
    def __init__(self, data_dir: str) -> None:
        self.service = DocumentService(Path(data_dir))

    def get(self):
        docs = self.service.list(self.parse_query(DocumentFilter))
        return {"data": [d.model_dump(mode="json") for d in docs]}, 200

    def post(self):
        upload = request.files.get("file")
        if not upload:
            raise ValidationError("A PDF is required in the 'file' form field")
        if not (upload.filename or "").lower().endswith(".pdf"):
            raise ValidationError(f"{upload.filename} is not a .pdf file")
        form = UploadForm.model_validate(request.form.to_dict())
        doc, created = self.service.create(upload, form.title)
        return {"data": doc.model_dump(mode="json")}, (202 if created else 200)


class DocumentDetailView(BaseView):
    def __init__(self, data_dir: str) -> None:
        self.service = DocumentService(Path(data_dir))

    def get(self, document_id: uuid.UUID):
        return {"data": self.service.get(document_id).model_dump(mode="json")}, 200


class AskView(BaseView):
    def __init__(self, service: AnswerService) -> None:
        self.service = service

    def post(self):
        result = self.service.ask(self.parse_body(AskIn).question)
        return {"data": result.model_dump(mode="json")}, 200
