import hashlib
import json
import logging
import re

import redis

from ..extensions import db
from ..repositories.chunk_repository import ChunkRepository
from ..schemas.document import AnswerOut, Citation
from .llm_client import LLMClient
from .text_pipeline import Embedder

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions about insurance policies using ONLY the context given. "
    "Cite every fact as [Title, p.N]. If the context does not contain the answer, reply exactly NOT_FOUND."
)
MAX_DISTANCE = 0.55  # tune with your evaluation set
NOT_FOUND_TEXT = "I could not find this in the uploaded documents."
CACHE_SECONDS = 86400
_CITATION = re.compile(r"\[([^\[\]]+?),\s*p\.?\s*(\d+)\]")


class AnswerService:
    def __init__(
        self,
        llm: LLMClient,
        cache: redis.Redis,
        embedder: Embedder | None = None,
        max_distance: float = MAX_DISTANCE,
    ) -> None:
        self.llm, self.cache = llm, cache
        self.embedder, self.chunks = embedder or Embedder(), ChunkRepository()
        self.max_distance = max_distance

    def ask(self, question: str) -> AnswerOut:
        with db.session() as s:
            corpus = self.chunks.corpus_signature(s)  # new documents invalidate cached answers
        digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()
        key = f"ans:{corpus}:{digest}"
        cached = self._cache_get(key)
        if cached:
            return AnswerOut(**cached, cached=True)

        qvec = self.embedder.embed([question])[0]
        with db.session() as s:
            hits = self.chunks.hybrid_search(s, question, qvec)

        best = min((h["dist"] for h in hits if h["dist"] is not None), default=None)
        if not hits or best is None or best > self.max_distance:
            return AnswerOut(answer=NOT_FOUND_TEXT, found=False, citations=[])

        context = "\n\n".join(f"[{h['title']}, p.{h['page']}]\n{h['text']}" for h in hits)
        answer = self.llm.complete(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
        if answer.strip().rstrip(".").upper() == "NOT_FOUND":
            result = AnswerOut(answer=NOT_FOUND_TEXT, found=False, citations=[])
        else:
            result = AnswerOut(answer=answer, found=True, citations=self._citations(answer, hits))
        self._cache_set(key, result)
        return result

    @staticmethod
    def _citations(answer: str, hits: list[dict]) -> list[Citation]:
        """The chunks the answer actually cites; all retrieved chunks if it cites none we can match."""
        cited = {(title.strip().lower(), int(page)) for title, page in _CITATION.findall(answer)}
        used = [h for h in hits if (h["title"].strip().lower(), h["page"]) in cited] or hits
        return [
            Citation(
                chunk_id=h["id"],
                document_id=h["document_id"],
                title=h["title"],
                page=h["page"],
                text=h["text"],
            )
            for h in used
        ]

    def _cache_get(self, key: str) -> dict | None:
        try:
            raw = self.cache.get(key)
            return json.loads(raw) if raw else None
        except redis.RedisError:
            logger.warning("cache_unavailable")  # cache is optional; never fail the request
            return None

    def _cache_set(self, key: str, value: AnswerOut) -> None:
        try:
            self.cache.setex(
                key, CACHE_SECONDS, json.dumps(value.model_dump(mode="json", exclude={"cached"}))
            )
        except redis.RedisError:
            logger.warning("cache_unavailable")
