import uuid

from sqlalchemy import func, insert, select, text
from sqlalchemy.orm import Session

from ..models.document import Chunk

HYBRID_SQL = text("""
WITH vec AS (
    SELECT id, embedding <=> CAST(:qvec AS vector) AS dist,
           ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:qvec AS vector)) AS r
      FROM chunks
     ORDER BY embedding <=> CAST(:qvec AS vector)
     LIMIT 20
),
kw AS (
    SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank(c.tsv, q) DESC) AS r
      FROM chunks c, plainto_tsquery('english', :question) q
     WHERE c.tsv @@ q
     ORDER BY ts_rank(c.tsv, q) DESC
     LIMIT 20
)
SELECT c.id, c.page, c.text, c.document_id, d.title, vec.dist,
       COALESCE(1.0 / (60 + vec.r), 0) + COALESCE(1.0 / (60 + kw.r), 0) AS score
  FROM chunks c
  JOIN documents d ON d.id = c.document_id
  LEFT JOIN vec ON vec.id = c.id
  LEFT JOIN kw  ON kw.id  = c.id
 WHERE vec.id IS NOT NULL OR kw.id IS NOT NULL
 ORDER BY score DESC
 LIMIT :k
""")


class ChunkRepository:
    def hybrid_search(self, s: Session, question: str, qvec: list[float], k: int = 5) -> list[dict]:
        params = {"question": question, "qvec": str(qvec), "k": k}
        return [dict(r) for r in s.execute(HYBRID_SQL, params).mappings()]

    def add_many(
        self, s: Session, document_id: uuid.UUID, pieces: list[tuple[int, str]], vectors: list[list[float]]
    ) -> None:
        rows = [
            {"document_id": document_id, "page": page, "text": body, "embedding": vector}
            for (page, body), vector in zip(pieces, vectors, strict=True)
        ]
        if rows:
            s.execute(insert(Chunk), rows)

    def corpus_signature(self, s: Session) -> str:
        """Changes whenever chunks are added or removed: part of the answer cache key."""
        count, last_id = s.execute(select(func.count(), func.coalesce(func.max(Chunk.id), 0))).one()
        return f"{count}-{last_id}"
