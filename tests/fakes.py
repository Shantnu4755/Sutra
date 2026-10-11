import hashlib
import math
import re

from app.models.document import EMBEDDING_DIM

_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "for",
    "to",
    "in",
    "on",
    "what",
    "how",
    "my",
    "i",
    "do",
    "does",
    "any",
}


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words are close, unrelated ones orthogonal.
    Stands in for the ONNX model so tests need no download and no network."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            vector = [0.0] * EMBEDDING_DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word not in _STOP:
                    vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in vector))
            vectors.append([x / norm for x in vector] if norm else [1.0] + [0.0] * (EMBEDDING_DIM - 1))
        return vectors
