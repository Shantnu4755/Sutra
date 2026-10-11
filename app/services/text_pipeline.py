import threading

from fastembed import TextEmbedding


class Chunker:
    def __init__(self, size: int = 800, overlap: int = 150) -> None:
        if overlap >= size // 2:
            raise ValueError("overlap must be less than half of size")
        self.size, self.overlap = size, overlap

    def split(self, text: str) -> list[str]:
        text = " ".join(text.split())
        chunks, start = [], 0
        while start < len(text):
            end = min(len(text), start + self.size)
            if end < len(text):
                cut = text.rfind(". ", start + self.size // 2, end)  # prefer a sentence boundary
                if cut != -1:
                    end = cut + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = end - self.overlap
        return [c for c in chunks if c]


class Embedder:
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    _models: dict[str, TextEmbedding] = {}  # loaded once per process (per cache folder)
    _lock = threading.Lock()

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = cache_dir

    def _get(self) -> TextEmbedding:
        key = self.cache_dir or ""
        with self._lock:
            model = self._models.get(key)
            if model is None:
                model = TextEmbedding(model_name=self.MODEL_NAME, cache_dir=self.cache_dir)
                self._models[key] = model
            return model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get().embed(texts, batch_size=32)]
