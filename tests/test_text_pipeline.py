import random
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.text_pipeline import Chunker, Embedder


def test_chunker_never_loops_and_respects_overlap():
    rng = random.Random(3)
    words = ["claim", "policy", "cover", "insured", "premium", "hospital", "waiting", "period"]
    for length in (0, 50, 799, 800, 801, 5_000, 50_000):
        text = " ".join(rng.choice(words) for _ in range(length // 6))[:length]
        started = time.perf_counter()
        chunks = Chunker(size=800, overlap=150).split(text)
        assert time.perf_counter() - started < 1
        assert all(len(c) <= 800 for c in chunks)
        assert " ".join(text.split()) == "" or chunks[0].startswith(" ".join(text.split())[:10])
        for first, second in zip(chunks, chunks[1:], strict=False):
            # each chunk starts inside the previous one, about `overlap` characters from its end
            overlap = len(first) - first.rfind(second[:100])
            assert second[:100] in first and 140 <= overlap <= 152


def test_chunker_prefers_sentence_boundaries():
    text = " ".join(f"Sentence number {n} talks about the policy cover." for n in range(60))
    chunks = Chunker(size=400, overlap=80).split(text)
    assert len(chunks) > 3
    assert all(c.endswith(".") for c in chunks)


def test_chunker_survives_pathological_input():
    for text in ("a. " * 5_000, "." * 3_000, ". " * 2_000, "x" * 10_000):
        chunks = Chunker(size=100, overlap=40).split(text)
        assert chunks and all(len(c) <= 100 for c in chunks)


def test_chunker_rejects_overlap_of_half_or_more():
    with pytest.raises(ValueError):
        Chunker(size=100, overlap=50)


def test_embedder_loads_the_model_once_per_cache_dir(tmp_path):
    model = MagicMock()
    model.embed.return_value = iter([np.array([0.5, 0.5])])
    Embedder._models.clear()
    with patch("app.services.text_pipeline.TextEmbedding", return_value=model) as factory:
        embedder = Embedder(str(tmp_path))
        assert embedder.embed(["hello"]) == [[0.5, 0.5]]
        model.embed.return_value = iter([np.array([1.0, 0.0])])
        Embedder(str(tmp_path)).embed(["again"])
    factory.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(tmp_path))
    Embedder._models.clear()
