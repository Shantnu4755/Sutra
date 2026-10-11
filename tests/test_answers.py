from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import redis
import requests

from app.evaluate import evaluate
from app.exceptions import ExternalServiceError
from app.services.answer_service import NOT_FOUND_TEXT, AnswerService
from app.services.document_service import DocumentService
from app.services.ingest_service import IngestService
from app.services.llm_client import LLMClient
from scripts.make_sample_pdfs import text_to_pdf
from tests.conftest import SAMPLES
from tests.fakes import FakeEmbedder

LOOSE = 0.8  # bag-of-words fake: the relevant chunk sits at ~0.73, unrelated text at >= 0.90


@pytest.fixture
def library(app):
    data_dir = Path(app.config["APP_CONFIG"].data_dir)
    for name in ("arogya-shield-health", "safar-motor", "yatra-secure-travel"):
        DocumentService(data_dir).create_from_path(SAMPLES / f"{name}.pdf")
        IngestService(data_dir, FakeEmbedder()).process_next()


@pytest.fixture
def llm():
    fake = MagicMock(spec=LLMClient)
    fake.complete.return_value = "The initial waiting period is 30 days [Arogya Shield Health Policy, p.1]."
    return fake


@pytest.fixture
def service(app, llm) -> AnswerService:
    return AnswerService(llm, app.extensions["sutra"]["cache"], FakeEmbedder(), max_distance=LOOSE)


# --- PDF "Tests to write" ---------------------------------------------------------------


def test_no_relevant_text_returns_not_found_without_calling_the_llm(library, service, llm):
    result = service.ask("vaccination schedule for pet dogs")
    assert (result.found, result.answer, result.citations) == (False, NOT_FOUND_TEXT, [])
    llm.complete.assert_not_called()


def test_llm_429_is_retried():
    client = LLMClient("https://llm.example/v1", "key", "model")
    busy = MagicMock(status_code=429)
    ok = MagicMock(status_code=200)
    ok.json.return_value = {"choices": [{"message": {"content": " An answer. "}}]}
    with patch.object(client.http, "post", side_effect=[busy, ok]) as post, patch("time.sleep") as sleep:
        assert client.complete("system", "user") == "An answer."
    assert post.call_count == 2
    sleep.assert_called_once_with(1)


def test_redis_down_still_answers(library, llm):
    dead = redis.Redis(host="127.0.0.1", port=9, socket_timeout=0.2, socket_connect_timeout=0.2)
    result = AnswerService(llm, dead, FakeEmbedder(), max_distance=LOOSE).ask("initial waiting period")
    assert result.found and "30 days" in result.answer


# --- answering ------------------------------------------------------------------------


def test_answer_cites_the_chunks_it_used(library, service, llm):
    result = service.ask("What is the initial waiting period?")
    assert result.found and not result.cached
    assert {(c.title, c.page) for c in result.citations} == {("Arogya Shield Health Policy", 1)}
    assert "initial waiting period" in result.citations[0].text.lower()
    prompt = llm.complete.call_args.args[1]
    assert "Context:" in prompt and "[Arogya Shield Health Policy, p.1]" in prompt


def test_answer_without_matching_citations_lists_all_sources(library, service, llm):
    llm.complete.return_value = "It is 30 days."
    result = service.ask("What is the initial waiting period?")
    assert len(result.citations) == 5


@pytest.mark.parametrize("reply", ["NOT_FOUND", "NOT_FOUND.", " not_found "])
def test_llm_not_found_is_an_honest_not_found(library, service, llm, reply):
    llm.complete.return_value = reply
    result = service.ask("What is the initial waiting period?")
    assert (result.found, result.answer) == (False, NOT_FOUND_TEXT)


def test_answers_are_cached_until_the_library_changes(app, library, service, llm):
    first = service.ask("What is the initial waiting period?")
    second = service.ask("  what is the INITIAL waiting period?")
    assert (first.cached, second.cached, llm.complete.call_count) == (False, True, 1)
    data_dir = Path(app.config["APP_CONFIG"].data_dir)
    policies = SAMPLES.parent / "policies"  # any new document changes the corpus signature
    path = data_dir / "extra.pdf"
    path.write_bytes(text_to_pdf("Extra", (policies / "safar-motor.txt").read_text() + "\nExtra clause."))
    DocumentService(data_dir).create_from_path(path)
    IngestService(data_dir, FakeEmbedder()).process_next()
    assert service.ask("What is the initial waiting period?").cached is False


def test_llm_errors_become_502(app, client, library, monkeypatch):
    answers = app.extensions["sutra"]["answers"]
    monkeypatch.setattr(answers, "embedder", FakeEmbedder())
    monkeypatch.setattr(answers, "max_distance", LOOSE)
    with (
        patch.object(answers.llm.http, "post", side_effect=requests.ConnectionError("down")),
        patch("time.sleep"),
    ):
        resp = client.post("/ask", json={"question": "What is the initial waiting period?"})
    assert resp.status_code == 502
    assert resp.get_json()["error"]["code"] == "external_service_error"


def test_unconfigured_llm_is_reported():
    with pytest.raises(ExternalServiceError, match="not configured"):
        LLMClient("https://llm.example/v1", "", "").complete("s", "u")


def test_llm_gives_up_after_retries():
    client = LLMClient("https://llm.example/v1", "key", "model", retries=1)
    broken = MagicMock(status_code=200)
    broken.json.return_value = {"unexpected": True}
    with patch.object(client.http, "post", return_value=broken), patch("time.sleep"):
        with pytest.raises(ExternalServiceError):
            client.complete("s", "u")


def test_ask_endpoint(app, client, library, monkeypatch, llm):
    answers = app.extensions["sutra"]["answers"]
    monkeypatch.setattr(answers, "embedder", FakeEmbedder())
    monkeypatch.setattr(answers, "max_distance", LOOSE)
    monkeypatch.setattr(answers, "llm", llm)
    resp = client.post("/ask", json={"question": "What is the initial waiting period?"})
    body = resp.get_json()["data"]
    assert resp.status_code == 200 and body["found"] is True
    assert body["citations"][0]["title"] == "Arogya Shield Health Policy"
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "extra": 1}).status_code == 422


def test_evaluate_scores_answers_and_not_found(library, service, llm):
    questions = [
        {
            "id": 1,
            "question": "initial waiting period",
            "expect": {"document": "Arogya Shield Health Policy", "keywords": ["30 days"]},
        },
        {
            "id": 2,
            "question": "initial waiting period?",
            "expect": {"document": "Safar Motor Policy", "keywords": ["30 days"]},
        },
        {"id": 3, "question": "vaccination schedule for pet dogs", "expect": "not_found"},
    ]
    summary = evaluate(service, questions)
    assert (summary["questions"], summary["accuracy"], summary["not_found_rate"]) == (3, 0.5, 1.0)
    assert [r["correct"] for r in summary["results"]] == [True, False, True]
