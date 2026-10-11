import io
import threading
import time
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import func, select

from app.extensions import db
from app.models.document import Document
from app.services.ingest_service import IngestService
from app.worker.worker import IngestWorker
from scripts.make_sample_pdfs import text_to_pdf
from tests.conftest import SAMPLES
from tests.fakes import FakeEmbedder


def _upload(client, content: bytes, name: str = "policy.pdf", title: str | None = None):
    data = {"file": (io.BytesIO(content), name)}
    if title:
        data["title"] = title
    return client.post("/documents", data=data, content_type="multipart/form-data")


def _ingest(data_dir: Path, embedder=None) -> bool:
    return IngestService(data_dir, embedder or FakeEmbedder()).process_next()


def _doc_count() -> int:
    with db.session() as s:
        return s.execute(select(func.count()).select_from(Document)).scalar_one()


def test_upload_then_ingest_a_sample_policy(client, data_dir):
    resp = _upload(client, (SAMPLES / "safar-motor.pdf").read_bytes(), name="motor.pdf")
    assert resp.status_code == 202
    doc = resp.get_json()["data"]
    assert (doc["status"], doc["title"], doc["chunks"]) == (
        "PENDING",
        "Safar Motor Policy",
        0,
    )  # title from the PDF
    assert (data_dir / "documents" / f"{doc['id']}.pdf").exists()

    assert _ingest(data_dir) is True
    ready = client.get(f"/documents/{doc['id']}").get_json()["data"]
    assert (ready["status"], ready["pages"], ready["error"]) == ("READY", 2, None)
    assert ready["chunks"] >= 4
    assert not (data_dir / "documents" / f"{doc['id']}.pdf").exists()  # removed once ingested
    assert _ingest(data_dir) is False  # nothing left
    assert [d["id"] for d in client.get("/documents").get_json()["data"]] == [doc["id"]]


def test_duplicate_upload_returns_the_existing_document(client):
    content = (SAMPLES / "yatra-secure-travel.pdf").read_bytes()
    first = _upload(client, content, title="Travel")
    second = _upload(client, content, name="copy.pdf", title="Another title")
    assert (first.status_code, second.status_code) == (202, 200)
    assert first.get_json()["data"]["id"] == second.get_json()["data"]["id"]
    assert second.get_json()["data"]["title"] == "Travel"
    assert _doc_count() == 1


def test_bad_uploads_are_rejected(client, data_dir):
    assert client.post("/documents", data={}, content_type="multipart/form-data").status_code == 422
    assert _upload(client, b"%PDF-1.4 fake", name="notes.txt").status_code == 422
    not_pdf = _upload(client, b"hello, I am a text file", name="sneaky.pdf")
    assert (not_pdf.status_code, not_pdf.get_json()["error"]["message"]) == (422, "File is not a PDF")
    too_big = _upload(client, b"%PDF-" + b"0" * (10 * 1024 * 1024), name="big.pdf")
    assert (too_big.status_code, too_big.get_json()["error"]["message"]) == (422, "PDF is larger than 10 MB")
    assert _doc_count() == 0
    assert list((data_dir / "incoming").iterdir()) == []  # temp files cleaned up


def test_broken_and_textless_pdfs_fail_with_a_reason(client, data_dir):
    broken = _upload(client, b"%PDF-1.4\nthis is not really a pdf", name="broken.pdf").get_json()["data"]
    blank = _upload(client, text_to_pdf("Blank", ""), name="blank.pdf").get_json()["data"]
    _ingest(data_dir)
    _ingest(data_dir)
    failed = {d["id"]: d for d in client.get("/documents").get_json()["data"]}
    assert failed[broken["id"]]["status"] == "FAILED" and failed[broken["id"]]["error"]
    assert failed[blank["id"]]["error"] == "No text could be extracted from this PDF (is it a scanned image?)"
    assert failed[blank["id"]]["chunks"] == 0


def test_uploading_a_failed_document_again_retries_it(client, data_dir):
    content = (SAMPLES / "safar-motor.pdf").read_bytes()
    doc_id = _upload(client, content).get_json()["data"]["id"]
    failing = FakeEmbedder()
    with patch.object(failing, "embed", side_effect=RuntimeError("model download failed")):
        _ingest(data_dir, failing)
    assert client.get(f"/documents/{doc_id}").get_json()["data"]["status"] == "FAILED"
    again = _upload(client, content)
    assert (again.status_code, again.get_json()["data"]["status"]) == (200, "PENDING")
    _ingest(data_dir)
    assert client.get(f"/documents/{doc_id}").get_json()["data"]["status"] == "READY"


def test_two_workers_never_ingest_the_same_document(client, data_dir):
    for name in ("arogya-shield-health", "safar-motor"):
        _upload(client, (SAMPLES / f"{name}.pdf").read_bytes())

    class SlowEmbedder(FakeEmbedder):
        def embed(self, texts):
            time.sleep(0.2)  # keep the first document locked while the second worker looks
            return super().embed(texts)

    results: list[bool] = []
    barrier = threading.Barrier(2)

    def work() -> None:
        barrier.wait()
        results.append(_ingest(data_dir, SlowEmbedder()))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [True, True]
    statuses = [d["status"] for d in client.get("/documents").get_json()["data"]]
    assert statuses == ["READY", "READY"]


def test_unknown_document_is_404(client):
    assert client.get("/documents/00000000-0000-0000-0000-000000000000").status_code == 404


def test_worker_loop_ingests_then_stops(client, data_dir):
    _upload(client, (SAMPLES / "safar-motor.pdf").read_bytes())
    service = IngestService(data_dir, FakeEmbedder())
    worker = IngestWorker(service, poll_interval=0.01)
    real = service.process_next

    def process_then_stop() -> bool:
        processed = real()
        if not processed:
            worker._request_stop()
        return processed

    with (
        patch.object(service, "process_next", side_effect=process_then_stop),
        patch("app.worker.worker.signal.signal"),
    ):
        worker.run()
    assert client.get("/documents").get_json()["data"][0]["status"] == "READY"


def test_worker_survives_database_errors(app, data_dir):
    service = IngestService(data_dir, FakeEmbedder())
    worker = IngestWorker(service, poll_interval=0.01)
    calls = {"n": 0}

    def flaky() -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db down")
        worker._request_stop()
        return False

    worker._stopping.wait = lambda timeout=None: None  # skip the 5 s back-off
    with patch.object(service, "process_next", side_effect=flaky), patch("app.worker.worker.signal.signal"):
        worker.run()
    assert calls["n"] == 2


def test_cli_and_worker_entry_point(app, monkeypatch, capsys):
    from app import cli
    from app.worker import __main__ as entry

    monkeypatch.setenv("DATABASE_URL", app.config["APP_CONFIG"].database_url)
    monkeypatch.setenv("DATA_DIR", app.config["APP_CONFIG"].data_dir)
    assert cli.main(["add-document", "--path", str(SAMPLES / "arogya-shield-health.pdf")]) == 0
    assert "\tPENDING\tqueued\tArogya Shield Health Policy" in capsys.readouterr().out
    assert cli.main(["add-document", "--path", str(SAMPLES / "arogya-shield-health.pdf")]) == 0
    assert "already in library" in capsys.readouterr().out
    with patch.object(entry, "db"), patch.object(entry, "IngestWorker") as fake_worker:
        entry.main()
    fake_worker.return_value.run.assert_called_once()
