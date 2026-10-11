# CLAUDE.md: Sutra

Policy document Q&A: hybrid RAG with exact clause citations and an honest "not found", on CPU only. Part of a six-project portfolio platform that runs on one free Oracle ARM VM.

- **Source of truth:** the PDF `~/Downloads/Portfolio Projects Complete Build & Free Deployment Plan.pdf`.
- **Progress and deviations:** `../PROGRESS.md`. Update it after every step. Any change to the PDF's design goes under "Deviations from the plan" with the reason; never change the design silently.

## Rules (PDF "Coding rules")
1. Class-based views only (`BaseView`, a Flask `MethodView`) with **only `get()` and `post()`**. Updates and deletes are a POST to an action endpoint (for example `POST /claims/<id>/approve`). `tests/test_view_rules.py` enforces it.
2. Layers: **View** (HTTP only) → **Service** (business logic) → **Repository** (DB access) → **Model**. A view never writes SQL or opens sessions; a repository never imports Flask.
3. Errors: raise `AppError` subclasses from `app/exceptions.py`. One global handler returns `{"error": {"code", "message", "details", "request_id"}}`. No bare `except:`, no `print`, no swallowed errors (log with `logger.exception` when a loop must continue).
4. Every request is validated with Pydantic (`BaseView.parse_body` / `parse_query`) before it reaches a service.
5. Config from environment variables only (`app/config.py`, fails fast at boot). `.env.example` is committed, `.env` never is.
6. Structured JSON logs with `request_id` on every line. Allowed extra fields: `code`, `error_message`, `job_id`, `service`.
7. pytest for every service class, against real PostgreSQL (never SQLite). External calls (LLM, Telegram, Docker, HTTP) are mocked. CI fails under 80% coverage.

Also: Python 3.12, type hints everywhere, `ruff check .` clean, small commits prefixed `feat:` `fix:` `test:` `docs:` `chore:`. Push after every verified change (the local folder was once deleted by an outside process; GitHub is the backup).

## Layout (PDF 2.1)
`app/{__init__,config,extensions,exceptions,error_handlers,logging_setup}.py`, `app/models`, `app/schemas`, `app/repositories`, `app/services`, `app/views` (`base.py`, `health.py`, `__init__.py` with `register_routes`), `migrations/` (Alembic), `tests/`, `wsgi.py`, `gunicorn.conf.py`, `Dockerfile`, `ui/` (React UI, PDF 3A).

## Commands
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock -r requirements-dev.txt
# local test databases: docker container "portfolio-test-pg" on port 55432 (password "test")
export TEST_DATABASE_URL=postgresql+psycopg://postgres:test@localhost:55432/sutra_test
ruff check . && pytest --cov=app --cov-report=term-missing
DATABASE_URL=... alembic revision --autogenerate -m "describe change" && alembic upgrade head
docker build -t sutra .
```
Dependencies: runtime packages go in `requirements.txt`, test tools in `requirements-dev.txt`. `requirements.lock` is the frozen runtime set (`pip freeze` of a runtime-only install, or of the built image), so pytest never ships in the image.

## Sutra specifics (PDF 3.5)
- **Extra deps:** pypdf, fastembed (ONNX, `BAAI/bge-small-en-v1.5`, 384 dims, cache in `MODELS_DIR`), pgvector, redis, requests. No PyTorch, no GPU.
- **Tables:** `documents` (id, title, sha256 unique, status PENDING|READY|FAILED, pages, error, created_at); `chunks` (id, document_id, page, text, embedding vector(384), tsv generated). HNSW and GIN indexes from a hand-written SQL migration.
- **Endpoints:** `POST /documents` (PDF ≤ 10 MB; duplicate returns the existing record; a FAILED one is re-queued), `GET /documents`, `GET /documents/<id>`, `POST /ask` → `{answer, found, citations[], cached}`.
- **Ingest worker:** `python -m app.worker`, row-locked `SKIP LOCKED` on `PENDING`; pypdf per page; `Chunker(800, 150)`; embed batches of 32; `READY` or `FAILED` with the error.
- **Retrieval:** vector top 20 + keyword top 20 fused with RRF (k = 60), top 5. Best distance above 0.55 → "not found" without calling the LLM.
- **LLM:** OpenAI-compatible (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`), temperature 0, retries on 429/5xx, `NOT_FOUND` sentinel. Redis answer cache 24 h keyed with a corpus signature.
- **Eval:** `eval/policies/*.txt` → `scripts/make_sample_pdfs.py` → `eval/samples/*.pdf`; `eval/questions.json` (30); `python -m app.evaluate`. Seed: `python -m app.cli add-document --path X.pdf`.
- **Tests to write:** chunker never loops and respects overlap; no relevant text → "not found" without calling the LLM; LLM 429 retried; Redis down still answers; duplicate upload does not create a second document.
- **UI accent:** indigo (hue 275).
