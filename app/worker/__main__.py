"""Ingest worker: python -m app.worker (same image as the API, different command)."""

from pathlib import Path

from ..config import Config
from ..extensions import db
from ..logging_setup import setup_logging
from ..services.ingest_service import IngestService
from ..services.text_pipeline import Embedder
from .worker import IngestWorker


def main() -> None:
    config = Config.from_env()
    setup_logging(config.log_level)
    db.init_app(config.database_url)
    IngestWorker(IngestService(Path(config.data_dir), Embedder(config.models_dir))).run()


if __name__ == "__main__":
    main()
