import os
from pathlib import Path

import pytest
from sqlalchemy import text

from app import create_app
from app.config import Config
from app.extensions import Base, db

SAMPLES = Path(__file__).resolve().parent.parent / "eval" / "samples"


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    root = tmp_path_factory.mktemp("sutra")
    config = Config(
        app_name="test",
        database_url=os.environ["TEST_DATABASE_URL"],
        log_level="WARNING",
        debug=False,
        redis_url=os.environ["TEST_REDIS_URL"],
        data_dir=str(root / "data"),
        models_dir=str(root / "models"),
        llm_api_key="test-key",
        llm_model="test-model",
    )
    application = create_app(config)
    with db.engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(db.engine)
    Base.metadata.create_all(db.engine)
    yield application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_tables(app):
    yield
    with db.engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture(autouse=True)
def clean_cache(app):
    yield
    app.extensions["sutra"]["cache"].flushdb()


@pytest.fixture
def data_dir(app) -> Path:
    return Path(app.config["APP_CONFIG"].data_dir)
