import logging

import redis
from flask import Flask

from ..services.answer_service import AnswerService
from ..services.llm_client import LLMClient
from ..services.text_pipeline import Embedder
from .documents import AskView, DocumentDetailView, DocumentListView
from .health import HealthView

logger = logging.getLogger(__name__)


def register_routes(app: Flask) -> None:
    app.add_url_rule("/health", view_func=HealthView.as_view("health"))
    # project-specific routes are added below this line
    config = app.config["APP_CONFIG"]
    cache = redis.Redis.from_url(config.redis_url, socket_timeout=0.2, socket_connect_timeout=0.2)
    llm = LLMClient(config.llm_base_url, config.llm_api_key, config.llm_model)
    if not llm.configured:
        logger.warning("llm_not_configured", extra={"code": "set LLM_API_KEY and LLM_MODEL to enable /ask"})
    answers = AnswerService(llm, cache, Embedder(config.models_dir))
    app.extensions["sutra"] = {"answers": answers, "cache": cache}

    app.add_url_rule("/documents", view_func=DocumentListView.as_view("documents", data_dir=config.data_dir))
    app.add_url_rule(
        "/documents/<uuid:document_id>",
        view_func=DocumentDetailView.as_view("document_detail", data_dir=config.data_dir),
    )
    app.add_url_rule("/ask", view_func=AskView.as_view("ask", service=answers))
