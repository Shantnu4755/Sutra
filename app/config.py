import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised at startup when required configuration is missing."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Config:
    app_name: str
    database_url: str
    log_level: str
    debug: bool
    redis_url: str = "redis://localhost:6379/1"
    data_dir: str = "/data"  # uploaded PDFs wait here until ingested
    models_dir: str = "/models"  # embedding model cache, shared by API and worker
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""  # without it /ask answers 502 "not configured"; the rest still works
    llm_model: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            app_name=os.getenv("APP_NAME", "app"),
            database_url=_require("DATABASE_URL"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/1"),
            data_dir=os.getenv("DATA_DIR", "/data"),
            models_dir=os.getenv("MODELS_DIR", "/models"),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
        )
