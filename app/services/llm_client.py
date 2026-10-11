import logging
import time

import requests

from ..exceptions import ExternalServiceError

logger = logging.getLogger(__name__)


class LLMClient:
    """Any OpenAI-compatible chat endpoint (Groq's free tier works)."""

    def __init__(self, base_url: str, api_key: str, model: str, retries: int = 2) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model, self.retries = model, retries
        self.configured = bool(api_key and model)
        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {api_key}"})

    def complete(self, system: str, user: str) -> str:
        if not self.configured:
            raise ExternalServiceError("Answer service is not configured (LLM_API_KEY and LLM_MODEL)")
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        for attempt in range(self.retries + 1):
            try:
                resp = self.http.post(self.url, json=body, timeout=(3, 30))
                if resp.status_code in (429, 500, 502, 503) and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except (requests.RequestException, KeyError, ValueError) as exc:
                if attempt >= self.retries:
                    logger.exception("llm_failed")
                    raise ExternalServiceError("Answer service is unavailable, try again shortly") from exc
                time.sleep(2**attempt)
        raise ExternalServiceError("Answer service is unavailable, try again shortly")
