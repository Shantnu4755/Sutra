import logging
import signal
import threading

from ..liveness import mark_alive
from ..services.ingest_service import IngestService

logger = logging.getLogger(__name__)


class IngestWorker:
    def __init__(self, service: IngestService, poll_interval: float = 1.0) -> None:
        self.service = service
        self.poll_interval = poll_interval
        self._stopping = threading.Event()

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        logger.info("worker_started")
        while not self._stopping.is_set():
            mark_alive()
            try:
                processed = self.service.process_next()
            except Exception:
                logger.exception("ingest_failed")  # database hiccup: back off, do not crash
                self._stopping.wait(5)
                continue
            if not processed:
                self._stopping.wait(self.poll_interval)
        logger.info("worker_stopped")

    def _request_stop(self, *_: object) -> None:
        logger.info("shutdown_requested")
        self._stopping.set()  # the current document finishes, then the loop exits
