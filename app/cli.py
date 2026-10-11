"""python -m app.cli add-document --path policy.pdf [--title "Health policy"]

Queues a PDF from disk (seeding the demo library). The ingest worker picks it up.
"""

import argparse
import sys
from pathlib import Path

from .config import Config
from .exceptions import AppError
from .extensions import db
from .logging_setup import setup_logging
from .services.document_service import DocumentService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add-document", help="queue a PDF from disk for ingestion")
    add.add_argument("--path", type=Path, required=True)
    add.add_argument("--title")
    args = parser.parse_args(argv)

    config = Config.from_env()
    setup_logging("WARNING")
    db.init_app(config.database_url)
    try:
        doc, created = DocumentService(Path(config.data_dir)).create_from_path(args.path, args.title)
    except AppError as exc:
        sys.stderr.write(f"{exc.message}\n")
        return 1
    sys.stdout.write(
        f"{doc.id}\t{doc.status.value}\t{'queued' if created else 'already in library'}\t{doc.title}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
