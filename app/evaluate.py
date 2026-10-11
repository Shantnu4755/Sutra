"""Evaluation for the README (PDF 3.5): python -m app.evaluate [--questions eval/questions.json]

Asks every question twice (cache miss, then cache hit) and writes a JSON summary to stdout:
accuracy on answerable questions, the "correctly said not found" rate, and median answer times.
A question counts as correct when the answer contains every expected keyword and cites the
expected document; a not-found question is correct when the service says it could not find it.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from . import create_app
from .services.answer_service import AnswerService


def evaluate(service: AnswerService, questions: list[dict]) -> dict:
    results = []
    for q in questions:
        started = time.perf_counter()
        out = service.ask(q["question"])
        miss_seconds = time.perf_counter() - started
        started = time.perf_counter()
        service.ask(q["question"])
        hit_seconds = time.perf_counter() - started

        expect = q["expect"]
        if expect == "not_found":
            correct = not out.found
        else:
            answer = " ".join(out.answer.lower().split())
            has_keywords = all(k.lower() in answer for k in expect["keywords"])
            cites_document = any(c.title == expect["document"] for c in out.citations)
            correct = out.found and has_keywords and cites_document
        results.append(
            {
                "id": q["id"],
                "kind": "not_found" if expect == "not_found" else "answer",
                "correct": correct,
                "answer": out.answer,
                "miss_seconds": round(miss_seconds, 3),
                "hit_seconds": round(hit_seconds, 4),
            }
        )

    def rate(kind: str) -> float | None:
        subset = [r for r in results if r["kind"] == kind]
        return round(sum(r["correct"] for r in subset) / len(subset), 3) if subset else None

    return {
        "questions": len(results),
        "accuracy": rate("answer"),
        "not_found_rate": rate("not_found"),
        "median_miss_seconds": round(statistics.median(r["miss_seconds"] for r in results), 2),
        "median_hit_ms": round(statistics.median(r["hit_seconds"] for r in results) * 1000, 1),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluate")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    args = parser.parse_args(argv)
    app = create_app()
    questions = json.loads(args.questions.read_text())
    with app.app_context():
        summary = evaluate(app.extensions["sutra"]["answers"], questions)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
