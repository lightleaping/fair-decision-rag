"""Sequentially benchmark the official HTTP API with distinct questions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def load_questions(path: Path, count: int) -> list[str]:
    questions = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question") or row.get("query") or "").strip()
            if question:
                questions.append(question)
            if len(questions) == count:
                break
    if len(questions) != count:
        raise ValueError(f"Expected {count} questions, found {len(questions)}.")
    return questions


def request_prediction(url: str, request_id: str, question: str) -> tuple[dict, float]:
    payload = json.dumps(
        {"id": request_id, "question": question}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument(
        "--questions", default="data/processed/silver_qa_eval.jsonl"
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()

    timings = []
    failures = []
    for index, question in enumerate(
        load_questions(Path(args.questions), args.count), 1
    ):
        request_id = f"stability_{index:04d}"
        try:
            result, elapsed = request_prediction(args.url, request_id, question)
            ids = result.get("retrieved_chunk_ids", [])
            valid = (
                list(result) == ["id", "retrieved_chunk_ids", "answer"]
                and result.get("id") == request_id
                and len(ids) == 5
                and len(set(ids)) == 5
                and isinstance(result.get("answer"), str)
                and bool(result["answer"].strip())
                and elapsed <= 30
            )
            if not valid:
                failures.append({"id": request_id, "elapsed_seconds": elapsed})
            timings.append(elapsed)
        except Exception as error:  # noqa: BLE001 - benchmark records all failures
            failures.append({"id": request_id, "error": str(error)})

    report = {
        "requests": args.count,
        "passed": args.count - len(failures),
        "failed": len(failures),
        "all_passed": not failures,
        "mean_seconds": round(statistics.mean(timings), 4) if timings else None,
        "p95_seconds": (
            round(sorted(timings)[int(0.95 * (len(timings) - 1))], 4)
            if timings
            else None
        ),
        "max_seconds": round(max(timings), 4) if timings else None,
        "failures": failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
