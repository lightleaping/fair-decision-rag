"""Benchmark the long-lived submission service against the 30-second limit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.submission_service import SubmissionService


QUESTIONS = [
    "시장지배적 지위 남용행위의 판단 기준은 무엇인가?",
    "부당한 공동행위에 대한 시정명령의 내용은 무엇인가?",
    "과징금 산정 시 고려하는 요소는 무엇인가?",
    "재판매가격 유지행위가 금지되는 이유는 무엇인가?",
    "기업결합 제한 여부를 판단하는 기준은 무엇인가?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="outputs/results/submission_benchmark.json"
    )
    args = parser.parse_args()
    service = SubmissionService.from_environment()
    predictions = []
    for index, question in enumerate(QUESTIONS, 1):
        result = service.predict(f"eval_{index:04d}", question)
        chunk_ids = result["retrieved_chunk_ids"]
        assert len(chunk_ids) == 5
        assert len(set(chunk_ids)) == 5
        assert set(chunk_ids) <= service.valid_chunk_ids
        predictions.append(
            {
                "id": result["id"],
                "question": question,
                "elapsed_seconds": result["_elapsed_seconds"],
                "retrieved_chunk_ids": chunk_ids,
                "answer": result["answer"],
            }
        )
    report = {
        "startup_seconds": round(service.startup_seconds, 4),
        "max_prediction_seconds": max(
            row["elapsed_seconds"] for row in predictions
        ),
        "all_within_20_seconds": all(
            row["elapsed_seconds"] <= 20 for row in predictions
        ),
        "all_within_30_seconds": all(
            row["elapsed_seconds"] <= 30 for row in predictions
        ),
        "predictions": predictions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
