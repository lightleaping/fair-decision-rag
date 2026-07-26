"""Evaluate retrieval and grounded answers against a JSONL QA set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.retrieval.answer_generator import generate_extractive_answer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.chunk_validator import load_valid_chunk_ids
from src.retrieval.evaluator import evaluate_rows, load_jsonl
from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="data/processed/silver_qa_eval.jsonl")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--mode", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", default="outputs/results/pipeline_evaluation.json"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    bm25 = BM25Retriever.from_jsonl(args.chunks)
    dense = None
    if args.mode == "hybrid":
        from src.retrieval.dense_retriever import DenseRetriever

        dense = DenseRetriever(offline=True)
        dense.load_index()
    retriever = HybridRetriever(
        bm25, dense, valid_chunk_ids=load_valid_chunk_ids(args.chunks)
    )
    qa_rows = load_jsonl(args.qa)
    if args.limit:
        qa_rows = qa_rows[: args.limit]

    evaluated_rows = []
    for index, qa in enumerate(qa_rows, 1):
        results = retriever.search(qa["query"])
        answer = generate_extractive_answer(qa["query"], results)
        evaluated_rows.append(
            {
                **qa,
                "top5_chunk_ids": [item["chunk_id"] for item in results],
                "answer": answer["answer"],
            }
        )
        if index % 50 == 0:
            print(f"[{index}/{len(qa_rows)}] 평가 완료")

    report = evaluate_rows(evaluated_rows)
    report.update(
        {
            "mode": args.mode,
            "qa_path": args.qa,
            "qa_quality": sorted(
                {row.get("quality", "unknown") for row in qa_rows}
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "reports"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
