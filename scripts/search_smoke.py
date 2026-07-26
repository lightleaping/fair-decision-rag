"""Build a BM25 index and run real-data Track 2 smoke queries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.query_classifier import QueryClassifier
from src.retrieval.section_boost import SectionBooster


DEFAULT_QUERIES = [
    "피심인의 위반 행위는 무엇인가요?",
    "공정거래위원회가 내린 시정명령은 무엇인가요?",
    "과징금은 얼마이고 어떻게 산정했나요?",
    "어떤 법률 조항을 위반했다고 판단했나요?",
    "이 사건의 핵심 내용을 요약해 주세요.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument(
        "--output", default="outputs/results/official_data_smoke.json"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    retriever = BM25Retriever.from_jsonl(args.chunks)
    classifier = QueryClassifier()
    booster = SectionBooster()
    outputs = []
    for query in DEFAULT_QUERIES:
        query_started = time.perf_counter()
        classification = classifier.classify(query)
        candidates = retriever.search(query, top_n=50)
        ranked = booster.apply_boost(candidates, classification.section_priority)
        top5 = ranked[:5]
        chunk_ids = [item["chunk_id"] for item in top5]
        assert len(chunk_ids) == 5 and len(set(chunk_ids)) == 5
        outputs.append(
            {
                "query": query,
                "query_type": classification.query_type,
                "matched_keywords": classification.matched_keywords,
                "elapsed_seconds": round(time.perf_counter() - query_started, 4),
                "top5_chunk_ids": chunk_ids,
                "top5_results": top5,
            }
        )
    artifact = {
        "corpus": args.chunks,
        "index": "bm25_korean_bigram",
        "num_chunks": len(retriever.chunks),
        "num_queries": len(outputs),
        "total_elapsed_seconds": round(time.perf_counter() - started, 4),
        "results": outputs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**artifact, "results": "omitted"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
