# src/retrieval/day10_bm25_pipeline.py

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.day10_a_runner import run_day10_a


CHUNK_PATH = "data/chunks.jsonl"
OUTPUT_PATH = "outputs/results/day10_bm25_top5.json"


TEST_QUERIES = [
    "과징금은 어떻게 산정되었나요?",
    "관련매출액은 어떻게 계산되었나요?",
    "어떤 행위가 문제 되었나요?",
    "위법하다고 판단한 근거는 무엇인가요?",
    "적용된 법 조항은 무엇인가요?",
    "시정명령의 내용은 무엇인가요?",
    "고발 조치가 있었나요?",
    "입찰담합의 행위 패턴은 무엇인가요?",
    "부당한 공동행위가 인정된 이유는 무엇인가요?",
    "이 의결서의 핵심 내용을 요약해 주세요.",
]


def load_valid_chunk_ids(chunk_jsonl_path: str) -> Set[str]:
    valid_chunk_ids = set()

    with open(chunk_jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            chunk_id = item.get("chunk_id")

            if not chunk_id:
                raise ValueError(f"{line_idx}번째 줄에 chunk_id가 없습니다.")

            valid_chunk_ids.add(chunk_id)

    return valid_chunk_ids


def run_day10_bm25_pipeline(
    chunk_path: str = CHUNK_PATH,
    output_path: str = OUTPUT_PATH,
    queries: List[str] = TEST_QUERIES,
    candidate_k: int = 50,
    max_seconds: float = 30.0,
) -> Dict[str, Any]:
    start = time.perf_counter()

    print("=== Day 10 BM25 Pipeline 실행 ===")
    print(f"chunk_path: {chunk_path}")

    retriever = BM25Retriever.from_jsonl(chunk_path)
    valid_chunk_ids = load_valid_chunk_ids(chunk_path)

    all_results = []

    for idx, query in enumerate(queries, start=1):
        query_start = time.perf_counter()

        bm25_candidates = retriever.search(
            query=query,
            top_n=candidate_k,
        )

        result = run_day10_a(
            query=query,
            search_results=bm25_candidates,
            valid_chunk_ids=valid_chunk_ids,
        )

        elapsed = time.perf_counter() - query_start

        result["retriever"] = "bm25"
        result["candidate_k"] = candidate_k
        result["elapsed_seconds"] = round(elapsed, 4)
        result["within_30_seconds"] = elapsed <= max_seconds

        top5_chunk_ids = result["top5_chunk_ids"]

        assert len(top5_chunk_ids) == 5
        assert len(set(top5_chunk_ids)) == 5
        assert elapsed <= max_seconds

        all_results.append(result)

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(queries)}] query: {query}")
        print(f"query_type: {result.get('query_type')}")
        print(f"matched_keywords: {result.get('matched_keywords')}")
        print(f"elapsed_seconds: {elapsed:.4f}")
        print(f"top5_chunk_ids: {top5_chunk_ids}")

    total_elapsed = time.perf_counter() - start

    output = {
        "day": 10,
        "task": "BM25 baseline with query classification, section boost, and Top-5 validation",
        "chunk_path": chunk_path,
        "num_queries": len(queries),
        "candidate_k": candidate_k,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "rules": {
            "exactly_5_chunk_ids": True,
            "no_duplicate_chunk_ids": True,
            "valid_existing_chunk_ids_only": True,
            "max_response_time_seconds": max_seconds,
            "external_api": False,
        },
        "results": all_results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Day 10 완료 ===")
    print(f"저장 경로: {output_path}")
    print(f"전체 실행 시간: {total_elapsed:.4f}초")

    return output


if __name__ == "__main__":
    run_day10_bm25_pipeline()