# main_day12_hybrid.py

import json
import time
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.chunk_validator import load_valid_chunk_ids
from src.retrieval.hybrid_retriever import HybridRetriever


CHUNK_PATH = "data/chunks.jsonl"
OUTPUT_PATH = "outputs/results/day12_hybrid_top5.json"


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


def main():
    print("=== Day 12 Hybrid Retriever 실행 ===")

    total_start = time.perf_counter()

    print("\n=== 1. BM25 로딩 ===")
    bm25_retriever = BM25Retriever.from_jsonl(CHUNK_PATH)

    print("\n=== 2. valid_chunk_ids 로딩 ===")
    valid_chunk_ids = load_valid_chunk_ids(CHUNK_PATH)

    print("\n=== 3. HybridRetriever 생성 ===")
    print("현재 Dense Retriever는 미연결 상태이므로 BM25 fallback으로 실행합니다.")

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=None,
        valid_chunk_ids=valid_chunk_ids,
    )

    all_results = []

    print("\n=== 4. 테스트 질의 실행 ===")

    for idx, query in enumerate(TEST_QUERIES, start=1):
        query_start = time.perf_counter()

        top5_results = hybrid_retriever.search(
            query=query,
            top_k=5,
            candidate_k=50,
            bm25_weight=1.0,
            dense_weight=0.0,
        )

        elapsed = time.perf_counter() - query_start

        top5_chunk_ids = [
            item["chunk_id"]
            for item in top5_results
        ]

        assert len(top5_chunk_ids) == 5
        assert len(set(top5_chunk_ids)) == 5
        assert all(chunk_id in valid_chunk_ids for chunk_id in top5_chunk_ids)
        assert elapsed <= 30

        result = {
            "query": query,
            "retriever": "hybrid_bm25_fallback",
            "dense_available": False,
            "candidate_k": 50,
            "elapsed_seconds": round(elapsed, 4),
            "within_30_seconds": elapsed <= 30,
            "top5_chunk_ids": top5_chunk_ids,
            "top5_results": top5_results,
        }

        all_results.append(result)

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(TEST_QUERIES)}] query: {query}")
        print(f"elapsed_seconds: {elapsed:.4f}")
        print(f"top5_chunk_ids: {top5_chunk_ids}")

    total_elapsed = time.perf_counter() - total_start

    output = {
        "day": 12,
        "task": "Hybrid retriever interface with BM25 fallback",
        "chunk_path": CHUNK_PATH,
        "dense_available": False,
        "num_queries": len(TEST_QUERIES),
        "candidate_k": 50,
        "total_elapsed_seconds": round(total_elapsed, 4),
        "rules": {
            "exactly_5_chunk_ids": True,
            "no_duplicate_chunk_ids": True,
            "valid_existing_chunk_ids_only": True,
            "max_response_time_seconds": 30.0,
            "external_api": False,
            "bm25_dense_fusion_ready": True,
        },
        "results": all_results,
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Day 12 완료 ===")
    print(f"저장 경로: {OUTPUT_PATH}")
    print(f"전체 실행 시간: {total_elapsed:.4f}초")


if __name__ == "__main__":
    main()