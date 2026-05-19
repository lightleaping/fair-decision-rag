# src/retrieval/day10_batch_test.py

import json
from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.day10_bm25_pipeline import (
    run_bm25_day10_pipeline,
    make_sample_chunk_jsonl,
)


DEFAULT_TEST_QUERIES = [
    "이 사건 과징금은 얼마야?",
    "왜 위법하다고 판단했어?",
    "어떤 행위가 문제였어?",
    "적용 법조항 알려줘",
    "최종 시정명령이 뭐야?",
    "전체 내용 요약해줘",
]


def run_batch_test(
    queries: List[str],
    chunk_jsonl_path: str = "data/chunks.jsonl",
    output_path: str = "outputs/results/day10_bm25_batch_results.json",
    bm25_top_n: int = 20,
) -> Dict[str, Any]:
    """
    여러 질문에 대해 Day 10 BM25 파이프라인을 반복 실행한다.
    """

    if not Path(chunk_jsonl_path).exists():
        print(f"{chunk_jsonl_path} 파일이 없어 sample chunk 파일을 생성합니다.")
        make_sample_chunk_jsonl(chunk_jsonl_path)

    all_results = []
    success_count = 0
    fail_count = 0

    for idx, query in enumerate(queries, start=1):
        print(f"\n[{idx}/{len(queries)}] 실행 중: {query}")

        try:
            result = run_bm25_day10_pipeline(
                query=query,
                chunk_jsonl_path=chunk_jsonl_path,
                output_path=f"outputs/results/day10_bm25_single_{idx}.json",
                bm25_top_n=bm25_top_n,
            )

            top5_chunk_ids = result["top5_chunk_ids"]

            assert len(top5_chunk_ids) == 5
            assert len(set(top5_chunk_ids)) == 5
            assert result["within_30_seconds"] is True

            all_results.append(
                {
                    "query": query,
                    "status": "success",
                    "query_type": result["query_type"],
                    "matched_keywords": result["matched_keywords"],
                    "top5_chunk_ids": top5_chunk_ids,
                    "elapsed_seconds": result["elapsed_seconds"],
                    "within_30_seconds": result["within_30_seconds"],
                    "top5_results": result["top5_results"],
                }
            )

            success_count += 1

        except Exception as e:
            all_results.append(
                {
                    "query": query,
                    "status": "fail",
                    "error": str(e),
                }
            )

            fail_count += 1

    summary = {
        "total_queries": len(queries),
        "success_count": success_count,
        "fail_count": fail_count,
        "all_success": fail_count == 0,
        "results": all_results,
    }

    save_batch_result(summary, output_path)

    return summary


def save_batch_result(result: Dict[str, Any], output_path: str) -> None:
    """
    배치 테스트 결과를 JSON으로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    summary = run_batch_test(
        queries=DEFAULT_TEST_QUERIES,
        chunk_jsonl_path="data/chunks.jsonl",
        output_path="outputs/results/day10_bm25_batch_results.json",
        bm25_top_n=20,
    )

    print("\n" + "=" * 80)
    print("Day 10 BM25 Batch Test Summary")
    print("=" * 80)
    print(f"total_queries: {summary['total_queries']}")
    print(f"success_count: {summary['success_count']}")
    print(f"fail_count: {summary['fail_count']}")
    print(f"all_success: {summary['all_success']}")
    print("\n저장 완료: outputs/results/day10_bm25_batch_results.json")