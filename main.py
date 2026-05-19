# main.py

import argparse
import json
import sys
from pathlib import Path

from src.retrieval.day10_bm25_pipeline import run_bm25_day10_pipeline
from src.data.build_chunks_jsonl import build_chunks_jsonl
from src.retrieval.day11_dense_pipeline import run_dense_day11_pipeline


def ensure_chunks_jsonl(
    raw_dir: str = "data/raw",
    chunk_jsonl_path: str = "data/chunks.jsonl",
) -> None:
    """
    data/chunks.jsonl이 없으면 data/raw의 *_hybrid.json 파일을 이용해 생성한다.
    """

    chunk_path = Path(chunk_jsonl_path)

    if chunk_path.exists():
        return

    print(f"[INFO] {chunk_jsonl_path} 파일이 없어 새로 생성합니다.")

    summary = build_chunks_jsonl(
        input_dir=raw_dir,
        output_path=chunk_jsonl_path,
    )

    print("[INFO] chunks.jsonl 생성 완료")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def print_top5(result: dict) -> None:
    """
    최종 Top-5 결과를 터미널에 보기 좋게 출력한다.
    """

    print("\n" + "=" * 80)
    print("검색 완료")
    print("=" * 80)

    print(f"질문: {result['query']}")
    print(f"질문 유형: {result['query_type']}")
    print(f"매칭 키워드: {result['matched_keywords']}")
    print(f"검색 방식: {result['retriever']}")
    print(f"응답 시간: {result['elapsed_seconds']}초")
    print(f"30초 이내 여부: {result['within_30_seconds']}")

    print("\nTop-5 chunk_id")
    for idx, chunk_id in enumerate(result["top5_chunk_ids"], start=1):
        print(f"{idx}. {chunk_id}")

    print("\nTop-5 상세 근거")
    for item in result["top5_results"]:
        text = item.get("text", "")
        preview = text[:120].replace("\n", " ")

        print("-" * 80)
        print(f"rank: {item['rank']}")
        print(f"chunk_id: {item['chunk_id']}")
        print(f"section_type: {item['section_type']}")
        print(f"score: {item['score']}")
        print(f"section_boost: {item['section_boost']}")
        print(f"boosted_score: {item['boosted_score']}")
        print(f"text: {preview}")


def main():
    parser = argparse.ArgumentParser(
        description="AI 의결서 Day 10 BM25 Top-5 검색 실행기"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="검색할 사용자 질문",
    )

    parser.add_argument(
        "--chunks",
        type=str,
        default="data/chunks.jsonl",
        help="chunk JSONL 파일 경로",
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="*_hybrid.json 파일들이 있는 폴더",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results/day10_bm25_top5.json",
        help="검색 결과 저장 경로",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="BM25 후보 개수",
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "dense"],
        help="검색 방식 선택: bm25 또는 dense",
    )

    args = parser.parse_args()

    try:
        ensure_chunks_jsonl(
            raw_dir=args.raw_dir,
            chunk_jsonl_path=args.chunks,
        )

        if args.retriever == "bm25":
            result = run_bm25_day10_pipeline(
                query=args.query,
                chunk_jsonl_path=args.chunks,
                output_path=args.output,
                bm25_top_n=args.top_n,
            )
        else:
            result = run_dense_day11_pipeline(
                query=args.query,
                chunk_jsonl_path=args.chunks,
                output_path=args.output,
                dense_top_n=args.top_n,
            )

        chunk_ids = result["top5_chunk_ids"]

        assert len(chunk_ids) == 5
        assert len(set(chunk_ids)) == 5

        print_top5(result)

        print(f"\n[저장 완료] {args.output}")

    except Exception as e:
        print("\n[ERROR] 실행 실패")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()