# main_hybrid.py

from src.preprocess import (
    load_chunks,
    normalize_chunks,
    check_chunk_integrity,
    get_section_statistics,
)
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.utils import save_json


DATA_PATH = "data/raw"
OUTPUT_PATH = "outputs/results/hybrid_results_day9.json"


TEST_QUERIES = [
    "기업들이 가격을 같이 정한 사례가 있나요?",
    "과징금은 얼마인가요?",
    "하도급 대금을 지급하지 않은 사례가 있나요?",
    "입찰담합 사건의 조치는 무엇인가요?",
    "어떤 법 조항을 위반했나요?",
    "시정명령이 내려졌나요?",
    "고발 조치가 있었나요?",
    "부당한 공동행위가 인정된 이유는 무엇인가요?",
    "관련매출액은 어떻게 산정되었나요?",
    "이 사건을 요약해줘.",
]


def print_integrity_report(chunks: list[dict]):
    print("=== 데이터 무결성 검사 ===")
    report = check_chunk_integrity(chunks)
    for key, value in report.items():
        print(f"{key}: {value}")


def print_section_statistics(chunks: list[dict]):
    print("\n=== section_type 통계 ===")
    stats = get_section_statistics(chunks)
    for section_type, count in stats.items():
        print(f"{section_type}: {count}")


def print_hybrid_result(result: dict):
    print("\n--------------------------------")
    print(f"질문: {result['query']}")
    print(f"질문 유형: {result['query_type']}")
    print(f"우선 section: {result['priority_sections']}")
    print(f"BM25 weight: {result['bm25_weight']}")
    print(f"Dense weight: {result['dense_weight']}")
    print(f"Dense available: {result['dense_available']}")
    print(f"Top-5 chunk_id: {result['top5_chunk_ids']}")

    for rank, item in enumerate(result["results"], start=1):
        print(f"\n[{rank}]")
        print(f"chunk_id: {item.get('chunk_id')}")
        print(f"section_type: {item.get('section_type')}")
        print(f"hybrid_score: {item.get('hybrid_score')}")
        print(f"bm25_score_norm: {item.get('bm25_score_norm')}")
        print(f"dense_score_norm: {item.get('dense_score_norm')}")
        print(f"section_boosted: {item.get('section_boosted')}")
        print(f"preview: {item.get('preview')}")


def run_day9_hybrid_pipeline():
    print("=== Day 9 Section-aware Hybrid Retrieval 실행 ===")

    print("\n=== 1. 데이터 로딩 ===")
    chunks = load_chunks(DATA_PATH)
    chunks = normalize_chunks(chunks)

    print_integrity_report(chunks)
    print_section_statistics(chunks)

    print("\n=== 2. BM25 Retriever 생성 ===")
    bm25_retriever = BM25Retriever(chunks)

    print("\n=== 3. Hybrid Retriever 생성 ===")
    hybrid_retriever = HybridRetriever(
        chunks=chunks,
        bm25_retriever=bm25_retriever,
        dense_retriever=None,
    )

    print("\n=== 4. 테스트 질문 실행 ===")

    all_results = []

    for query in TEST_QUERIES:
        result = hybrid_retriever.search(
            query=query,
            top_k=5,
            candidate_k=10,
        )

        print_hybrid_result(result)
        all_results.append(result)

    save_json(all_results, OUTPUT_PATH)

    print("\n=== Day 9 완료 ===")
    print(f"결과 저장 경로: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_day9_hybrid_pipeline()