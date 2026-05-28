from src.preprocess import (
    load_chunks,
    normalize_chunks,
    check_chunk_integrity,
    get_section_statistics,
)
from src.bm25_retriever import BM25Retriever
from src.utils import save_json, get_valid_chunk_ids, validate_top5


DATA_PATH = "data/raw/chunks.jsonl"
OUTPUT_PATH = "outputs/results/bm25_results.json"


def run_bm25_pipeline():
    print("=== 1. 데이터 로딩 ===")
    chunks = load_chunks(DATA_PATH)
    chunks = normalize_chunks(chunks)

    print("=== 2. 데이터 무결성 검사 ===")
    integrity_report = check_chunk_integrity(chunks)
    for key, value in integrity_report.items():
        print(f"{key}: {value}")

    print("\n=== 3. section_type 통계 ===")
    section_stats = get_section_statistics(chunks)
    for section, count in section_stats.items():
        print(f"{section}: {count}")

    print("\n=== 4. BM25 검색기 생성 ===")
    retriever = BM25Retriever(chunks)
    valid_chunk_ids = get_valid_chunk_ids(chunks)

    test_queries = [
        "기업들이 가격을 같이 정한 사례가 있나요?",
        "과징금이 부과된 사건은?",
        "하도급 대금을 지급하지 않은 사례가 있나요?",
        "입찰담합 사건의 조치는 무엇인가요?",
        "어떤 법 조항을 위반했나요?",
    ]

    all_results = []

    print("\n=== 5. 테스트 질문 검색 ===")

    for query in test_queries:
        print("\n--------------------------------")
        print(f"질문: {query}")

        candidates = retriever.search(query, top_k=10)
        top5 = validate_top5(candidates, valid_chunk_ids)

        query_result = {
            "query": query,
            "top5_chunk_ids": [item["chunk_id"] for item in top5],
            "results": top5,
        }

        all_results.append(query_result)

        for rank, item in enumerate(top5, start=1):
            print(f"\n[{rank}] chunk_id: {item['chunk_id']}")
            print(f"score: {item['score']}")
            print(f"section_type: {item['section_type']}")
            print(f"preview: {item['preview']}")

    save_json(all_results, OUTPUT_PATH)
    print(f"\n검색 결과 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_bm25_pipeline()