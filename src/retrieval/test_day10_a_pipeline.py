# src/retrieval/test_day10_a_pipeline.py

from query_classifier import QueryClassifier
from section_boost import SectionBooster
from topk_selector import TopKSelector


def run_day10_a_pipeline(query: str, search_results: list[dict]) -> dict:
    """
    Day 10-A 통합 파이프라인.

    입력:
    - 사용자 질문
    - 검색 결과 후보 리스트

    출력:
    - query_type
    - matched_keywords
    - top5_results
    - top5_chunk_ids
    """

    classifier = QueryClassifier()
    booster = SectionBooster()
    selector = TopKSelector(k=5)

    # 1. 질문 유형 분류
    classified = classifier.classify(query)

    # 2. section boost 적용
    boosted_results = booster.apply_boost(
        results=search_results,
        section_priority=classified.section_priority,
        score_key="score",
        section_key="section_type",
        boosted_score_key="boosted_score",
    )

    # 3. 정확히 Top-5 선택
    top5_results = selector.select(
        results=boosted_results,
        score_key="boosted_score",
        chunk_id_key="chunk_id",
    )

    top5_chunk_ids = [item["chunk_id"] for item in top5_results]

    # 4. 최종 검증
    assert len(top5_chunk_ids) == 5
    assert len(set(top5_chunk_ids)) == 5

    return {
        "query": query,
        "query_type": classified.query_type,
        "matched_keywords": classified.matched_keywords,
        "section_priority": classified.section_priority,
        "top5_chunk_ids": top5_chunk_ids,
        "top5_results": top5_results,
    }


def main():
    sample_search_results = [
        {
            "chunk_id": "case_001_chunk_001",
            "section_type": "fact",
            "score": 8.4,
            "text": "피심인은 거래상대방에게 특정 조건을 요구하였다."
        },
        {
            "chunk_id": "case_001_chunk_002",
            "section_type": "legal_reasoning",
            "score": 7.9,
            "text": "이 행위는 거래상 지위를 이용한 불이익 제공행위로 판단된다."
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "score": 6.8,
            "text": "피심인에게 과징금 1억 원을 부과한다."
        },
        {
            "chunk_id": "case_001_chunk_004",
            "section_type": "law_article",
            "score": 7.2,
            "text": "이 사건에는 공정거래법 제23조 제1항이 적용된다."
        },
        {
            "chunk_id": "case_001_chunk_005",
            "section_type": "order",
            "score": 6.9,
            "text": "피심인은 향후 동일하거나 유사한 행위를 반복하여서는 아니 된다."
        },
        {
            "chunk_id": "case_001_chunk_006",
            "section_type": "summary",
            "score": 7.0,
            "text": "이 사건은 거래상 지위를 이용한 불공정거래행위에 관한 건이다."
        },
        {
            "chunk_id": "case_001_chunk_007",
            "section_type": "conclusion",
            "score": 6.7,
            "text": "위원회는 피심인의 행위에 대해 시정명령과 과징금 부과를 결정하였다."
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "score": 6.5,
            "text": "중복 chunk_id 테스트용 데이터이다."
        },
    ]

    test_queries = [
        "이 사건 과징금은 얼마야?",
        "왜 위법하다고 판단했어?",
        "어떤 행위가 문제였어?",
        "적용 법조항 알려줘",
        "최종 시정명령이 뭐야?",
        "전체 내용 요약해줘",
    ]

    for query in test_queries:
        result = run_day10_a_pipeline(
            query=query,
            search_results=sample_search_results,
        )

        print("\n" + "=" * 80)
        print("질문:", result["query"])
        print("질문 유형:", result["query_type"])
        print("매칭 키워드:", result["matched_keywords"])
        print("Top-5 chunk_id:", result["top5_chunk_ids"])

        print("\nTop-5 상세 결과")
        for rank, item in enumerate(result["top5_results"], start=1):
            print(
                f"{rank}. "
                f"chunk_id={item['chunk_id']} | "
                f"section={item['section_type']} | "
                f"base_score={item['score']} | "
                f"boost={item['section_boost']} | "
                f"boosted_score={item['boosted_score']:.3f}"
            )


if __name__ == "__main__":
    main()