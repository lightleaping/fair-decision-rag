# src/retrieval/test_query_boost.py

from query_classifier import QueryClassifier
from section_boost import SectionBooster


def main():
    classifier = QueryClassifier()
    booster = SectionBooster()

    sample_results = [
        {
            "chunk_id": "case_001_chunk_001",
            "section_type": "fact",
            "score": 8.2,
            "text": "피심인은 거래상 지위를 이용하여 거래상대방에게 불리한 조건을 요구하였다."
        },
        {
            "chunk_id": "case_001_chunk_002",
            "section_type": "legal_reasoning",
            "score": 7.8,
            "text": "이 행위는 거래상 지위 남용행위에 해당한다고 판단된다."
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "score": 6.9,
            "text": "피심인에게 과징금 1억 원을 부과한다."
        },
        {
            "chunk_id": "case_001_chunk_004",
            "section_type": "law_article",
            "score": 7.1,
            "text": "공정거래법 제23조 제1항이 적용된다."
        },
        {
            "chunk_id": "case_001_chunk_005",
            "section_type": "order",
            "score": 6.7,
            "text": "피심인은 향후 동일한 행위를 반복하여서는 아니 된다."
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
        print("\n" + "=" * 80)
        print("질문:", query)

        classified = classifier.classify(query)

        print("질문 유형:", classified.query_type)
        print("매칭 키워드:", classified.matched_keywords)
        print("section priority:", classified.section_priority)

        boosted_results = booster.apply_boost(
            results=sample_results,
            section_priority=classified.section_priority,
            score_key="score",
            section_key="section_type",
            boosted_score_key="boosted_score"
        )

        print("\nBoost 적용 결과")
        for rank, item in enumerate(boosted_results, start=1):
            print(
                f"{rank}. "
                f"chunk_id={item['chunk_id']} | "
                f"section_type={item['section_type']} | "
                f"score={item['score']} | "
                f"boost={item['section_boost']} | "
                f"boosted_score={item['boosted_score']:.3f}"
            )


if __name__ == "__main__":
    main()