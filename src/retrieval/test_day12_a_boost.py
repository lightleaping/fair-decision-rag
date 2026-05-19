# src/retrieval/test_day12_a_boost.py

from src.retrieval.query_classifier import QueryClassifier
from src.retrieval.section_boost import SectionBooster


def main():
    classifier = QueryClassifier()
    booster = SectionBooster()

    sample_results = [
        {
            "chunk_id": "chunk_001",
            "section_type": "사실관계",
            "score": 8.0,
            "text": "피심인은 거래상대방에게 불리한 조건을 요구하였다."
        },
        {
            "chunk_id": "chunk_002",
            "section_type": "판단근거",
            "score": 7.8,
            "text": "이 행위는 거래상 지위를 남용한 행위로 판단된다."
        },
        {
            "chunk_id": "chunk_003",
            "section_type": "과징금",
            "score": 7.0,
            "text": "피심인에게 과징금 1억 원을 부과한다."
        },
        {
            "chunk_id": "chunk_004",
            "section_type": "적용법조",
            "score": 7.5,
            "text": "공정거래법 제23조 제1항이 적용된다."
        },
        {
            "chunk_id": "chunk_005",
            "section_type": "시정명령",
            "score": 7.2,
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
        classified = classifier.classify(query)

        boosted = booster.apply_boost(
            results=sample_results,
            section_priority=classified.section_priority,
        )

        print("\n" + "=" * 80)
        print("질문:", query)
        print("질문 유형:", classified.query_type)
        print("매칭 키워드:", classified.matched_keywords)

        print("\nBoost 결과")
        for rank, item in enumerate(boosted, start=1):
            print(
                f"{rank}. "
                f"chunk_id={item['chunk_id']} | "
                f"original={item['original_section_type']} | "
                f"normalized={item['section_type']} | "
                f"base={item['score']} | "
                f"boost={item['section_boost']} | "
                f"boosted={item['boosted_score']:.3f}"
            )


if __name__ == "__main__":
    main()