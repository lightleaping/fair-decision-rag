# src/retrieval/query_classifier.py

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QueryTypeResult:
    query_type: str
    matched_keywords: List[str]
    section_priority: Dict[str, float]


class QueryClassifier:
    """
    공정거래 의결서 질의 유형 분류기.

    목적:
    - 사용자 질문을 단순 키워드 기반으로 빠르게 분류한다.
    - 외부 API를 사용하지 않는다.
    - 모델 학습이 필요 없다.
    - BM25 / Dense / Hybrid Retrieval의 section boost에 사용한다.
    """

    QUERY_KEYWORDS = {
        "penalty": [
            "과징금", "벌금", "금액", "얼마", "부과", "납부", "산정", "감경", "가중"
        ],
        "legal_reasoning": [
            "왜", "이유", "판단", "위법", "법 위반", "인정", "근거", "판단 근거",
            "부당성", "경쟁 제한성", "거래상 지위"
        ],
        "fact_pattern": [
            "행위", "무슨 일", "사실관계", "어떤 행위", "문제된 행위", "사건 내용",
            "거래", "요구", "강요", "제한", "거절"
        ],
        "law_article": [
            "법조항", "조항", "법률", "시행령", "근거 법", "적용 법", "공정거래법",
            "하도급법", "가맹사업법", "표시광고법"
        ],
        "corrective_order": [
            "시정명령", "조치", "명령", "처분", "재발방지", "공표", "통지",
            "최종 조치", "결론"
        ],
        "summary": [
            "요약", "정리", "핵심", "간단히", "전체 내용", "무슨 사건"
        ],
    }

    SECTION_PRIORITY = {
        "penalty": {
            "order": 1.30,
            "penalty": 1.50,
            "conclusion": 1.25,
            "legal_reasoning": 1.10,
            "fact": 0.95,
            "default": 1.00,
        },
        "legal_reasoning": {
            "legal_reasoning": 1.50,
            "law_article": 1.25,
            "fact": 1.15,
            "conclusion": 1.10,
            "default": 1.00,
        },
        "fact_pattern": {
            "fact": 1.50,
            "summary": 1.20,
            "legal_reasoning": 1.10,
            "default": 1.00,
        },
        "law_article": {
            "law_article": 1.50,
            "legal_reasoning": 1.30,
            "conclusion": 1.10,
            "default": 1.00,
        },
        "corrective_order": {
            "order": 1.50,
            "conclusion": 1.30,
            "penalty": 1.15,
            "default": 1.00,
        },
        "summary": {
            "summary": 1.35,
            "fact": 1.25,
            "legal_reasoning": 1.15,
            "conclusion": 1.10,
            "default": 1.00,
        },
        "general": {
            "summary": 1.15,
            "fact": 1.10,
            "legal_reasoning": 1.10,
            "default": 1.00,
        },
    }

    def classify(self, query: str) -> QueryTypeResult:
        query = query.strip()

        if not query:
            return QueryTypeResult(
                query_type="general",
                matched_keywords=[],
                section_priority=self.SECTION_PRIORITY["general"],
            )

        scores = {}
        matched = {}

        for query_type, keywords in self.QUERY_KEYWORDS.items():
            matched_keywords = [kw for kw in keywords if kw in query]
            if matched_keywords:
                scores[query_type] = len(matched_keywords)
                matched[query_type] = matched_keywords

        if not scores:
            query_type = "general"
            matched_keywords = []
        else:
            query_type = max(scores, key=scores.get)
            matched_keywords = matched[query_type]

        return QueryTypeResult(
            query_type=query_type,
            matched_keywords=matched_keywords,
            section_priority=self.SECTION_PRIORITY.get(
                query_type,
                self.SECTION_PRIORITY["general"]
            ),
        )


if __name__ == "__main__":
    classifier = QueryClassifier()

    test_queries = [
        "이 사건 과징금은 얼마야?",
        "왜 위법하다고 판단했어?",
        "어떤 행위가 문제였어?",
        "적용 법조항 알려줘",
        "최종 시정명령이 뭐야?",
        "전체 내용 요약해줘",
    ]

    for q in test_queries:
        result = classifier.classify(q)
        print("=" * 60)
        print("질문:", q)
        print("분류:", result.query_type)
        print("매칭 키워드:", result.matched_keywords)
        print("section priority:", result.section_priority)