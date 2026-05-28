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
    - 사용자 질문을 의결서 검색에 적합한 유형으로 분류한다.
    - 질문 유형별로 중요한 section_type 우선순위를 반환한다.
    - BM25, Dense, Hybrid Retrieval 모두에서 공통 사용한다.
    """

    QUERY_KEYWORDS = {
        "penalty": [
            "과징금", "벌금", "제재금", "금액", "얼마", "부과", "납부",
            "산정", "감경", "가중", "부과기준", "기준금액", "최종금액"
        ],
        "legal_reasoning": [
            "왜", "이유", "판단", "위법", "법 위반", "인정", "근거",
            "판단 근거", "부당성", "경쟁 제한성", "거래상 지위",
            "위법성", "법리", "판단 이유", "인정 이유"
        ],
        "fact_pattern": [
            "행위", "무슨 일", "사실관계", "어떤 행위", "문제된 행위",
            "사건 내용", "거래", "요구", "강요", "제한", "거절",
            "불이익", "조건", "계약", "발생", "경위"
        ],
        "law_article": [
            "법조항", "조항", "법률", "시행령", "근거 법", "적용 법",
            "공정거래법", "하도급법", "가맹사업법", "표시광고법",
            "약관법", "전자상거래법", "제 몇 조", "적용 조항"
        ],
        "corrective_order": [
            "시정명령", "조치", "명령", "처분", "재발방지", "공표",
            "통지", "최종 조치", "결론", "의결", "주문", "시정조치",
            "향후", "금지명령"
        ],
        "summary": [
            "요약", "정리", "핵심", "간단히", "전체 내용", "무슨 사건",
            "개요", "한 줄", "요지는", "핵심만"
        ],
    }

    SECTION_PRIORITY = {
        "penalty": {
            "penalty": 1.65,
            "order": 1.25,
            "conclusion": 1.20,
            "legal_reasoning": 1.05,
            "law_article": 1.05,
            "fact": 0.90,
            "summary": 1.00,
            "default": 1.00,
        },
        "legal_reasoning": {
            "legal_reasoning": 1.65,
            "law_article": 1.30,
            "fact": 1.15,
            "conclusion": 1.10,
            "order": 1.00,
            "penalty": 0.95,
            "summary": 1.05,
            "default": 1.00,
        },
        "fact_pattern": {
            "fact": 1.65,
            "summary": 1.20,
            "legal_reasoning": 1.15,
            "law_article": 1.00,
            "order": 0.95,
            "penalty": 0.90,
            "default": 1.00,
        },
        "law_article": {
            "law_article": 1.65,
            "legal_reasoning": 1.30,
            "conclusion": 1.10,
            "fact": 0.95,
            "order": 1.00,
            "penalty": 0.95,
            "summary": 1.00,
            "default": 1.00,
        },
        "corrective_order": {
            "order": 1.65,
            "conclusion": 1.35,
            "penalty": 1.10,
            "legal_reasoning": 1.05,
            "fact": 0.95,
            "summary": 1.05,
            "default": 1.00,
        },
        "summary": {
            "summary": 1.40,
            "fact": 1.25,
            "legal_reasoning": 1.20,
            "conclusion": 1.15,
            "order": 1.05,
            "penalty": 1.00,
            "law_article": 1.00,
            "default": 1.00,
        },
        "general": {
            "summary": 1.20,
            "fact": 1.15,
            "legal_reasoning": 1.15,
            "conclusion": 1.05,
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
                # 긴 키워드가 매칭되면 조금 더 높은 점수 부여
                score = sum(1.5 if len(kw) >= 4 else 1.0 for kw in matched_keywords)
                scores[query_type] = score
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
                self.SECTION_PRIORITY["general"],
            ),
        )