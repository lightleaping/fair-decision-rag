# src/retrieval/query_classifier.py

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QueryTypeResult:
    query_type: str
    matched_keywords: List[str]
    section_priority: Dict[str, float]


class QueryClassifier:
    QUERY_KEYWORDS = {
        "penalty": ["과징금", "벌금", "금액", "얼마", "부과", "납부", "산정", "감경", "가중"],
        "legal_reasoning": ["왜", "이유", "판단", "위법", "법 위반", "인정", "근거", "판단 근거", "부당성"],
        "fact_pattern": ["행위", "무슨 일", "사실관계", "어떤 행위", "문제된 행위", "사건 내용", "거래", "요구", "강요", "제한"],
        "law_article": ["법조항", "조항", "법률", "시행령", "근거 법", "적용 법", "공정거래법", "하도급법", "가맹사업법", "표시광고법"],
        "corrective_order": ["시정명령", "조치", "명령", "처분", "재발방지", "공표", "통지", "최종 조치", "결론"],
        "summary": ["요약", "정리", "핵심", "간단히", "전체 내용", "무슨 사건"],
    }

    SECTION_PRIORITY = {
        "penalty": {
            "penalty": 1.50,
            "order": 1.30,
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
                self.SECTION_PRIORITY["general"],
            ),
        )