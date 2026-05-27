# src/query_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class QueryAnalysis:
    query: str
    query_type: str
    priority_sections: List[str]
    section_boost: Dict[str, float]
    keywords: List[str]


class QueryClassifier:
    """
    공정거래 의결서 질문 유형 분류기.

    역할:
    - 사용자 질문 유형 분류
    - 우선 검색할 section_type 결정
    - Hybrid Retrieval에서 사용할 section boost 반환

    조건:
    - 외부 API 호출 없음
    - 모델 학습 없음
    - 30초 이내 응답 구조에 부담 없음
    """

    VIOLATION_KEYWORDS = [
        "위반", "위법", "불공정", "담합", "부당", "제한",
        "거래상지위", "하도급", "가맹", "표시광고",
        "시장지배", "사업자단체", "금지행위"
    ]

    FACT_KEYWORDS = [
        "사실관계", "무슨 일", "행위", "어떤 행위", "경위",
        "배경", "계약", "거래", "요구", "강요", "제공",
        "판매", "공급", "피심인"
    ]

    LAW_KEYWORDS = [
        "법 조항", "조항", "근거 법", "적용 법", "공정거래법",
        "하도급법", "가맹사업법", "표시광고법", "시행령",
        "법률", "규정"
    ]

    SANCTION_KEYWORDS = [
        "시정명령", "과징금", "제재", "처분", "고발",
        "명령", "납부", "금액", "조치", "부과"
    ]

    JUDGMENT_KEYWORDS = [
        "판단", "이유", "근거", "왜", "인정",
        "해당하는 이유", "위원회 판단", "판단 근거",
        "위법하다고 본 이유"
    ]

    SUMMARY_KEYWORDS = [
        "요약", "정리", "핵심", "간단히", "전체 내용",
        "무슨 사건", "설명"
    ]

    SECTION_PROFILES = {
        "violation": {
            "priority_sections": [
                "legal_judgment",
                "violation_type",
                "facts",
                "order"
            ],
            "section_boost": {
                "legal_judgment": 1.30,
                "violation_type": 1.25,
                "facts": 1.10,
                "order": 1.05,
            },
        },
        "facts": {
            "priority_sections": [
                "facts",
                "background",
                "transaction_structure",
                "legal_judgment"
            ],
            "section_boost": {
                "facts": 1.35,
                "background": 1.20,
                "transaction_structure": 1.15,
                "legal_judgment": 1.05,
            },
        },
        "law": {
            "priority_sections": [
                "law",
                "legal_basis",
                "legal_judgment",
                "violation_type"
            ],
            "section_boost": {
                "law": 1.35,
                "legal_basis": 1.30,
                "legal_judgment": 1.15,
                "violation_type": 1.05,
            },
        },
        "sanction": {
            "priority_sections": [
                "order",
                "sanction",
                "penalty",
                "conclusion"
            ],
            "section_boost": {
                "order": 1.35,
                "sanction": 1.30,
                "penalty": 1.25,
                "conclusion": 1.10,
            },
        },
        "judgment": {
            "priority_sections": [
                "legal_judgment",
                "reasoning",
                "facts",
                "law"
            ],
            "section_boost": {
                "legal_judgment": 1.35,
                "reasoning": 1.25,
                "facts": 1.10,
                "law": 1.10,
            },
        },
        "summary": {
            "priority_sections": [
                "summary",
                "facts",
                "legal_judgment",
                "order"
            ],
            "section_boost": {
                "summary": 1.25,
                "facts": 1.15,
                "legal_judgment": 1.15,
                "order": 1.10,
            },
        },
        "general": {
            "priority_sections": [
                "facts",
                "legal_judgment",
                "law",
                "order"
            ],
            "section_boost": {
                "facts": 1.15,
                "legal_judgment": 1.15,
                "law": 1.05,
                "order": 1.05,
            },
        },
    }

    def classify(self, query: str) -> QueryAnalysis:
        normalized_query = self._normalize(query)
        query_type = self._detect_query_type(normalized_query)
        profile = self.SECTION_PROFILES[query_type]
        keywords = self._extract_keywords(normalized_query)

        return QueryAnalysis(
            query=query,
            query_type=query_type,
            priority_sections=profile["priority_sections"],
            section_boost=profile["section_boost"],
            keywords=keywords,
        )

    def _detect_query_type(self, query: str) -> str:
        scores = {
            "violation": self._count_matches(query, self.VIOLATION_KEYWORDS),
            "facts": self._count_matches(query, self.FACT_KEYWORDS),
            "law": self._count_matches(query, self.LAW_KEYWORDS),
            "sanction": self._count_matches(query, self.SANCTION_KEYWORDS),
            "judgment": self._count_matches(query, self.JUDGMENT_KEYWORDS),
            "summary": self._count_matches(query, self.SUMMARY_KEYWORDS),
        }

        best_type = max(scores, key=scores.get)

        if scores[best_type] == 0:
            return "general"

        return best_type

    def _extract_keywords(self, query: str) -> List[str]:
        cleaned = (
            query.replace("?", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace("!", " ")
        )

        tokens = [token.strip() for token in cleaned.split() if len(token.strip()) >= 2]

        stopwords = {
            "이", "가", "은", "는", "을", "를", "에", "의", "와", "과",
            "에서", "으로", "인가요", "무엇인가요", "알려줘", "설명해줘",
            "해줘", "뭐야", "어떻게", "되었어"
        }

        keywords = [token for token in tokens if token not in stopwords]

        return keywords[:10]

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.strip().lower().split())

    @staticmethod
    def _count_matches(query: str, keywords: List[str]) -> int:
        return sum(1 for keyword in keywords if keyword in query)


def analyze_query(query: str) -> QueryAnalysis:
    classifier = QueryClassifier()
    return classifier.classify(query)


if __name__ == "__main__":
    samples = [
        "한국파파존스 사건에서 어떤 법 위반이 있었어?",
        "이 사건의 사실관계를 요약해줘",
        "적용된 법 조항은 뭐야?",
        "과징금이나 시정명령은 어떻게 됐어?",
        "위원회가 위법하다고 판단한 이유는 뭐야?",
    ]

    classifier = QueryClassifier()

    for sample in samples:
        result = classifier.classify(sample)
        print("=" * 80)
        print("query:", result.query)
        print("query_type:", result.query_type)
        print("priority_sections:", result.priority_sections)
        print("section_boost:", result.section_boost)
        print("keywords:", result.keywords)