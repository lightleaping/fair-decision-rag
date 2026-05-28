# src/retrieval/query_classifier.py

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QueryTypeResult:
    """
    Day 10 이후 A 템플릿 기준 분류 결과.
    SectionBooster, TopKSelector와 함께 사용한다.
    """
    query_type: str
    matched_keywords: List[str]
    section_priority: Dict[str, float]


@dataclass
class QueryClassification:
    """
    Day 9 이전 코드와의 호환용 분류 결과.
    기존 hybrid_retriever.py가 classify_query()를 호출할 수 있으므로 유지한다.
    """
    query_type: str
    priority_sections: List[str]
    bm25_weight: float
    dense_weight: float


class QueryClassifier:
    """
    공정거래 의결서 질의 유형 분류기.

    역할:
    - 사용자 질문을 검색 목적에 맞는 유형으로 분류한다.
    - 질문 유형별 우선 section_type 가중치를 반환한다.
    - BM25, Dense, Hybrid Retrieval에서 공통 사용한다.

    기준:
    - 외부 API 호출 없음
    - 규칙 기반 분류
    - 빠른 실행
    - 설명 가능한 검색 로직
    """

    QUERY_KEYWORDS = {
        "penalty": [
            "과징금", "벌금", "제재금", "금액", "얼마", "부과", "납부",
            "산정", "감경", "가중", "부과기준", "부과기준율",
            "기준금액", "최종금액", "관련매출액", "매출액"
        ],
        "legal_reasoning": [
            "왜", "이유", "판단", "위법", "법 위반", "인정", "근거",
            "판단 근거", "부당성", "경쟁 제한성", "거래상 지위",
            "위법성", "법리", "판단 이유", "인정 이유"
        ],
        "fact_pattern": [
            "행위", "무슨 일", "사실관계", "어떤 행위", "문제된 행위",
            "사건 내용", "거래", "요구", "강요", "제한", "거절",
            "불이익", "조건", "계약", "발생", "경위",
            "담합", "합의", "공동행위", "하도급"
        ],
        "law_article": [
            "법조항", "조항", "법률", "시행령", "근거 법", "적용 법",
            "공정거래법", "하도급법", "가맹사업법", "표시광고법",
            "약관법", "전자상거래법", "제 몇 조", "적용 조항",
            "위반 조항", "법 제"
        ],
        "corrective_order": [
            "시정명령", "조치", "명령", "처분", "재발방지", "공표",
            "통지", "최종 조치", "결론", "의결", "주문", "시정조치",
            "향후", "금지명령", "고발", "경고", "납부명령"
        ],
        "summary": [
            "요약", "정리", "핵심", "간단히", "전체 내용", "무슨 사건",
            "개요", "한 줄", "요지는", "핵심만"
        ],
    }

    SECTION_PRIORITY = {
        "penalty": {
            "penalty": 1.65,
            "order": 1.30,
            "conclusion": 1.20,
            "legal_reasoning": 1.10,
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
            "legal_reasoning": 1.25,
            "summary": 1.15,
            "law_article": 1.00,
            "order": 0.95,
            "penalty": 0.90,
            "default": 1.00,
        },
        "law_article": {
            "law_article": 1.65,
            "legal_reasoning": 1.35,
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
            "order": 1.00,
            "default": 1.00,
        },
    }

    def classify(self, query: str) -> QueryTypeResult:
        query = (query or "").strip()

        if not query:
            return QueryTypeResult(
                query_type="general",
                matched_keywords=[],
                section_priority=self.SECTION_PRIORITY["general"],
            )

        scores: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}

        for query_type, keywords in self.QUERY_KEYWORDS.items():
            matched_keywords = [kw for kw in keywords if kw in query]

            if matched_keywords:
                # 긴 키워드는 의도가 더 명확하므로 가중치를 더 준다.
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


# ---------------------------------------------------------------------
# 아래는 Day 9 이전 코드 호환용 함수들.
# 기존 src/hybrid_retriever.py 등이 classify_query()를 import하고 있을 수 있어 유지한다.
# ---------------------------------------------------------------------

KOREAN_QUERY_TYPE = {
    "penalty": "과징금 질문",
    "legal_reasoning": "위반 이유 질문",
    "fact_pattern": "행위 패턴 질문",
    "law_article": "법 조항 질문",
    "corrective_order": "처분 결과 질문",
    "summary": "사건 요약 질문",
    "general": "일반 질문",
}


KOREAN_PRIORITY_SECTIONS = {
    "penalty": ["주문", "별지", "이유"],
    "legal_reasoning": ["이유"],
    "fact_pattern": ["이유"],
    "law_article": ["이유"],
    "corrective_order": ["주문", "결론"],
    "summary": ["주문", "이유"],
    "general": ["주문", "이유", "별지", "결론", "기타"],
}


RETRIEVAL_WEIGHTS = {
    "penalty": {"bm25": 0.7, "dense": 0.3},
    "corrective_order": {"bm25": 0.6, "dense": 0.4},
    "legal_reasoning": {"bm25": 0.5, "dense": 0.5},
    "fact_pattern": {"bm25": 0.4, "dense": 0.6},
    "law_article": {"bm25": 0.7, "dense": 0.3},
    "summary": {"bm25": 0.3, "dense": 0.7},
    "general": {"bm25": 0.5, "dense": 0.5},
}


def classify_question_type(query: str) -> str:
    """
    기존 코드 호환용.
    한국어 query_type 문자열을 반환한다.
    """
    result = QueryClassifier().classify(query)
    return KOREAN_QUERY_TYPE.get(result.query_type, "일반 질문")


def get_priority_sections(query_type: str) -> List[str]:
    """
    기존 코드 호환용.
    한국어 query_type 또는 내부 query_type을 받아 한국어 section 리스트를 반환한다.
    """
    reverse_map = {v: k for k, v in KOREAN_QUERY_TYPE.items()}
    internal_type = reverse_map.get(query_type, query_type)
    return KOREAN_PRIORITY_SECTIONS.get(
        internal_type,
        KOREAN_PRIORITY_SECTIONS["general"],
    )


def get_retrieval_weights(query_type: str) -> Tuple[float, float]:
    """
    기존 코드 호환용.
    한국어 query_type 또는 내부 query_type을 받아 BM25/Dense weight를 반환한다.
    """
    reverse_map = {v: k for k, v in KOREAN_QUERY_TYPE.items()}
    internal_type = reverse_map.get(query_type, query_type)
    weights = RETRIEVAL_WEIGHTS.get(internal_type, RETRIEVAL_WEIGHTS["general"])
    return weights["bm25"], weights["dense"]


def classify_query(query: str) -> QueryClassification:
    """
    기존 Day 9 hybrid_retriever.py 호환용 함수.
    """
    classified = QueryClassifier().classify(query)
    korean_query_type = KOREAN_QUERY_TYPE.get(classified.query_type, "일반 질문")
    priority_sections = get_priority_sections(classified.query_type)
    bm25_weight, dense_weight = get_retrieval_weights(classified.query_type)

    return QueryClassification(
        query_type=korean_query_type,
        priority_sections=priority_sections,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )