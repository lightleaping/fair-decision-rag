# src/retrieval/section_boost.py

from typing import Any, Dict, List


def normalize_section_type(section_type: str) -> str:
    """
    다양한 section_type 표현을 프로젝트 표준 section_type으로 정규화한다.
    """

    if not section_type:
        return "default"

    value = str(section_type).strip().lower()

    mapping = {
        "fact": "fact",
        "facts": "fact",
        "사실": "fact",
        "사실관계": "fact",
        "사건개요": "fact",
        "행위사실": "fact",

        "legal_reasoning": "legal_reasoning",
        "reasoning": "legal_reasoning",
        "judgment": "legal_reasoning",
        "판단": "legal_reasoning",
        "판단근거": "legal_reasoning",
        "법적판단": "legal_reasoning",
        "위법성판단": "legal_reasoning",
        "부당성판단": "legal_reasoning",

        "law_article": "law_article",
        "law": "law_article",
        "article": "law_article",
        "법조항": "law_article",
        "적용법조": "law_article",
        "관련법령": "law_article",
        "관련 법령": "law_article",

        "penalty": "penalty",
        "fine": "penalty",
        "과징금": "penalty",
        "제재": "penalty",
        "부과금": "penalty",
        "과태료": "penalty",

        "order": "order",
        "corrective_order": "order",
        "시정명령": "order",
        "시정조치": "order",
        "조치": "order",
        "주문": "order",
        "명령": "order",

        "conclusion": "conclusion",
        "결론": "conclusion",
        "의결": "conclusion",
        "최종판단": "conclusion",

        "summary": "summary",
        "요약": "summary",
        "개요": "summary",
    }

    return mapping.get(value, value if value else "default")


class SectionBooster:
    """
    검색 결과 chunk의 section_type에 따라 score를 보정하는 클래스.
    """

    def __init__(self, default_boost: float = 1.0):
        self.default_boost = default_boost

    def get_boost(
        self,
        section_type: str,
        section_priority: Dict[str, float],
    ) -> float:
        normalized = normalize_section_type(section_type)

        return section_priority.get(
            normalized,
            section_priority.get("default", self.default_boost),
        )

    def apply_boost(
        self,
        results: List[Dict[str, Any]],
        section_priority: Dict[str, float],
        score_key: str = "score",
        section_key: str = "section_type",
        boosted_score_key: str = "boosted_score",
    ) -> List[Dict[str, Any]]:
        boosted_results = []

        for item in results:
            copied = dict(item)

            base_score = float(copied.get(score_key, 0.0))
            original_section_type = copied.get(section_key, "default")
            normalized_section_type = normalize_section_type(original_section_type)

            boost = self.get_boost(
                section_type=normalized_section_type,
                section_priority=section_priority,
            )

            copied["original_section_type"] = original_section_type
            copied["section_type"] = normalized_section_type
            copied["section_boost"] = boost
            copied[boosted_score_key] = base_score * boost

            boosted_results.append(copied)

        boosted_results.sort(
            key=lambda x: x.get(boosted_score_key, 0.0),
            reverse=True,
        )

        return boosted_results