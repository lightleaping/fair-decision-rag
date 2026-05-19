# src/retrieval/section_boost.py

from typing import Any, Dict, List


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
        if not section_type:
            return section_priority.get("default", self.default_boost)

        section_type = section_type.strip()

        return section_priority.get(
            section_type,
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
            section_type = copied.get(section_key, "default")

            boost = self.get_boost(
                section_type=section_type,
                section_priority=section_priority,
            )

            copied["section_boost"] = boost
            copied[boosted_score_key] = base_score * boost

            boosted_results.append(copied)

        boosted_results.sort(
            key=lambda x: x.get(boosted_score_key, 0.0),
            reverse=True,
        )

        return boosted_results