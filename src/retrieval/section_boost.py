from typing import Any, Dict, List


def normalize_section_type(section_type: str) -> str:
    value = str(section_type or "").strip().lower().replace(" ", "")
    if "주문" in value or value == "order":
        return "order"
    if "이유" in value or "판단" in value or value in {"legal_reasoning", "reasoning"}:
        return "legal_reasoning"
    if "과징금" in value or "과태료" in value or value == "penalty":
        return "penalty"
    if "법령" in value or "법조" in value or value == "law_article":
        return "law_article"
    if "사실" in value or "행위" in value or value == "fact":
        return "fact"
    if "결론" in value or value == "conclusion":
        return "conclusion"
    return value or "default"


class SectionBooster:
    def apply_boost(
        self,
        results: List[Dict[str, Any]],
        section_priority: Dict[str, float],
        score_key: str = "score",
        section_key: str = "section_type",
        boosted_score_key: str = "boosted_score",
    ) -> List[Dict[str, Any]]:
        boosted = []
        for result in results:
            item = dict(result)
            normalized = normalize_section_type(item.get(section_key, "default"))
            factor = section_priority.get(
                normalized, section_priority.get("default", 1.0)
            )
            item["section_type"] = normalized
            item["section_boost"] = factor
            item[boosted_score_key] = float(item.get(score_key, 0.0)) * factor
            boosted.append(item)
        return sorted(boosted, key=lambda item: item[boosted_score_key], reverse=True)
