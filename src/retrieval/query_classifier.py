from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QueryTypeResult:
    query_type: str
    matched_keywords: List[str]
    section_priority: Dict[str, float]


class QueryClassifier:
    STRONG_CUES = [
        ("penalty", ["과징금", "과태료", "산정"]),
        ("order", ["주문", "시정명령", "처분", "명령"]),
        ("law_article", ["법조항", "법률 조항", "몇 조", "적용법"]),
        ("summary", ["요약", "핵심", "개요"]),
        ("legal_reasoning", ["판단 근거", "위법한 이유", "왜 위반"]),
        ("fact", ["사실관계", "어떤 행위", "무엇을 했"]),
    ]
    KEYWORDS = {
        "penalty": ["과징금", "과태료", "부과", "금액", "산정"],
        "legal_reasoning": ["이유", "판단", "위법", "위반", "근거"],
        "fact": ["행위", "사실관계", "무엇을", "어떤 행위"],
        "law_article": ["법조항", "법률", "법령", "몇 조", "적용법"],
        "order": ["주문", "시정명령", "조치", "처분", "명령"],
        "summary": ["요약", "핵심", "개요", "정리"],
    }
    PRIORITY = {
        "penalty": {"penalty": 3.0, "order": 1.8, "legal_reasoning": 1.1},
        "legal_reasoning": {"legal_reasoning": 2.2, "law_article": 1.5, "fact": 1.1},
        "fact": {"fact": 2.2, "legal_reasoning": 1.2},
        "law_article": {"law_article": 2.5, "legal_reasoning": 1.5},
        "order": {"order": 4.0, "conclusion": 1.8, "penalty": 1.3},
        "summary": {"summary": 1.4, "fact": 1.2, "legal_reasoning": 1.2},
        "general": {"order": 1.1, "legal_reasoning": 1.1, "default": 1.0},
    }

    def classify(self, query: str) -> QueryTypeResult:
        matches = {
            kind: [keyword for keyword in keywords if keyword in query]
            for kind, keywords in self.KEYWORDS.items()
        }
        matches = {kind: values for kind, values in matches.items() if values}
        strong_match = next(
            (
                (kind, [cue for cue in cues if cue in query])
                for kind, cues in self.STRONG_CUES
                if any(cue in query for cue in cues)
            ),
            None,
        )
        if strong_match:
            query_type, strong_keywords = strong_match
            matches[query_type] = list(
                dict.fromkeys(matches.get(query_type, []) + strong_keywords)
            )
        else:
            query_type = max(
                matches,
                key=lambda kind: sum(len(keyword) for keyword in matches[kind]),
                default="general",
            )
        return QueryTypeResult(
            query_type,
            matches.get(query_type, []),
            {**{"default": 1.0}, **self.PRIORITY[query_type]},
        )
