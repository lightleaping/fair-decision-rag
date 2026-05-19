# src/retrieval/section_boost.py

from typing import Dict, List, Any


class SectionBooster:
    """
    검색 결과 chunk의 section_type에 따라 score를 보정하는 클래스.

    목적:
    - 질문 유형별로 중요한 section을 더 높게 평가한다.
    - BM25, Dense, Hybrid Retrieval 결과에 공통으로 적용 가능하다.
    - 외부 API나 모델 학습 없이 동작한다.
    """

    def __init__(self, default_boost: float = 1.0):
        self.default_boost = default_boost

    def get_boost(
        self,
        section_type: str,
        section_priority: Dict[str, float],
    ) -> float:
        """
        section_type에 해당하는 boost 값을 반환한다.

        Args:
            section_type:
                chunk의 section_type.
                예: fact, legal_reasoning, law_article, penalty, order, conclusion

            section_priority:
                QueryClassifier에서 반환한 section priority dict.

        Returns:
            float: score에 곱할 boost 값
        """

        if not section_type:
            return section_priority.get("default", self.default_boost)

        section_type = section_type.strip()

        return section_priority.get(
            section_type,
            section_priority.get("default", self.default_boost)
        )

    def apply_boost(
        self,
        results: List[Dict[str, Any]],
        section_priority: Dict[str, float],
        score_key: str = "score",
        section_key: str = "section_type",
        boosted_score_key: str = "boosted_score",
    ) -> List[Dict[str, Any]]:
        """
        검색 결과 리스트에 section boost를 적용한다.

        Args:
            results:
                검색 결과 리스트.
                각 item은 최소 score와 section_type을 가져야 한다.

            section_priority:
                QueryClassifier에서 반환된 section priority.

            score_key:
                원래 검색 점수 key.

            section_key:
                section_type이 들어 있는 key.

            boosted_score_key:
                보정된 점수를 저장할 key.

        Returns:
            List[Dict[str, Any]]:
                boosted_score 기준으로 정렬된 검색 결과.
        """

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
            reverse=True
        )

        return boosted_results


if __name__ == "__main__":
    sample_results = [
        {
            "chunk_id": "doc1_chunk_001",
            "section_type": "fact",
            "score": 7.2,
            "text": "피심인은 거래상 지위를 이용하여..."
        },
        {
            "chunk_id": "doc1_chunk_002",
            "section_type": "penalty",
            "score": 6.8,
            "text": "과징금은 다음과 같이 산정한다..."
        },
        {
            "chunk_id": "doc1_chunk_003",
            "section_type": "legal_reasoning",
            "score": 7.0,
            "text": "이 행위는 공정거래법상 부당한 행위에 해당한다..."
        },
    ]

    section_priority = {
        "penalty": 1.5,
        "order": 1.3,
        "legal_reasoning": 1.1,
        "fact": 0.95,
        "default": 1.0,
    }

    booster = SectionBooster()
    boosted = booster.apply_boost(sample_results, section_priority)

    for item in boosted:
        print("=" * 60)
        print("chunk_id:", item["chunk_id"])
        print("section_type:", item["section_type"])
        print("base_score:", item["score"])
        print("section_boost:", item["section_boost"])
        print("boosted_score:", item["boosted_score"])