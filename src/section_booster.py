# src/section_booster.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class BoostedSearchResult:
    chunk_id: str
    original_score: float
    boosted_score: float
    section_type: str
    boost: float
    text: str
    metadata: Dict


class SectionBooster:
    """
    Day 12-A section boost 적용기.

    역할:
    - query_classifier.py가 반환한 section_boost를 검색 결과에 적용한다.
    - chunk metadata 안의 section_type을 읽는다.
    - 검색 점수에 section별 가중치를 곱한다.
    - 중복 없는 Top-K 결과를 반환한다.

    조건:
    - 외부 API 호출 없음
    - 모델 학습 없음
    - chunk_id 변경 없음
    """

    DEFAULT_SECTION_TYPE = "unknown"
    DEFAULT_BOOST = 1.0

    def apply_boost(
        self,
        results: List,
        section_boost: Dict[str, float],
        top_k: int = 5,
    ) -> List[BoostedSearchResult]:
        boosted_results: List[BoostedSearchResult] = []

        for result in results:
            chunk_id = self._get_value(result, "chunk_id")
            original_score = float(self._get_value(result, "score", default=0.0))
            text = self._get_value(result, "text", default="")
            metadata = self._get_value(result, "metadata", default={})

            if not isinstance(metadata, dict):
                metadata = {}

            section_type = self._extract_section_type(metadata)
            boost = section_boost.get(section_type, self.DEFAULT_BOOST)
            boosted_score = original_score * boost

            boosted_results.append(
                BoostedSearchResult(
                    chunk_id=str(chunk_id),
                    original_score=original_score,
                    boosted_score=boosted_score,
                    section_type=section_type,
                    boost=boost,
                    text=str(text),
                    metadata=metadata,
                )
            )

        boosted_results.sort(key=lambda item: item.boosted_score, reverse=True)

        return self._deduplicate_top_k(boosted_results, top_k=top_k)

    def _deduplicate_top_k(
        self,
        results: List[BoostedSearchResult],
        top_k: int,
    ) -> List[BoostedSearchResult]:
        deduplicated: List[BoostedSearchResult] = []
        seen_chunk_ids = set()

        for result in results:
            if not result.chunk_id:
                continue

            if result.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(result.chunk_id)
            deduplicated.append(result)

            if len(deduplicated) == top_k:
                break

        if len(deduplicated) != top_k:
            raise ValueError(
                f"SectionBooster must return exactly {top_k} unique chunk_ids, "
                f"but got {len(deduplicated)}."
            )

        return deduplicated

    def _extract_section_type(self, metadata: Dict) -> str:
        for key in [
            "section_type",
            "section",
            "section_name",
            "type",
            "label",
        ]:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return self.DEFAULT_SECTION_TYPE

    def _get_value(self, obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)


def apply_section_boost(
    results: List,
    section_boost: Dict[str, float],
    top_k: int = 5,
) -> List[BoostedSearchResult]:
    booster = SectionBooster()
    return booster.apply_boost(
        results=results,
        section_boost=section_boost,
        top_k=top_k,
    )


if __name__ == "__main__":
    from dense_retriever import SimpleDenseRetriever
    from query_classifier import QueryClassifier

    chunk_file = "data/chunks.jsonl"
    query = "한국파파존스 사건에서 어떤 법 위반이 있었어?"

    classifier = QueryClassifier()
    analysis = classifier.classify(query)

    retriever = SimpleDenseRetriever(chunk_file)
    dense_results = retriever.search(query=query, top_k=20)

    booster = SectionBooster()
    boosted_results = booster.apply_boost(
        results=dense_results,
        section_boost=analysis.section_boost,
        top_k=5,
    )

    print("query:", query)
    print("query_type:", analysis.query_type)
    print("priority_sections:", analysis.priority_sections)
    print("section_boost:", analysis.section_boost)

    for index, result in enumerate(boosted_results, start=1):
        print("=" * 80)
        print("rank:", index)
        print("chunk_id:", result.chunk_id)
        print("section_type:", result.section_type)
        print("original_score:", result.original_score)
        print("boost:", result.boost)
        print("boosted_score:", result.boosted_score)
        print("text:", result.text[:300])