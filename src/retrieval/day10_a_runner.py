# src/retrieval/day10_a_runner.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from query_classifier import QueryClassifier
from section_boost import SectionBooster
from topk_selector import TopKSelector


REQUIRED_RESULT_KEYS = ["chunk_id", "section_type", "score", "text"]


def validate_search_results(search_results: List[Dict[str, Any]]) -> None:
    """
    BM25/Dense/Hybrid 검색 결과의 기본 형식을 검증한다.
    """

    if not isinstance(search_results, list):
        raise TypeError("search_results는 list[dict] 형식이어야 합니다.")

    if len(search_results) < 5:
        raise ValueError(
            f"검색 후보가 {len(search_results)}개입니다. "
            "Top-5를 만들기 위해 최소 5개 이상 필요합니다."
        )

    for idx, item in enumerate(search_results):
        if not isinstance(item, dict):
            raise TypeError(f"{idx}번째 검색 결과가 dict 형식이 아닙니다.")

        missing_keys = [
            key for key in REQUIRED_RESULT_KEYS
            if key not in item
        ]

        if missing_keys:
            raise ValueError(
                f"{idx}번째 검색 결과에 필수 key가 없습니다: {missing_keys}"
            )

        if not item["chunk_id"]:
            raise ValueError(f"{idx}번째 검색 결과의 chunk_id가 비어 있습니다.")

        try:
            float(item["score"])
        except ValueError:
            raise ValueError(
                f"{idx}번째 검색 결과의 score를 float으로 변환할 수 없습니다."
            )


def run_day10_a(
    query: str,
    search_results: List[Dict[str, Any]],
    valid_chunk_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Day 10-A 최종 실행 함수.

    입력:
    - query: 사용자 질문
    - search_results: BM25/Dense/Hybrid 검색 후보 결과
    - valid_chunk_ids: 공개본 데이터에 존재하는 chunk_id 집합

    출력:
    - query_type
    - matched_keywords
    - top5_chunk_ids
    - top5_results
    """

    validate_search_results(search_results)

    classifier = QueryClassifier()
    booster = SectionBooster()
    selector = TopKSelector(k=5)

    classified = classifier.classify(query)

    boosted_results = booster.apply_boost(
        results=search_results,
        section_priority=classified.section_priority,
        score_key="score",
        section_key="section_type",
        boosted_score_key="boosted_score",
    )

    top5_results = selector.select(
        results=boosted_results,
        score_key="boosted_score",
        chunk_id_key="chunk_id",
        valid_chunk_ids=valid_chunk_ids,
    )

    formatted_top5 = []

    for rank, item in enumerate(top5_results, start=1):
        formatted_top5.append(
            {
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "section_type": item.get("section_type", "default"),
                "score": float(item.get("score", 0.0)),
                "section_boost": float(item.get("section_boost", 1.0)),
                "boosted_score": float(item.get("boosted_score", 0.0)),
                "text": item.get("text", ""),
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "source_file": item.get("source_file"),
                "page": item.get("page"),
            }
        )

    top5_chunk_ids = [item["chunk_id"] for item in formatted_top5]

    assert len(top5_chunk_ids) == 5
    assert len(set(top5_chunk_ids)) == 5

    return {
        "query": query,
        "query_type": classified.query_type,
        "matched_keywords": classified.matched_keywords,
        "top5_chunk_ids": top5_chunk_ids,
        "top5_results": formatted_top5,
    }


def save_result_json(result: Dict[str, Any], output_path: str) -> None:
    """
    결과를 JSON 파일로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sample_results = [
        {
            "chunk_id": "case_001_chunk_001",
            "section_type": "fact",
            "score": 8.4,
            "text": "피심인은 거래상대방에게 특정 조건을 요구하였다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_002",
            "section_type": "legal_reasoning",
            "score": 7.9,
            "text": "이 행위는 거래상 지위를 이용한 불이익 제공행위로 판단된다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "score": 6.8,
            "text": "피심인에게 과징금 1억 원을 부과한다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_004",
            "section_type": "law_article",
            "score": 7.2,
            "text": "공정거래법 제23조 제1항이 적용된다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_005",
            "section_type": "order",
            "score": 6.9,
            "text": "피심인은 향후 동일한 행위를 반복하여서는 아니 된다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_006",
            "section_type": "summary",
            "score": 7.0,
            "text": "이 사건은 거래상 지위 남용행위에 관한 건이다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
    ]

    query = "이 사건 과징금은 얼마야?"

    result = run_day10_a(
        query=query,
        search_results=sample_results,
    )

    save_result_json(
        result=result,
        output_path="outputs/results/day10_a_sample_result.json",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n저장 완료: outputs/results/day10_a_sample_result.json")