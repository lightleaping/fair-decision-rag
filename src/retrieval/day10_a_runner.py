# src/retrieval/day10_a_runner.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.retrieval.query_classifier import QueryClassifier
from src.retrieval.section_boost import SectionBooster
from src.retrieval.topk_selector import TopKSelector


REQUIRED_RESULT_KEYS = ["chunk_id", "section_type", "score", "text"]


def validate_search_results(search_results: List[Dict[str, Any]]) -> None:
    """
    BM25/Dense/Hybrid 寃??寃곌낵??湲곕낯 ?뺤떇??寃利앺븳??
    """

    if not isinstance(search_results, list):
        raise TypeError("search_results??list[dict] ?뺤떇?댁뼱???⑸땲??")

    if len(search_results) < 5:
        raise ValueError(
            f"寃???꾨낫媛 {len(search_results)}媛쒖엯?덈떎. "
            "Top-5瑜?留뚮뱾湲??꾪빐 理쒖냼 5媛??댁긽 ?꾩슂?⑸땲??"
        )

    for idx, item in enumerate(search_results):
        if not isinstance(item, dict):
            raise TypeError(f"{idx}踰덉㎏ 寃??寃곌낵媛 dict ?뺤떇???꾨떃?덈떎.")

        missing_keys = [
            key for key in REQUIRED_RESULT_KEYS
            if key not in item
        ]

        if missing_keys:
            raise ValueError(
                f"{idx}踰덉㎏ 寃??寃곌낵???꾩닔 key媛 ?놁뒿?덈떎: {missing_keys}"
            )

        if not item["chunk_id"]:
            raise ValueError(f"{idx}踰덉㎏ 寃??寃곌낵??chunk_id媛 鍮꾩뼱 ?덉뒿?덈떎.")

        try:
            float(item["score"])
        except ValueError:
            raise ValueError(
                f"{idx}踰덉㎏ 寃??寃곌낵??score瑜?float?쇰줈 蹂?섑븷 ???놁뒿?덈떎."
            )


def run_day10_a(
    query: str,
    search_results: List[Dict[str, Any]],
    valid_chunk_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Day 10-A 理쒖쥌 ?ㅽ뻾 ?⑥닔.

    ?낅젰:
    - query: ?ъ슜??吏덈Ц
    - search_results: BM25/Dense/Hybrid 寃???꾨낫 寃곌낵
    - valid_chunk_ids: 怨듦컻蹂??곗씠?곗뿉 議댁옱?섎뒗 chunk_id 吏묓빀

    異쒕젰:
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
    寃곌낵瑜?JSON ?뚯씪濡???ν븳??
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
            "text": "?쇱떖?몄? 嫄곕옒?곷?諛⑹뿉寃??뱀젙 議곌굔???붽뎄?섏???",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_002",
            "section_type": "legal_reasoning",
            "score": 7.9,
            "text": "???됱쐞??嫄곕옒??吏?꾨? ?댁슜??遺덉씠???쒓났?됱쐞濡??먮떒?쒕떎.",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "score": 6.8,
            "text": "?쇱떖?몄뿉寃?怨쇱쭠湲?1???먯쓣 遺怨쇳븳??",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_004",
            "section_type": "law_article",
            "score": 7.2,
            "text": "怨듭젙嫄곕옒踰???3議?????씠 ?곸슜?쒕떎.",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_005",
            "section_type": "order",
            "score": 6.9,
            "text": "?쇱떖?몄? ?ν썑 ?숈씪???됱쐞瑜?諛섎났?섏뿬?쒕뒗 ?꾨땲 ?쒕떎.",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_006",
            "section_type": "summary",
            "score": 7.0,
            "text": "???ш굔? 嫄곕옒??吏???⑥슜?됱쐞??愿??嫄댁씠??",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
    ]

    query = "???ш굔 怨쇱쭠湲덉? ?쇰쭏??"

    result = run_day10_a(
        query=query,
        search_results=sample_results,
    )

    save_result_json(
        result=result,
        output_path="outputs/results/day10_a_sample_result.json",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n????꾨즺: outputs/results/day10_a_sample_result.json")
