# src/retrieval/evaluator.py

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


DEFAULT_RESULT_PATH = "outputs/results/day12_hybrid_top5.json"
DEFAULT_QA_PATH = "data/processed/qa_eval_set.jsonl"
DEFAULT_OUTPUT_PATH = "outputs/results/day13_eval_report.json"


def load_json(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)

    if not path.exists():
        return []

    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{line_idx}번째 줄 JSON 파싱 실패: {error}")

    return items


def load_qa_eval_set(file_path: str) -> List[Dict[str, Any]]:
    """
    QA 검증셋을 JSONL 형식으로 불러온다.
    기존 evaluator.py 함수명 호환용.
    """
    return load_jsonl(file_path)


def normalize_gold_chunk_ids(item: Dict[str, Any]) -> List[str]:
    """
    QA 평가셋의 정답 chunk_id 형식을 통일한다.

    지원 형식:
    1. {"gold_chunk_ids": ["..."]}
    2. {"positive_chunk_id": "..."}
    3. {"positive_chunk_ids": ["..."]}
    """

    if isinstance(item.get("gold_chunk_ids"), list):
        return item["gold_chunk_ids"]

    if isinstance(item.get("positive_chunk_ids"), list):
        return item["positive_chunk_ids"]

    if item.get("positive_chunk_id"):
        return [item["positive_chunk_id"]]

    return []


def build_gold_map(qa_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    gold_map = {}

    for item in qa_items:
        query = item.get("query")
        gold_chunk_ids = normalize_gold_chunk_ids(item)

        if query:
            gold_map[query] = gold_chunk_ids

    return gold_map


def calculate_recall_at_5(
    retrieved_chunk_ids: List[str],
    positive_chunk_id_or_ids,
) -> Optional[float]:
    """
    Recall@5 계산.

    기존 코드 호환:
    - positive_chunk_id: str이면 Top-5 안에 있으면 1.0, 없으면 0.0

    확장:
    - positive_chunk_ids: list이면 Top-5 안에 들어온 정답 비율 계산
    - 정답이 비어 있으면 None 반환
    """

    if isinstance(positive_chunk_id_or_ids, str):
        if not positive_chunk_id_or_ids:
            return None
        return 1.0 if positive_chunk_id_or_ids in retrieved_chunk_ids[:5] else 0.0

    gold_chunk_ids = positive_chunk_id_or_ids or []

    if not gold_chunk_ids:
        return None

    predicted_set = set(retrieved_chunk_ids[:5])
    gold_set = set(gold_chunk_ids)

    hit_count = len(predicted_set & gold_set)

    return hit_count / len(gold_set)


def calculate_mrr(
    retrieved_chunk_ids: List[str],
    positive_chunk_id_or_ids,
) -> Optional[float]:
    """
    MRR 계산.

    기존 코드 호환:
    - positive_chunk_id: str 하나를 기준으로 reciprocal rank 계산

    확장:
    - positive_chunk_ids: list 중 하나라도 등장하면 가장 빠른 순위 기준으로 reciprocal rank 계산
    - 정답이 비어 있으면 None 반환
    """

    if isinstance(positive_chunk_id_or_ids, str):
        if not positive_chunk_id_or_ids:
            return None
        gold_set = {positive_chunk_id_or_ids}
    else:
        gold_chunk_ids = positive_chunk_id_or_ids or []

        if not gold_chunk_ids:
            return None

        gold_set = set(gold_chunk_ids)

    for idx, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / idx

    return 0.0


def evaluate_result_item(
    item: Dict[str, Any],
    gold_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    query = item.get("query")
    top5_chunk_ids = item.get("top5_chunk_ids", [])
    elapsed_seconds = float(item.get("elapsed_seconds", 0.0))
    within_30_seconds = item.get("within_30_seconds", elapsed_seconds <= 30.0)

    gold_chunk_ids = gold_map.get(query, [])

    has_exactly_5 = len(top5_chunk_ids) == 5
    has_no_duplicates = len(set(top5_chunk_ids)) == 5

    recall_at_5 = calculate_recall_at_5(
        retrieved_chunk_ids=top5_chunk_ids,
        positive_chunk_id_or_ids=gold_chunk_ids,
    )

    mrr = calculate_mrr(
        retrieved_chunk_ids=top5_chunk_ids,
        positive_chunk_id_or_ids=gold_chunk_ids,
    )

    return {
        "query": query,
        "retriever": item.get("retriever"),
        "dense_available": item.get("dense_available"),
        "elapsed_seconds": elapsed_seconds,
        "within_30_seconds": within_30_seconds,
        "top5_chunk_ids": top5_chunk_ids,
        "has_exactly_5": has_exactly_5,
        "has_no_duplicates": has_no_duplicates,
        "gold_chunk_ids": gold_chunk_ids,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "format_passed": has_exactly_5 and has_no_duplicates and within_30_seconds,
    }


def evaluate_result_file(
    result_path: str = DEFAULT_RESULT_PATH,
    qa_path: str = DEFAULT_QA_PATH,
) -> Dict[str, Any]:
    result_data = load_json(result_path)
    qa_items = load_qa_eval_set(qa_path)
    gold_map = build_gold_map(qa_items)

    result_items = result_data.get("results", [])

    if not result_items:
        raise ValueError("평가할 results가 비어 있습니다.")

    reports = [
        evaluate_result_item(item, gold_map)
        for item in result_items
    ]

    elapsed_values = [item["elapsed_seconds"] for item in reports]

    format_passed_count = sum(1 for item in reports if item["format_passed"])
    exactly_5_count = sum(1 for item in reports if item["has_exactly_5"])
    no_duplicate_count = sum(1 for item in reports if item["has_no_duplicates"])
    within_30_count = sum(1 for item in reports if item["within_30_seconds"])

    recall_values = [
        item["recall_at_5"]
        for item in reports
        if item["recall_at_5"] is not None
    ]

    mrr_values = [
        item["mrr"]
        for item in reports
        if item["mrr"] is not None
    ]

    return {
        "day": 13,
        "task": "Evaluation report for Day 12 retrieval results",
        "result_path": result_path,
        "qa_path": qa_path,
        "num_queries": len(reports),
        "format_passed_count": format_passed_count,
        "format_failed_count": len(reports) - format_passed_count,
        "all_format_passed": format_passed_count == len(reports),
        "exactly_5_count": exactly_5_count,
        "no_duplicate_count": no_duplicate_count,
        "within_30_seconds_count": within_30_count,
        "avg_elapsed_seconds": round(mean(elapsed_values), 4),
        "max_elapsed_seconds": round(max(elapsed_values), 4),
        "min_elapsed_seconds": round(min(elapsed_values), 4),
        "recall_at_5_available": len(recall_values) > 0,
        "avg_recall_at_5": round(mean(recall_values), 4) if recall_values else None,
        "mrr_available": len(mrr_values) > 0,
        "avg_mrr": round(mean(mrr_values), 4) if mrr_values else None,
        "gold_query_count": len(gold_map),
        "evaluated_with_gold_count": len(recall_values),
        "reports": reports,
    }


def save_json(data: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    report = evaluate_result_file(
        result_path=DEFAULT_RESULT_PATH,
        qa_path=DEFAULT_QA_PATH,
    )

    save_json(report, DEFAULT_OUTPUT_PATH)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n저장 완료: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()