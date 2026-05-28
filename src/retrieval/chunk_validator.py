# src/retrieval/chunk_validator.py

import json
from pathlib import Path
from typing import Any, Dict, List, Set


def load_valid_chunk_ids(chunk_jsonl_path: str = "data/chunks.jsonl") -> Set[str]:
    path = Path(chunk_jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"chunk 파일이 없습니다: {chunk_jsonl_path}")

    valid_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            chunk_id = item.get("chunk_id")

            if not chunk_id:
                raise ValueError(f"{line_idx}번째 줄에 chunk_id가 없습니다.")

            valid_ids.add(chunk_id)

    return valid_ids


def validate_top5_chunk_ids(
    top5_chunk_ids: List[str],
    valid_chunk_ids: Set[str],
) -> Dict[str, Any]:
    has_exactly_5 = len(top5_chunk_ids) == 5
    has_no_duplicates = len(set(top5_chunk_ids)) == 5
    all_exist = all(chunk_id in valid_chunk_ids for chunk_id in top5_chunk_ids)

    return {
        "top5_chunk_ids": top5_chunk_ids,
        "has_exactly_5": has_exactly_5,
        "has_no_duplicates": has_no_duplicates,
        "all_exist_in_public_chunks": all_exist,
        "passed": has_exactly_5 and has_no_duplicates and all_exist,
    }


def validate_result_file(
    result_path: str,
    chunk_jsonl_path: str = "data/chunks.jsonl",
) -> Dict[str, Any]:
    valid_chunk_ids = load_valid_chunk_ids(chunk_jsonl_path)

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    reports = []

    if "results" in result and isinstance(result["results"], list):
        items = result["results"]
    else:
        items = [result]

    for item in items:
        top5_chunk_ids = item.get("top5_chunk_ids", [])
        report = validate_top5_chunk_ids(top5_chunk_ids, valid_chunk_ids)
        report["query"] = item.get("query")
        report["query_type"] = item.get("query_type")
        report["elapsed_seconds"] = item.get("elapsed_seconds")
        report["within_30_seconds"] = item.get("within_30_seconds")
        reports.append(report)

    passed_count = sum(1 for item in reports if item["passed"])
    failed_count = len(reports) - passed_count

    return {
        "result_path": result_path,
        "chunk_jsonl_path": chunk_jsonl_path,
        "total": len(reports),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "all_passed": failed_count == 0,
        "reports": reports,
    }


def main():
    summary = validate_result_file(
        result_path="outputs/results/day10_bm25_top5.json",
        chunk_jsonl_path="data/chunks.jsonl",
    )

    output_path = Path("outputs/results/day11_chunk_validation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()