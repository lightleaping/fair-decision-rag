# src/retrieval/day10_verify_results.py

import json
from pathlib import Path
from typing import Any, Dict, List


RESULT_FILES = [
    "outputs/results/day10_test_penalty.json",
    "outputs/results/day10_test_reasoning.json",
    "outputs/results/day10_test_fact.json",
    "outputs/results/day10_test_law.json",
    "outputs/results/day10_test_order.json",
    "outputs/results/day10_test_summary.json",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_result(result: Dict[str, Any]) -> Dict[str, Any]:
    top5_chunk_ids = result.get("top5_chunk_ids", [])

    has_exactly_5 = len(top5_chunk_ids) == 5
    has_no_duplicates = len(set(top5_chunk_ids)) == 5
    within_30_seconds = result.get("within_30_seconds") is True
    elapsed_seconds = result.get("elapsed_seconds")

    passed = has_exactly_5 and has_no_duplicates and within_30_seconds

    return {
        "query": result.get("query"),
        "query_type": result.get("query_type"),
        "top5_chunk_ids": top5_chunk_ids,
        "has_exactly_5": has_exactly_5,
        "has_no_duplicates": has_no_duplicates,
        "elapsed_seconds": elapsed_seconds,
        "within_30_seconds": within_30_seconds,
        "passed": passed,
    }


def main():
    reports: List[Dict[str, Any]] = []

    for file_path in RESULT_FILES:
        path = Path(file_path)

        if not path.exists():
            reports.append(
                {
                    "file": file_path,
                    "passed": False,
                    "error": "파일이 존재하지 않습니다.",
                }
            )
            continue

        result = load_json(file_path)
        report = verify_result(result)
        report["file"] = file_path
        reports.append(report)

    total = len(reports)
    passed_count = sum(1 for item in reports if item.get("passed") is True)
    failed_count = total - passed_count

    summary = {
        "day": "Day 10",
        "retriever": "BM25",
        "goal": "30초 이내 중복 없는 Top-5 chunk_id 반환",
        "total_tests": total,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "all_passed": failed_count == 0,
        "reports": reports,
    }

    output_path = Path("outputs/results/day10_verification_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n저장 완료: outputs/results/day10_verification_report.json")


if __name__ == "__main__":
    main()