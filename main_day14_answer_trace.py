# main_day14_answer_trace.py

import json
from pathlib import Path

from src.retrieval.answer_generator import generate_extractive_answer
from src.retrieval.evidence_trace import (
    build_evidence_trace,
    validate_evidence_trace,
)


INPUT_PATH = "outputs/results/day12_hybrid_top5.json"
OUTPUT_PATH = "outputs/results/day14_answer_trace.json"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        day12_result = json.load(f)

    traces = []

    for item in day12_result.get("results", []):
        query = item.get("query")
        top5_results = item.get("top5_results", [])

        answer_result = generate_extractive_answer(
            query=query,
            top5_results=top5_results,
        )

        trace = build_evidence_trace(
            query=query,
            top5_results=top5_results,
            answer_result=answer_result,
        )

        trace["trace_valid"] = validate_evidence_trace(trace)
        trace["retriever"] = item.get("retriever")
        trace["dense_available"] = item.get("dense_available")
        trace["elapsed_seconds"] = item.get("elapsed_seconds")

        traces.append(trace)

    valid_count = sum(1 for item in traces if item.get("trace_valid") is True)

    output = {
        "day": 14,
        "task": "Grounded extractive answer and evidence trace",
        "input_path": INPUT_PATH,
        "num_queries": len(traces),
        "valid_trace_count": valid_count,
        "invalid_trace_count": len(traces) - valid_count,
        "all_traces_valid": valid_count == len(traces),
        "external_api": False,
        "llm_generation": False,
        "answer_mode": "extractive_fallback",
        "results": traces,
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()