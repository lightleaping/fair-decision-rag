from typing import Any, Dict, List


def build_evidence_trace(
    query: str,
    top5_results: List[Dict[str, Any]],
    answer_result: Dict[str, Any],
) -> Dict[str, Any]:
    top5_ids = [item["chunk_id"] for item in top5_results]
    evidence_ids = answer_result.get("evidence_chunk_ids", [])
    return {
        "query": query,
        "top5_chunk_ids": top5_ids,
        **answer_result,
        "trace_rules": {
            "answer_uses_only_top5_chunks": set(evidence_ids) <= set(top5_ids),
            "has_evidence": bool(evidence_ids),
            "external_api": False,
            "generation_mode": answer_result.get("answer_type"),
        },
    }


def validate_evidence_trace(trace: Dict[str, Any]) -> bool:
    rules = trace["trace_rules"]
    return bool(rules["answer_uses_only_top5_chunks"] and rules["has_evidence"])
