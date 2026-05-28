# src/retrieval/evidence_trace.py

from typing import Any, Dict, List


def build_evidence_trace(
    query: str,
    top5_results: List[Dict[str, Any]],
    answer_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    질문, Top-5 검색 결과, 생성 답변을 하나의 evidence trace로 묶는다.
    """

    top5_chunk_ids = [
        item.get("chunk_id")
        for item in top5_results
        if item.get("chunk_id")
    ]

    evidence_chunk_ids = answer_result.get("evidence_chunk_ids", [])

    return {
        "query": query,
        "top5_chunk_ids": top5_chunk_ids,
        "answer": answer_result.get("answer"),
        "answer_type": answer_result.get("answer_type"),
        "evidence_chunk_ids": evidence_chunk_ids,
        "evidences": answer_result.get("evidences", []),
        "trace_rules": {
            "answer_uses_only_top5_chunks": all(
                chunk_id in top5_chunk_ids
                for chunk_id in evidence_chunk_ids
            ),
            "has_evidence": len(evidence_chunk_ids) > 0,
            "external_api": False,
            "llm_generation": False,
        },
    }


def validate_evidence_trace(trace: Dict[str, Any]) -> bool:
    rules = trace.get("trace_rules", {})

    return (
        rules.get("answer_uses_only_top5_chunks") is True
        and rules.get("has_evidence") is True
        and rules.get("external_api") is False
    )