# src/retrieval/answer_generator.py

from typing import Any, Dict, List


def clean_text(text: str, max_length: int = 500) -> str:
    if not text:
        return ""

    text = str(text).replace("\n", " ").strip()
    text = " ".join(text.split())

    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."

    return text


def select_evidence_sentences(
    top5_results: List[Dict[str, Any]],
    max_evidence_count: int = 3,
    max_text_length: int = 500,
) -> List[Dict[str, Any]]:
    evidences = []

    for item in top5_results[:max_evidence_count]:
        chunk_id = item.get("chunk_id")
        text = item.get("text") or item.get("chunk_text") or item.get("preview") or ""

        if not chunk_id or not text:
            continue

        evidences.append(
            {
                "chunk_id": chunk_id,
                "section_type": item.get("section_type", "default"),
                "score": item.get("score"),
                "text": clean_text(text, max_length=max_text_length),
            }
        )

    return evidences


def generate_extractive_answer(
    query: str,
    top5_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Top-5 chunk 기반 extractive answer 생성기.

    원칙:
    - 외부 API 호출 없음
    - 검색된 chunk 외 정보 사용 금지
    - 답변은 근거 chunk에서 추출한 내용 중심
    - 명확하지 않으면 보수적으로 답변
    """

    evidences = select_evidence_sentences(top5_results)

    if not evidences:
        return {
            "answer": "검색된 근거만으로는 명확한 답변을 생성하기 어렵습니다.",
            "answer_type": "extractive_fallback",
            "evidence_chunk_ids": [],
            "evidences": [],
        }

    evidence_chunk_ids = [
        evidence["chunk_id"]
        for evidence in evidences
    ]

    first_evidence = evidences[0]["text"]

    answer = (
        "검색된 의결서 근거에 따르면, 다음 내용이 질문과 가장 관련성이 높습니다. "
        f"{first_evidence} "
        "다만 이 답변은 검색된 Top-5 chunk에 근거한 요약이며, "
        "최종 판단은 표시된 근거 chunk를 함께 확인해야 합니다."
    )

    return {
        "answer": answer,
        "answer_type": "extractive_fallback",
        "evidence_chunk_ids": evidence_chunk_ids,
        "evidences": evidences,
    }


if __name__ == "__main__":
    sample_top5 = [
        {
            "chunk_id": "DOC-sample-CH-001",
            "section_type": "penalty",
            "score": 1.0,
            "text": "피심인에게 과징금 1억 원을 부과한다.",
        },
        {
            "chunk_id": "DOC-sample-CH-002",
            "section_type": "legal_reasoning",
            "score": 0.8,
            "text": "해당 행위는 공정거래법 위반으로 판단된다.",
        },
    ]

    result = generate_extractive_answer(
        query="과징금은 얼마인가요?",
        top5_results=sample_top5,
    )

    print(result)