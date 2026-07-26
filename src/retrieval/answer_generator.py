"""Deterministic grounded answer generation for offline evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.retrieval.bm25_retriever import simple_tokenize


SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def clean_text(text: str) -> str:
    return " ".join(str(text or "").split())


def generate_extractive_answer(
    query: str,
    top5_results: List[Dict[str, Any]],
    *,
    max_sentences: int = 3,
    max_chars: int = 900,
) -> Dict[str, Any]:
    query_terms = set(simple_tokenize(query))
    candidates = []
    for rank, result in enumerate(top5_results, 1):
        chunk_id = result.get("chunk_id")
        for sentence_index, sentence in enumerate(
            SENTENCE_SPLIT.split(result.get("text", ""))
        ):
            sentence = clean_text(sentence)
            if len(sentence) < 15:
                continue
            terms = set(simple_tokenize(sentence))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            score = overlap + 0.15 / rank - 0.00005 * len(sentence)
            candidates.append(
                (score, rank, sentence_index, chunk_id, result, sentence)
            )
    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = []
    seen_sentences = set()
    total_chars = 0
    for _, _, _, chunk_id, result, sentence in candidates:
        signature = sentence[:100]
        if signature in seen_sentences or total_chars + len(sentence) > max_chars:
            continue
        selected.append(
            {
                "chunk_id": chunk_id,
                "section_type": result.get("section_type", "default"),
                "text": sentence,
            }
        )
        seen_sentences.add(signature)
        total_chars += len(sentence)
        if len(selected) == max_sentences:
            break
    if not selected:
        return {
            "answer": "검색된 공개 의결서 근거만으로는 답변을 확인할 수 없습니다.",
            "answer_type": "grounded_refusal",
            "evidence_chunk_ids": [],
            "evidences": [],
        }
    answer = " ".join(
        f"{evidence['text']} [{evidence['chunk_id']}]" for evidence in selected
    )
    return {
        "answer": answer,
        "answer_type": "grounded_extractive",
        "evidence_chunk_ids": list(
            dict.fromkeys(evidence["chunk_id"] for evidence in selected)
        ),
        "evidences": selected,
    }
