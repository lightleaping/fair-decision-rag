"""Official-style retrieval and generation metrics."""

from __future__ import annotations

import json
import re
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def recall_at_k(predicted: List[str], gold: List[str], k: int = 5) -> Optional[float]:
    if not gold:
        return None
    return len(set(predicted[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(predicted: List[str], gold: List[str]) -> Optional[float]:
    if not gold:
        return None
    gold_set = set(gold)
    return next(
        (1.0 / rank for rank, chunk_id in enumerate(predicted, 1) if chunk_id in gold_set),
        0.0,
    )


def answer_tokens(text: str) -> List[str]:
    return re.findall(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?", str(text).lower())


def token_f1(prediction: str, reference: str) -> Optional[float]:
    predicted = Counter(answer_tokens(prediction))
    gold = Counter(answer_tokens(reference))
    if not gold:
        return None
    overlap = sum((predicted & gold).values())
    if not predicted or overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def evaluate_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    reports = []
    for row in rows:
        predicted = row.get("top5_chunk_ids", [])
        gold = row.get("gold_chunk_ids", [])
        reports.append(
            {
                "query": row.get("query"),
                "recall_at_5": recall_at_k(predicted, gold),
                "mrr": reciprocal_rank(predicted, gold),
                "token_f1": token_f1(
                    row.get("answer", ""), row.get("answer_reference", "")
                ),
                "top5_chunk_ids": predicted,
                "gold_chunk_ids": gold,
            }
        )
    retrieval_recall = [x["recall_at_5"] for x in reports if x["recall_at_5"] is not None]
    retrieval_mrr = [x["mrr"] for x in reports if x["mrr"] is not None]
    generation_f1 = [x["token_f1"] for x in reports if x["token_f1"] is not None]
    return {
        "num_queries": len(reports),
        "avg_recall_at_5": mean(retrieval_recall) if retrieval_recall else None,
        "avg_mrr": mean(retrieval_mrr) if retrieval_mrr else None,
        "avg_token_f1": mean(generation_f1) if generation_f1 else None,
        "bertscore": None,
        "bertscore_note": "선택 의존성 및 reference answer가 있을 때 별도 산출",
        "reports": reports,
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
