"""Create a transparent, reproducible QA set for local regression tests.

This is not a substitute for the hidden contest evaluation set.  Each sample is
anchored to one official chunk_id and is marked as ``silver``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def first_sentence(text: str, limit: int = 500) -> str:
    value = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)
    return parts[0][:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--output", default="data/processed/silver_qa_eval.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    selected: dict[str, list[dict]] = {}
    fallbacks: dict[str, dict] = {}
    with Path(args.chunks).open(encoding="utf-8") as stream:
        for line in stream:
            chunk = json.loads(line)
            doc_id = chunk["doc_id"]
            fallbacks.setdefault(doc_id, chunk)
            if chunk["section_type"] == "order":
                selected.setdefault(doc_id, []).append(chunk)

    rows = []
    for doc_id in sorted(fallbacks):
        gold_chunks = selected.get(doc_id) or [fallbacks[doc_id]]
        chunk = gold_chunks[0]
        title = chunk["title"]
        section = chunk["section_type"]
        if section == "order":
            query = f"{title} 사건에서 공정거래위원회가 내린 주문은 무엇인가요?"
        else:
            query = f"{title} 사건의 핵심 내용은 무엇인가요?"
        rows.append(
            {
                "query": query,
                "gold_chunk_ids": [item["chunk_id"] for item in gold_chunks],
                "answer_reference": first_sentence(chunk["text"]),
                "quality": "silver",
                "source": "official_public_decision_chunk",
            }
        )
        if len(rows) == args.limit:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
