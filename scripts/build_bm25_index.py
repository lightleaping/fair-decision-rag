"""Build a persisted BM25 index for fast API-server startup."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--output", default="indexes/bm25.pkl")
    args = parser.parse_args()
    started = time.perf_counter()
    retriever = BM25Retriever.from_jsonl(args.chunks)
    retriever.save(args.output)
    print(
        json.dumps(
            {
                "chunks": len(retriever.chunks),
                "output": str(Path(args.output)),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
