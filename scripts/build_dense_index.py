"""Build the offline sentence-embedding index."""

from __future__ import annotations

import argparse
import json
import time

from src.retrieval.dense_retriever import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--index", default="indexes/dense_embeddings.npy")
    parser.add_argument("--metadata", default="indexes/dense_chunks.jsonl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    retriever = DenseRetriever(
        args.model,
        offline=True,
        max_seq_length=args.max_seq_length,
    )
    retriever.build_index(args.chunks, batch_size=args.batch_size)
    retriever.save_index(args.index, args.metadata)
    print(
        json.dumps(
            {
                "chunks": len(retriever.chunks),
                "dimensions": int(retriever.embeddings.shape[1]),
                "max_seq_length": args.max_seq_length,
                "index": args.index,
                "metadata": args.metadata,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
