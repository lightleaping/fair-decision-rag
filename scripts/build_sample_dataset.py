"""Create a deterministic document-level sample and its matching dense index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_document_order(doc_ids: set[str], seed: str) -> list[str]:
    return sorted(
        doc_ids,
        key=lambda doc_id: hashlib.sha256(f"{seed}:{doc_id}".encode()).digest(),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=int, default=50)
    parser.add_argument("--seed", default="fair-decision-rag-track2")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--dense-index", default="indexes/dense_embeddings.npy")
    parser.add_argument("--dense-metadata", default="indexes/dense_chunks.jsonl")
    parser.add_argument("--output-chunks", default="data/sample/chunks.jsonl")
    parser.add_argument(
        "--output-index", default="indexes/sample/dense_embeddings.npy"
    )
    parser.add_argument(
        "--output-metadata", default="indexes/sample/dense_chunks.jsonl"
    )
    args = parser.parse_args()

    chunks = read_jsonl(Path(args.chunks))
    metadata = read_jsonl(Path(args.dense_metadata))
    embeddings = np.load(args.dense_index, mmap_mode="r")

    if len(chunks) != len(metadata) or len(metadata) != len(embeddings):
        raise ValueError("Chunk, dense metadata, and embedding row counts differ.")
    for row, dense_row in zip(chunks, metadata):
        if row["chunk_id"] != dense_row["chunk_id"]:
            raise ValueError(f"Dense index is misaligned at {row['chunk_id']}.")

    doc_ids = {str(row["doc_id"]) for row in chunks}
    if not 1 <= args.documents <= len(doc_ids):
        raise ValueError(
            f"--documents must be between 1 and {len(doc_ids)} (got {args.documents})."
        )
    selected_docs = set(stable_document_order(doc_ids, args.seed)[: args.documents])
    selected_indices = [
        index for index, row in enumerate(chunks) if row["doc_id"] in selected_docs
    ]
    sample_chunks = [chunks[index] for index in selected_indices]

    write_jsonl(Path(args.output_chunks), sample_chunks)
    write_jsonl(Path(args.output_metadata), sample_chunks)
    output_index = Path(args.output_index)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_index, np.asarray(embeddings[selected_indices], dtype="float32"))

    print(
        json.dumps(
            {
                "documents": len(selected_docs),
                "chunks": len(sample_chunks),
                "dimensions": int(embeddings.shape[1]),
                "section_types": sorted(
                    {str(row.get("section_type", "")) for row in sample_chunks}
                ),
                "chunks_path": str(Path(args.output_chunks)),
                "index_path": str(output_index),
                "metadata_path": str(Path(args.output_metadata)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
