"""Single entrypoint for the offline Track 2 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.retrieval.answer_generator import generate_extractive_answer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.chunk_validator import load_valid_chunk_ids
from src.retrieval.evidence_trace import build_evidence_trace, validate_evidence_trace
from src.retrieval.hybrid_retriever import HybridRetriever


def load_dense(args):
    if args.mode == "bm25":
        return None
    from src.retrieval.dense_retriever import DenseRetriever

    dense = DenseRetriever(model_name=args.embedding_model, offline=True)
    index = Path(args.dense_index)
    metadata = Path(args.dense_metadata)
    if args.build_dense or not (index.exists() and metadata.exists()):
        dense.build_index(args.chunks, batch_size=args.batch_size)
        dense.save_index(str(index), str(metadata))
    else:
        dense.load_index(str(index), str(metadata))
    return dense


def main() -> None:
    parser = argparse.ArgumentParser(description="공정거래 의결서 근거 기반 QA")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["bm25", "hybrid"], default="hybrid")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--build-dense", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "EMBEDDING_MODEL_PATH",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )
    parser.add_argument("--dense-index", default="indexes/dense_embeddings.npy")
    parser.add_argument("--dense-metadata", default="indexes/dense_chunks.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()

    started = time.perf_counter()
    bm25 = BM25Retriever.from_jsonl(args.chunks)
    dense = load_dense(args)
    retriever = HybridRetriever(
        bm25,
        dense,
        valid_chunk_ids=load_valid_chunk_ids(args.chunks),
    )
    results = retriever.search(args.query)
    answer = generate_extractive_answer(args.query, results)
    trace = build_evidence_trace(args.query, results, answer)
    trace["trace_valid"] = validate_evidence_trace(trace)
    trace["mode"] = args.mode
    trace["dense_available"] = dense is not None
    trace["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    rendered = json.dumps(trace, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
