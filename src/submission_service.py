"""Long-lived, offline inference service for the official Track 2 API."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from src.retrieval.answer_generator import generate_extractive_answer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.chunk_validator import validate_top5_chunk_ids
from src.retrieval.hybrid_retriever import HybridRetriever


class SubmissionService:
    def __init__(
        self,
        *,
        chunks_path: str | Path,
        bm25_index_path: str | Path,
        dense_index_path: str | Path,
        dense_metadata_path: str | Path,
        embedding_model_path: str,
    ):
        started = time.perf_counter()
        bm25_path = Path(bm25_index_path)
        self.bm25 = (
            BM25Retriever.load(bm25_path)
            if bm25_path.exists()
            else BM25Retriever.from_jsonl(str(chunks_path))
        )

        from src.retrieval.dense_retriever import DenseRetriever

        self.dense = DenseRetriever(
            model_name=embedding_model_path,
            offline=True,
            max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "64")),
            cpu_threads=int(os.environ.get("CPU_THREADS", "8")),
        )
        self.dense.load_index(str(dense_index_path), str(dense_metadata_path))
        self.valid_chunk_ids = {str(row["chunk_id"]) for row in self.bm25.chunks}
        self.retriever = HybridRetriever(
            self.bm25,
            self.dense,
            valid_chunk_ids=self.valid_chunk_ids,
        )
        self.default_results = self.bm25.chunks[:5]
        if len(self.default_results) != 5:
            raise ValueError("The corpus must contain at least five chunks.")
        self.lock = threading.Lock()
        self.startup_seconds = time.perf_counter() - started

    @classmethod
    def from_environment(cls) -> "SubmissionService":
        return cls(
            chunks_path=os.environ.get("CHUNKS_PATH", "data/chunks.jsonl"),
            bm25_index_path=os.environ.get("BM25_INDEX_PATH", "indexes/bm25.pkl"),
            dense_index_path=os.environ.get(
                "DENSE_INDEX_PATH", "indexes/dense_embeddings.npy"
            ),
            dense_metadata_path=os.environ.get(
                "DENSE_METADATA_PATH", "indexes/dense_chunks.jsonl"
            ),
            embedding_model_path=os.environ.get(
                "EMBEDDING_MODEL_PATH",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
        )

    def predict(self, request_id: str, question: str) -> dict[str, Any]:
        if not request_id.strip() or not question.strip():
            raise ValueError("Both id and question must be non-empty.")
        started = time.perf_counter()
        with self.lock:
            results = self.retriever.search(question, top_k=5)
        chunk_ids = [str(item["chunk_id"]) for item in results]
        validation = validate_top5_chunk_ids(chunk_ids, self.valid_chunk_ids)
        if not validation["passed"]:
            results = self.default_results
            chunk_ids = [str(item["chunk_id"]) for item in results]
        generated = generate_extractive_answer(question, results)
        return {
            "id": request_id,
            "retrieved_chunk_ids": chunk_ids,
            "answer": generated["answer"],
            "_elapsed_seconds": round(time.perf_counter() - started, 4),
        }
