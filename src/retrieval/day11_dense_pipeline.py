# src/retrieval/day11_dense_pipeline.py

import json
import time
from pathlib import Path
from typing import Any, Dict, Set

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.day10_a_runner import run_day10_a, save_result_json


def load_valid_chunk_ids(jsonl_path: str) -> Set[str]:
    valid_chunk_ids = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            chunk_id = item.get("chunk_id")

            if not chunk_id:
                raise ValueError(f"{line_idx}번째 줄에 chunk_id가 없습니다.")

            valid_chunk_ids.add(chunk_id)

    return valid_chunk_ids


def ensure_dense_index(
    retriever: DenseRetriever,
    chunk_jsonl_path: str,
    index_path: str = "indexes/dense_embeddings.npy",
    meta_path: str = "indexes/dense_chunks.jsonl",
) -> None:
    if Path(index_path).exists() and Path(meta_path).exists():
        print("[INFO] 기존 Dense index를 로드합니다.")
        retriever.load_index(
            index_path=index_path,
            meta_path=meta_path,
        )
        return

    print("[INFO] Dense index가 없어 새로 생성합니다.")
    retriever.build_index(
        chunk_jsonl_path=chunk_jsonl_path,
        batch_size=32,
    )

    retriever.save_index(
        index_path=index_path,
        meta_path=meta_path,
    )

    print("[INFO] Dense index 저장 완료")


def run_dense_day11_pipeline(
    query: str,
    chunk_jsonl_path: str = "data/chunks.jsonl",
    output_path: str = "outputs/results/day11_dense_top5.json",
    dense_top_n: int = 20,
    max_seconds: float = 30.0,
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    retriever = DenseRetriever()

    ensure_dense_index(
        retriever=retriever,
        chunk_jsonl_path=chunk_jsonl_path,
        index_path="indexes/dense_embeddings.npy",
        meta_path="indexes/dense_chunks.jsonl",
    )

    valid_chunk_ids = load_valid_chunk_ids(chunk_jsonl_path)

    dense_results = retriever.search(
        query=query,
        top_n=dense_top_n,
    )

    final_result = run_day10_a(
        query=query,
        search_results=dense_results,
        valid_chunk_ids=valid_chunk_ids,
    )

    elapsed_seconds = time.perf_counter() - start_time

    final_result["retriever"] = "dense"
    final_result["dense_top_n"] = dense_top_n
    final_result["embedding_model"] = retriever.model_name
    final_result["elapsed_seconds"] = round(elapsed_seconds, 4)
    final_result["within_30_seconds"] = elapsed_seconds <= max_seconds

    if elapsed_seconds > max_seconds:
        raise TimeoutError(
            f"응답 시간이 {elapsed_seconds:.2f}초로 {max_seconds:.2f}초 제한을 초과했습니다."
        )

    save_result_json(
        result=final_result,
        output_path=output_path,
    )

    return final_result


if __name__ == "__main__":
    result = run_dense_day11_pipeline(
        query="이 사건 과징금은 얼마야?",
        chunk_jsonl_path="data/chunks.jsonl",
        output_path="outputs/results/day11_dense_top5.json",
        dense_top_n=20,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n저장 완료: outputs/results/day11_dense_top5.json")