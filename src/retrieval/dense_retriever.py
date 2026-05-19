# src/retrieval/dense_retriever.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks_jsonl(chunk_jsonl_path: str) -> List[Dict[str, Any]]:
    path = Path(chunk_jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"chunk 파일을 찾을 수 없습니다: {chunk_jsonl_path}")

    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if "chunk_id" not in item:
                raise ValueError(f"{line_idx}번째 줄에 chunk_id가 없습니다.")

            if "text" not in item:
                raise ValueError(f"{line_idx}번째 줄에 text가 없습니다.")

            chunks.append(item)

    if not chunks:
        raise ValueError("chunks.jsonl에서 chunk를 하나도 읽지 못했습니다.")

    return chunks


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class DenseRetriever:
    """
    1GB 이하 경량 임베딩 모델 기반 Dense Retrieval.

    특징:
    - 외부 API 호출 없음
    - 모델 학습 없음
    - chunk_id 유지
    - cosine similarity 기반 검색
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(
        self,
        chunk_jsonl_path: str,
        batch_size: int = 32,
    ) -> None:
        """
        data/chunks.jsonl을 읽고 chunk embedding index를 생성한다.
        """

        self.chunks = load_chunks_jsonl(chunk_jsonl_path)

        texts = [chunk.get("text", "") for chunk in self.chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        embeddings = embeddings.astype("float32")
        embeddings = l2_normalize(embeddings)

        self.embeddings = embeddings

    def save_index(
        self,
        index_path: str = "indexes/dense_embeddings.npy",
        meta_path: str = "indexes/dense_chunks.jsonl",
    ) -> None:
        """
        embedding matrix와 chunk metadata를 저장한다.
        """

        if self.embeddings is None:
            raise ValueError("저장할 embeddings가 없습니다. build_index를 먼저 실행하세요.")

        index_output = Path(index_path)
        meta_output = Path(meta_path)

        index_output.parent.mkdir(parents=True, exist_ok=True)
        meta_output.parent.mkdir(parents=True, exist_ok=True)

        np.save(index_output, self.embeddings)

        with open(meta_output, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def load_index(
        self,
        index_path: str = "indexes/dense_embeddings.npy",
        meta_path: str = "indexes/dense_chunks.jsonl",
    ) -> None:
        """
        저장된 embedding matrix와 chunk metadata를 불러온다.
        """

        index_file = Path(index_path)
        meta_file = Path(meta_path)

        if not index_file.exists():
            raise FileNotFoundError(f"Dense index 파일이 없습니다: {index_path}")

        if not meta_file.exists():
            raise FileNotFoundError(f"Dense metadata 파일이 없습니다: {meta_path}")

        self.embeddings = np.load(index_file).astype("float32")
        self.chunks = load_chunks_jsonl(meta_path)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                f"chunk 수와 embedding 수가 다릅니다. "
                f"chunks={len(self.chunks)}, embeddings={len(self.embeddings)}"
            )

    def search(
        self,
        query: str,
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        질문을 dense embedding으로 변환하고 cosine similarity 기준으로 검색한다.
        """

        if self.embeddings is None or not self.chunks:
            raise ValueError("Dense index가 로드되지 않았습니다.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        query_embedding = l2_normalize(query_embedding)

        scores = np.dot(self.embeddings, query_embedding[0])

        ranked_indices = np.argsort(scores)[::-1][:top_n]

        results = []

        for idx in ranked_indices:
            chunk = self.chunks[int(idx)]

            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "section_type": chunk.get("section_type", "default"),
                    "score": float(scores[int(idx)]),
                    "text": chunk.get("text", ""),
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title"),
                    "source_file": chunk.get("source_file"),
                    "page": chunk.get("page"),
                }
            )

        return results


if __name__ == "__main__":
    retriever = DenseRetriever()

    chunk_path = "data/chunks.jsonl"

    retriever.build_index(
        chunk_jsonl_path=chunk_path,
        batch_size=32,
    )

    retriever.save_index(
        index_path="indexes/dense_embeddings.npy",
        meta_path="indexes/dense_chunks.jsonl",
    )

    results = retriever.search(
        query="이 사건 과징금은 얼마야?",
        top_n=5,
    )

    for rank, item in enumerate(results, start=1):
        print("=" * 80)
        print("rank:", rank)
        print("chunk_id:", item["chunk_id"])
        print("section_type:", item["section_type"])
        print("score:", item["score"])
        print("text:", item["text"][:150])