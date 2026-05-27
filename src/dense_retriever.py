# src/dense_retriever.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DenseSearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: Dict


class SimpleDenseRetriever:
    """
    Day 11-A Dense Retrieval baseline.

    현재 버전은 외부 API 없이 동작하는 안전한 baseline이다.
    sentence-transformers 모델이 아직 준비되지 않은 상황에서도
    TF-IDF/코사인 유사도 기반으로 Dense Retriever 인터페이스를 먼저 고정한다.

    이후 1GB 이하 임베딩 모델이 준비되면 _vectorize 부분을 embedding encode로 교체하면 된다.
    """

    def __init__(self, chunk_path: str | Path):
        self.chunk_path = Path(chunk_path)
        self.chunks = self._load_chunks(self.chunk_path)
        self.documents = [self._get_text(chunk) for chunk in self.chunks]
        self.vocab = self._build_vocab(self.documents)
        self.doc_vectors = [self._vectorize(text) for text in self.documents]

    def search(self, query: str, top_k: int = 5) -> List[DenseSearchResult]:
        query_vector = self._vectorize(query)

        scored = []
        for chunk, doc_vector in zip(self.chunks, self.doc_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[DenseSearchResult] = []
        seen_chunk_ids = set()

        for score, chunk in scored:
            chunk_id = self._get_chunk_id(chunk)

            if not chunk_id:
                continue

            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)

            results.append(
                DenseSearchResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=self._get_text(chunk),
                    metadata=self._get_metadata(chunk),
                )
            )

            if len(results) == top_k:
                break

        if len(results) != top_k:
            raise ValueError(
                f"Dense search must return exactly {top_k} unique chunk_ids, "
                f"but got {len(results)}."
            )

        return results

    def _load_chunks(self, path: Path) -> List[Dict]:
        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        if path.suffix == ".jsonl":
            chunks = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
            return chunks

        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                for key in ["chunks", "data", "items", "documents"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]

            raise ValueError("Unsupported JSON chunk structure.")

        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _build_vocab(self, documents: List[str]) -> Dict[str, int]:
        vocab = {}

        for document in documents:
            for token in self._tokenize(document):
                if token not in vocab:
                    vocab[token] = len(vocab)

        return vocab

    def _vectorize(self, text: str) -> Dict[int, float]:
        tokens = self._tokenize(text)
        vector: Dict[int, float] = {}

        for token in tokens:
            if token not in self.vocab:
                continue

            index = self.vocab[token]
            vector[index] = vector.get(index, 0.0) + 1.0

        norm = math.sqrt(sum(value * value for value in vector.values()))

        if norm == 0:
            return vector

        return {index: value / norm for index, value in vector.items()}

    def _cosine_similarity(
        self,
        query_vector: Dict[int, float],
        doc_vector: Dict[int, float],
    ) -> float:
        if not query_vector or not doc_vector:
            return 0.0

        if len(query_vector) > len(doc_vector):
            query_vector, doc_vector = doc_vector, query_vector

        return sum(
            value * doc_vector.get(index, 0.0)
            for index, value in query_vector.items()
        )

    def _tokenize(self, text: str) -> List[str]:
        cleaned = (
            text.lower()
            .replace("\n", " ")
            .replace("\t", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("?", " ")
            .replace("!", " ")
            .replace("·", " ")
            .replace("ㆍ", " ")
            .replace("「", " ")
            .replace("」", " ")
            .replace("『", " ")
            .replace("』", " ")
        )

        return [token.strip() for token in cleaned.split() if len(token.strip()) >= 2]

    def _get_chunk_id(self, chunk: Dict) -> Optional[str]:
        for key in ["chunk_id", "id", "chunkId"]:
            value = chunk.get(key)
            if value:
                return str(value)

        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            for key in ["chunk_id", "id", "chunkId"]:
                value = metadata.get(key)
                if value:
                    return str(value)

        return None

    def _get_text(self, chunk: Dict) -> str:
        for key in ["text", "content", "chunk_text", "body"]:
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return json.dumps(chunk, ensure_ascii=False)

    def _get_metadata(self, chunk: Dict) -> Dict:
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            return metadata

        return {
            key: value
            for key, value in chunk.items()
            if key not in {"text", "content", "chunk_text", "body"}
        }


def dense_search(
    chunk_path: str | Path,
    query: str,
    top_k: int = 5,
) -> List[DenseSearchResult]:
    retriever = SimpleDenseRetriever(chunk_path)
    return retriever.search(query=query, top_k=top_k)


if __name__ == "__main__":
    chunk_file = "data/chunks.jsonl"

    query = "한국파파존스 사건에서 어떤 법 위반이 있었어?"

    retriever = SimpleDenseRetriever(chunk_file)
    results = retriever.search(query=query, top_k=5)

    for index, result in enumerate(results, start=1):
        print("=" * 80)
        print("rank:", index)
        print("chunk_id:", result.chunk_id)
        print("score:", result.score)
        print("text:", result.text[:300])