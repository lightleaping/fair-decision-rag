# src/retrieval/bm25_retriever.py

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """
    한국어/영어 혼합 문장을 단순 토큰화한다.

    현재는 빠른 baseline 구현이 목적이므로
    형태소 분석기 없이 공백 + 특수문자 기준으로 나눈다.
    """

    if not text:
        return []

    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text)
    tokens = text.split()

    return tokens


class BM25Retriever:
    """
    공정거래 의결서 chunk용 BM25 검색기.

    목적:
    - 외부 API 없이 동작
    - 모델 학습 없음
    - 빠른 baseline 검색
    - chunk_id 유지
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            raise ValueError("chunks가 비어 있습니다.")

        self.chunks = chunks
        self.tokenized_corpus = [
            simple_tokenize(chunk.get("text", ""))
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> "BM25Retriever":
        """
        JSONL chunk 파일에서 BM25Retriever를 생성한다.

        각 줄은 최소한 아래 key를 가져야 한다.

        {
          "chunk_id": "...",
          "text": "...",
          "section_type": "..."
        }
        """

        path = Path(jsonl_path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {jsonl_path}")

        chunks = []

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{line_idx}번째 줄 JSON 파싱 실패: {e}"
                    )

                if "chunk_id" not in item:
                    raise ValueError(f"{line_idx}번째 줄에 chunk_id가 없습니다.")

                if "text" not in item:
                    raise ValueError(f"{line_idx}번째 줄에 text가 없습니다.")

                chunks.append(item)

        return cls(chunks)

    def search(self, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        사용자 질문에 대해 BM25 검색을 수행한다.

        Args:
            query:
                사용자 질문

            top_n:
                반환할 검색 후보 개수.
                최종 Top-5를 안정적으로 만들기 위해 20개 이상 권장.

        Returns:
            BM25 검색 결과 리스트
        """

        query_tokens = simple_tokenize(query)

        if not query_tokens:
            raise ValueError("query가 비어 있거나 토큰화 결과가 없습니다.")

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_n]

        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])

            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "section_type": chunk.get("section_type", "default"),
                    "score": score,
                    "text": chunk.get("text", ""),
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title"),
                    "source_file": chunk.get("source_file"),
                    "page": chunk.get("page"),
                }
            )

        return results


if __name__ == "__main__":
    sample_chunks = [
        {
            "chunk_id": "case_001_chunk_001",
            "section_type": "fact",
            "text": "피심인은 거래상대방에게 특정 조건을 요구하였다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_002",
            "section_type": "legal_reasoning",
            "text": "이 행위는 거래상 지위를 이용한 불이익 제공행위로 판단된다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "text": "피심인에게 과징금 1억 원을 부과한다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_004",
            "section_type": "law_article",
            "text": "이 사건에는 공정거래법 제23조 제1항이 적용된다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_005",
            "section_type": "order",
            "text": "피심인은 향후 동일한 행위를 반복하여서는 아니 된다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
        {
            "chunk_id": "case_001_chunk_006",
            "section_type": "summary",
            "text": "이 사건은 거래상 지위 남용행위에 관한 건이다.",
            "doc_id": "case_001",
            "title": "샘플 사건"
        },
    ]

    retriever = BM25Retriever(sample_chunks)

    query = "이 사건 과징금은 얼마야?"
    results = retriever.search(query, top_n=5)

    for rank, item in enumerate(results, start=1):
        print("=" * 60)
        print("rank:", rank)
        print("chunk_id:", item["chunk_id"])
        print("section_type:", item["section_type"])
        print("score:", item["score"])
        print("text:", item["text"])