# src/retrieval/bm25_retriever.py

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """
    한국어/영어 혼합 문장을 단순 토큰화한다.

    Day 10 기준:
    - 외부 API 없음
    - 모델 학습 없음
    - 형태소 분석기 없이 빠르게 동작
    - BM25 baseline 목적

    추후 검색 품질 개선이 필요하면 형태소 분석기 또는 사용자 사전 기반 토큰화로 교체 가능하다.
    """

    if not text:
        return []

    text = str(text).lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text)
    tokens = text.split()

    return tokens


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    """
    다양한 chunk 구조에서 검색 대상 텍스트를 추출한다.

    지원 필드:
    - text: A 템플릿 표준 필드
    - chunk_text: 기존 B 코드 표준 필드
    - page_content: 원본 hybrid JSON 필드
    - content: 예비 호환 필드
    """

    return (
        chunk.get("text")
        or chunk.get("chunk_text")
        or chunk.get("page_content")
        or chunk.get("content")
        or ""
    )


def get_section_type(chunk: Dict[str, Any]) -> str:
    """
    다양한 chunk 구조에서 section_type을 추출한다.
    """

    metadata = chunk.get("metadata") or {}

    return (
        chunk.get("section_type")
        or metadata.get("section")
        or metadata.get("Header")
        or "default"
    )


def get_chunk_id(chunk: Dict[str, Any], idx: Optional[int] = None) -> str:
    """
    다양한 chunk 구조에서 chunk_id를 추출한다.

    원칙:
    - 가능한 경우 원본 공개본 chunk_id를 그대로 사용한다.
    - metadata.chunk_id가 있으면 그것을 사용한다.
    - 새 chunk_id 생성은 최후의 fallback이다.
    """

    metadata = chunk.get("metadata") or {}

    chunk_id = (
        chunk.get("chunk_id")
        or metadata.get("chunk_id")
    )

    if chunk_id:
        return str(chunk_id)

    raise ValueError(f"chunk_id가 없습니다. idx={idx}")


def normalize_chunk_for_bm25(chunk: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    BM25 검색기가 사용할 수 있는 표준 chunk 구조로 정규화한다.

    표준 필드:
    - chunk_id
    - section_type
    - text
    - chunk_text
    - doc_id
    - title
    - source_file
    - page
    """

    metadata = chunk.get("metadata") or {}

    chunk_id = get_chunk_id(chunk, idx)
    text = str(get_chunk_text(chunk)).strip()
    section_type = get_section_type(chunk)

    if "-CH-" in chunk_id:
        doc_id = chunk_id.split("-CH-")[0]
    else:
        doc_id = (
            chunk.get("doc_id")
            or chunk.get("document_id")
            or metadata.get("doc_id")
            or metadata.get("document_id")
        )

    title = (
        chunk.get("title")
        or metadata.get("title")
        or metadata.get("case_name")
        or metadata.get("source")
    )

    page = (
        chunk.get("page")
        or metadata.get("page")
        or metadata.get("page_number")
    )

    source_file = (
        chunk.get("source_file")
        or metadata.get("source_file")
        or metadata.get("source")
    )

    normalized = dict(chunk)
    normalized.update(
        {
            "chunk_id": chunk_id,
            "section_type": section_type,
            "text": text,
            "chunk_text": text,
            "doc_id": doc_id,
            "title": title,
            "source_file": source_file,
            "page": page,
        }
    )

    return normalized


class BM25Retriever:
    """
    공정거래 의결서 chunk용 BM25 검색기.

    목적:
    - 외부 API 없이 동작
    - 모델 학습 없음
    - 빠른 baseline 검색
    - 공개본 chunk_id 유지
    - Day 10 이후 A 템플릿과 호환
    - Day 9 이전 B 코드와도 최대한 호환
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            raise ValueError("chunks가 비어 있습니다.")

        self.chunks = [
            normalize_chunk_for_bm25(chunk, idx)
            for idx, chunk in enumerate(chunks)
        ]

        self.tokenized_corpus = [
            simple_tokenize(chunk.get("text", ""))
            for chunk in self.chunks
        ]

        if not any(self.tokenized_corpus):
            raise ValueError("BM25 corpus가 비어 있습니다. chunk text를 확인하세요.")

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> "BM25Retriever":
        """
        JSONL chunk 파일에서 BM25Retriever를 생성한다.

        권장 표준 형식:
        {
          "chunk_id": "...",
          "text": "...",
          "section_type": "..."
        }

        단, chunk_text, page_content도 호환한다.
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

                text = get_chunk_text(item)
                chunk_id = get_chunk_id(item, line_idx)

                if not chunk_id:
                    raise ValueError(f"{line_idx}번째 줄에 chunk_id가 없습니다.")

                if not text:
                    raise ValueError(f"{line_idx}번째 줄에 text/chunk_text/page_content가 없습니다.")

                chunks.append(item)

        return cls(chunks)

    @classmethod
    def from_json(cls, json_path: str) -> "BM25Retriever":
        """
        JSON 파일에서 BM25Retriever를 생성한다.

        지원 구조:
        1. [ {...}, {...} ]
        2. { "chunks": [ ... ] }
        3. { "data": [ ... ] }
        """

        path = Path(json_path)

        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            chunks = data
        elif isinstance(data, dict) and "chunks" in data:
            chunks = data["chunks"]
        elif isinstance(data, dict) and "data" in data:
            chunks = data["data"]
        else:
            raise ValueError(f"JSON 구조에서 chunk 리스트를 찾을 수 없습니다: {json_path}")

        return cls(chunks)

    @classmethod
    def from_path(cls, path: str) -> "BM25Retriever":
        """
        파일 확장자에 따라 JSONL 또는 JSON 로딩을 자동 선택한다.
        """

        file_path = Path(path)

        if file_path.suffix == ".jsonl":
            return cls.from_jsonl(path)

        if file_path.suffix == ".json":
            return cls.from_json(path)

        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_path.suffix}")

    def search(
        self,
        query: str,
        top_n: int = 20,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        사용자 질문에 대해 BM25 검색을 수행한다.

        Args:
            query:
                사용자 질문

            top_n:
                반환할 검색 후보 개수.
                A 템플릿 기준 인자명.

            top_k:
                기존 B 코드 호환용 인자명.
                top_k가 들어오면 top_n보다 우선 사용한다.

        Returns:
            BM25 검색 결과 리스트
        """

        if top_k is not None:
            top_n = top_k

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
            text = chunk.get("text", "")

            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "section_type": chunk.get("section_type", "default"),
                    "score": score,
                    "retriever": "bm25",

                    # A 템플릿 표준
                    "text": text,
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title"),
                    "source_file": chunk.get("source_file"),
                    "page": chunk.get("page"),

                    # 기존 B 코드 호환
                    "chunk_text": text,
                    "preview": text[:250],
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
            "title": "샘플 사건",
        },
        {
            "chunk_id": "case_001_chunk_002",
            "section_type": "legal_reasoning",
            "text": "이 행위는 거래상 지위를 이용한 불이익 제공행위로 판단된다.",
            "doc_id": "case_001",
            "title": "샘플 사건",
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "text": "피심인에게 과징금 1억 원을 부과한다.",
            "doc_id": "case_001",
            "title": "샘플 사건",
        },
        {
            "chunk_id": "case_001_chunk_004",
            "section_type": "law_article",
            "text": "이 사건에는 공정거래법 제23조 제1항이 적용된다.",
            "doc_id": "case_001",
            "title": "샘플 사건",
        },
        {
            "chunk_id": "case_001_chunk_005",
            "section_type": "order",
            "text": "피심인은 향후 동일한 행위를 반복하여서는 아니 된다.",
            "doc_id": "case_001",
            "title": "샘플 사건",
        },
        {
            "chunk_id": "case_001_chunk_006",
            "section_type": "summary",
            "text": "이 사건은 거래상 지위 남용행위에 관한 건이다.",
            "doc_id": "case_001",
            "title": "샘플 사건",
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