# src/retrieval/hybrid_retriever.py

from typing import Any, Dict, List, Optional, Set

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.score_fusion import fuse_bm25_dense


class HybridRetriever:
    """
    BM25 + Dense Hybrid Retriever.

    Day 12 기준:
    - BM25 결과와 Dense 결과를 결합할 수 있는 인터페이스 제공
    - Dense가 아직 연결되지 않은 경우 BM25-only fallback으로 동작
    - 정확히 Top-5 반환
    - 중복 chunk_id 제거
    - 공개본 데이터에 존재하는 chunk_id만 반환 가능
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: Optional[Any] = None,
        valid_chunk_ids: Optional[Set[str]] = None,
    ):
        if bm25_retriever is None:
            raise ValueError("bm25_retriever는 필수입니다.")

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.valid_chunk_ids = valid_chunk_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid 검색을 수행한다.

        Args:
            query:
                사용자 질문

            top_k:
                최종 반환할 chunk 개수.
                공모전 기준 기본값은 5.

            candidate_k:
                BM25/Dense에서 각각 가져올 후보 개수.

            bm25_weight:
                BM25 정규화 점수 가중치.

            dense_weight:
                Dense 정규화 점수 가중치.

        Returns:
            정확히 top_k개의 검색 결과.
        """

        if not query or not query.strip():
            raise ValueError("query가 비어 있습니다.")

        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        if candidate_k < top_k:
            raise ValueError("candidate_k는 top_k 이상이어야 합니다.")

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_n=candidate_k,
        )

        dense_available = self.dense_retriever is not None

        if dense_available:
            dense_results = self.dense_retriever.search(
                query=query,
                top_n=candidate_k,
            )
        else:
            dense_results = []
            bm25_weight = 1.0
            dense_weight = 0.0

        fused_results = fuse_bm25_dense(
            bm25_results=bm25_results,
            dense_results=dense_results,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            valid_chunk_ids=self.valid_chunk_ids,
            top_k=top_k,
        )

        for item in fused_results:
            item["dense_available"] = dense_available
            item["candidate_k"] = candidate_k

        chunk_ids = [item["chunk_id"] for item in fused_results]

        assert len(chunk_ids) == top_k
        assert len(set(chunk_ids)) == top_k

        return fused_results