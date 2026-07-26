"""Query-aware BM25 + dense retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.query_classifier import QueryClassifier
from src.retrieval.score_fusion import fuse_bm25_dense
from src.retrieval.section_boost import SectionBooster


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: Optional[Any] = None,
        valid_chunk_ids: Optional[Set[str]] = None,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.valid_chunk_ids = valid_chunk_ids
        self.classifier = QueryClassifier()
        self.booster = SectionBooster()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 500,
        bm25_weight: float = 0.70,
        dense_weight: float = 0.30,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError("질문이 비어 있습니다.")
        if candidate_k < top_k:
            raise ValueError("candidate_k는 top_k 이상이어야 합니다.")

        classification = self.classifier.classify(query)
        bm25 = self.bm25_retriever.search(query, top_n=candidate_k)
        dense_available = self.dense_retriever is not None
        dense = (
            self.dense_retriever.search(query, top_n=candidate_k)
            if dense_available
            else []
        )
        if not dense_available:
            bm25_weight, dense_weight = 1.0, 0.0

        # Keep a wider fused pool so section awareness can affect the final top 5.
        fused = fuse_bm25_dense(
            bm25,
            dense,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            valid_chunk_ids=self.valid_chunk_ids,
            top_k=min(candidate_k, len({x["chunk_id"] for x in bm25 + dense})),
        )
        boosted = self.booster.apply_boost(
            fused,
            classification.section_priority,
            score_key="score",
        )
        selected = boosted[:top_k]
        if len(selected) != top_k:
            raise ValueError(f"Top-{top_k} 근거를 구성하지 못했습니다.")
        for item in selected:
            item["query_type"] = classification.query_type
            item["matched_keywords"] = classification.matched_keywords
            item["dense_available"] = dense_available
        return selected
