# src/hybrid_retriever.py

from src.query_classifier import classify_query


class HybridRetriever:
    """
    Day 9용 Section-aware Hybrid Retriever.

    현재 Dense Retrieval이 실제 구현 전일 수 있으므로,
    BM25-only fallback을 지원한다.

    흐름:
    query
    -> 질문 유형 분류
    -> priority section 결정
    -> BM25 검색
    -> Dense 검색 시도
    -> 점수 정규화
    -> BM25/Dense 점수 결합
    -> section-aware boost
    -> Top-5 반환
    """

    def __init__(
        self,
        chunks: list[dict],
        bm25_retriever,
        dense_retriever=None,
        section_boost_ratio: float = 0.10,
    ):
        self.chunks = chunks
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.section_boost_ratio = section_boost_ratio
        self.valid_chunk_ids = {
            chunk["chunk_id"]
            for chunk in chunks
            if chunk.get("chunk_id")
        }

    def search(self, query: str, top_k: int = 5, candidate_k: int = 10) -> dict:
        classification = classify_query(query)

        bm25_results = self._safe_bm25_search(query, candidate_k)
        dense_results = self._safe_dense_search(query, candidate_k)

        merged_candidates = self._merge_results(
            bm25_results=bm25_results,
            dense_results=dense_results,
            bm25_weight=classification.bm25_weight,
            dense_weight=classification.dense_weight,
        )

        boosted_candidates = self._apply_section_boost(
            candidates=merged_candidates,
            priority_sections=classification.priority_sections,
        )

        top_results = self._validate_top_k(boosted_candidates, top_k)

        return {
            "query": query,
            "query_type": classification.query_type,
            "priority_sections": classification.priority_sections,
            "bm25_weight": classification.bm25_weight,
            "dense_weight": classification.dense_weight,
            "dense_available": len(dense_results) > 0,
            "top5_chunk_ids": [item["chunk_id"] for item in top_results],
            "results": top_results,
        }

    def _safe_bm25_search(self, query: str, top_k: int) -> list[dict]:
        try:
            results = self.bm25_retriever.search(query, top_k=top_k)
            return self._standardize_results(results, retriever_name="bm25")
        except Exception as error:
            print(f"[WARN] BM25 검색 실패: {error}")
            return []

    def _safe_dense_search(self, query: str, top_k: int) -> list[dict]:
        if self.dense_retriever is None:
            return []

        try:
            results = self.dense_retriever.search(query, top_k=top_k)
            return self._standardize_results(results, retriever_name="dense")
        except NotImplementedError:
            return []
        except Exception as error:
            print(f"[WARN] Dense 검색 실패. BM25-only fallback으로 진행합니다: {error}")
            return []

    def _standardize_results(self, results: list[dict], retriever_name: str) -> list[dict]:
        standardized = []

        for item in results:
            standardized.append({
                "chunk_id": item.get("chunk_id"),
                "score": float(item.get("score", 0.0)),
                "section_type": item.get("section_type", "기타"),
                "retriever": item.get("retriever", retriever_name),
                "preview": item.get("preview", ""),
                "chunk_text": item.get("chunk_text", ""),
            })

        return standardized

    def _normalize_scores(self, results: list[dict]) -> dict[str, float]:
        """
        검색 결과 score를 0~1 사이로 min-max normalize한다.
        결과가 비어 있거나 모든 점수가 같으면 1.0으로 처리한다.
        """
        if not results:
            return {}

        scores = [item.get("score", 0.0) for item in results]
        min_score = min(scores)
        max_score = max(scores)

        normalized = {}

        for item in results:
            chunk_id = item.get("chunk_id")
            score = item.get("score", 0.0)

            if not chunk_id:
                continue

            if max_score == min_score:
                normalized[chunk_id] = 1.0
            else:
                normalized[chunk_id] = (score - min_score) / (max_score - min_score)

        return normalized

    def _merge_results(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
        bm25_weight: float,
        dense_weight: float,
    ) -> list[dict]:
        bm25_norm = self._normalize_scores(bm25_results)
        dense_norm = self._normalize_scores(dense_results)

        candidate_map = {}

        for item in bm25_results:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue

            candidate_map[chunk_id] = {
                **item,
                "bm25_score_norm": bm25_norm.get(chunk_id, 0.0),
                "dense_score_norm": 0.0,
                "hybrid_score": 0.0,
                "section_boosted": False,
            }

        for item in dense_results:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue

            if chunk_id not in candidate_map:
                candidate_map[chunk_id] = {
                    **item,
                    "bm25_score_norm": 0.0,
                    "dense_score_norm": dense_norm.get(chunk_id, 0.0),
                    "hybrid_score": 0.0,
                    "section_boosted": False,
                }
            else:
                candidate_map[chunk_id]["dense_score_norm"] = dense_norm.get(chunk_id, 0.0)
                candidate_map[chunk_id]["retriever"] = "hybrid"

        # Dense가 아직 없으면 BM25-only fallback.
        dense_available = len(dense_results) > 0

        for chunk_id, item in candidate_map.items():
            if dense_available:
                hybrid_score = (
                    bm25_weight * item.get("bm25_score_norm", 0.0)
                    + dense_weight * item.get("dense_score_norm", 0.0)
                )
            else:
                hybrid_score = item.get("bm25_score_norm", 0.0)

            item["hybrid_score"] = float(hybrid_score)

        return list(candidate_map.values())

    def _apply_section_boost(
        self,
        candidates: list[dict],
        priority_sections: list[str],
    ) -> list[dict]:
        boosted = []

        for item in candidates:
            section_type = item.get("section_type", "기타")
            hybrid_score = item.get("hybrid_score", 0.0)

            if section_type in priority_sections:
                item["hybrid_score"] = hybrid_score * (1 + self.section_boost_ratio)
                item["section_boosted"] = True
            else:
                item["section_boosted"] = False

            boosted.append(item)

        return boosted

    def _validate_top_k(self, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Top-k 반환 안정성 검증.
        - hybrid_score 기준 정렬
        - chunk_id 중복 제거
        - 실제 존재하는 chunk_id만 유지
        - 최대 top_k개 반환
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda item: item.get("hybrid_score", 0.0),
            reverse=True,
        )

        seen = set()
        top_results = []

        for item in sorted_candidates:
            chunk_id = item.get("chunk_id")

            if not chunk_id:
                continue

            if chunk_id in seen:
                continue

            if chunk_id not in self.valid_chunk_ids:
                continue

            top_results.append(item)
            seen.add(chunk_id)

            if len(top_results) == top_k:
                break

        return top_results