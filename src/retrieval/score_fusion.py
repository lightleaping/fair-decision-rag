# src/retrieval/score_fusion.py

from typing import Any, Dict, List, Optional, Set


def min_max_normalize(
    results: List[Dict[str, Any]],
    score_key: str = "score",
    output_key: str = "normalized_score",
) -> List[Dict[str, Any]]:
    """
    검색 결과 score를 0~1 범위로 정규화한다.
    BM25 score와 Dense score는 스케일이 다르므로 fusion 전에 정규화가 필요하다.
    """

    if not results:
        return []

    scores = [float(item.get(score_key, 0.0)) for item in results]
    min_score = min(scores)
    max_score = max(scores)

    normalized = []

    for item in results:
        copied = dict(item)
        score = float(copied.get(score_key, 0.0))

        if max_score == min_score:
            norm = 1.0
        else:
            norm = (score - min_score) / (max_score - min_score)

        copied[output_key] = norm
        normalized.append(copied)

    return normalized


def fuse_bm25_dense(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    valid_chunk_ids: Optional[Set[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    BM25 결과와 Dense 결과를 chunk_id 기준으로 병합한다.

    규칙:
    - chunk_id 기준 중복 병합
    - BM25/Dense score를 각각 min-max normalize
    - weighted sum으로 final_score 계산
    - 정확히 top_k개 반환
    - valid_chunk_ids가 있으면 공개본 데이터에 존재하는 chunk_id만 허용
    """

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    if bm25_weight < 0 or dense_weight < 0:
        raise ValueError("bm25_weight와 dense_weight는 0 이상이어야 합니다.")

    weight_sum = bm25_weight + dense_weight
    if weight_sum == 0:
        raise ValueError("bm25_weight와 dense_weight의 합은 0일 수 없습니다.")

    # 가중치 합이 1이 아니어도 자동 정규화
    bm25_weight = bm25_weight / weight_sum
    dense_weight = dense_weight / weight_sum

    bm25_norm = min_max_normalize(
        bm25_results,
        score_key="score",
        output_key="bm25_norm_score",
    )

    dense_norm = min_max_normalize(
        dense_results,
        score_key="score",
        output_key="dense_norm_score",
    )

    merged: Dict[str, Dict[str, Any]] = {}

    for item in bm25_norm:
        chunk_id = item.get("chunk_id")

        if not chunk_id:
            continue

        if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
            continue

        merged[chunk_id] = {
            **item,
            "bm25_score": float(item.get("score", 0.0)),
            "dense_score": 0.0,
            "bm25_norm_score": float(item.get("bm25_norm_score", 0.0)),
            "dense_norm_score": 0.0,
        }

    for item in dense_norm:
        chunk_id = item.get("chunk_id")

        if not chunk_id:
            continue

        if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
            continue

        if chunk_id not in merged:
            merged[chunk_id] = {
                **item,
                "bm25_score": 0.0,
                "dense_score": float(item.get("score", 0.0)),
                "bm25_norm_score": 0.0,
                "dense_norm_score": float(item.get("dense_norm_score", 0.0)),
            }
        else:
            merged[chunk_id]["dense_score"] = float(item.get("score", 0.0))
            merged[chunk_id]["dense_norm_score"] = float(item.get("dense_norm_score", 0.0))

    fused = []

    for chunk_id, item in merged.items():
        final_score = (
            bm25_weight * float(item.get("bm25_norm_score", 0.0))
            + dense_weight * float(item.get("dense_norm_score", 0.0))
        )

        copied = dict(item)
        copied["final_score"] = final_score
        copied["score"] = final_score
        copied["retriever"] = "hybrid"
        copied["bm25_weight"] = bm25_weight
        copied["dense_weight"] = dense_weight

        fused.append(copied)

    fused.sort(
        key=lambda x: float(x.get("final_score", 0.0)),
        reverse=True,
    )

    selected = []
    seen_chunk_ids = set()

    for item in fused:
        chunk_id = item.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id in seen_chunk_ids:
            continue

        if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
            continue

        selected.append(item)
        seen_chunk_ids.add(chunk_id)

        if len(selected) == top_k:
            break

    if len(selected) != top_k:
        raise ValueError(
            f"Hybrid Top-{top_k} 생성 실패: 현재 {len(selected)}개만 선택됨. "
            f"bm25_results={len(bm25_results)}, dense_results={len(dense_results)}"
        )

    chunk_ids = [item["chunk_id"] for item in selected]

    if len(set(chunk_ids)) != top_k:
        raise ValueError("Hybrid 결과에 중복 chunk_id가 있습니다.")

    return selected


if __name__ == "__main__":
    bm25_results = [
        {"chunk_id": "c1", "score": 10.0, "text": "BM25 1", "section_type": "fact"},
        {"chunk_id": "c2", "score": 8.0, "text": "BM25 2", "section_type": "penalty"},
        {"chunk_id": "c3", "score": 6.0, "text": "BM25 3", "section_type": "order"},
        {"chunk_id": "c4", "score": 4.0, "text": "BM25 4", "section_type": "law_article"},
        {"chunk_id": "c5", "score": 2.0, "text": "BM25 5", "section_type": "summary"},
    ]

    dense_results = [
        {"chunk_id": "c3", "score": 0.95, "text": "Dense 3", "section_type": "order"},
        {"chunk_id": "c4", "score": 0.90, "text": "Dense 4", "section_type": "law_article"},
        {"chunk_id": "c5", "score": 0.85, "text": "Dense 5", "section_type": "summary"},
        {"chunk_id": "c6", "score": 0.80, "text": "Dense 6", "section_type": "fact"},
        {"chunk_id": "c7", "score": 0.75, "text": "Dense 7", "section_type": "penalty"},
    ]

    results = fuse_bm25_dense(
        bm25_results=bm25_results,
        dense_results=dense_results,
        bm25_weight=0.5,
        dense_weight=0.5,
        top_k=5,
    )

    print("Hybrid Top-5")
    for rank, item in enumerate(results, start=1):
        print(rank, item["chunk_id"], item["final_score"])

    assert len(results) == 5
    assert len(set(item["chunk_id"] for item in results)) == 5

    print("\n검증 완료: 정확히 5개, 중복 없음")