# src/retrieval/topk_selector.py

from typing import Any, Dict, List, Optional, Set


class TopKSelector:
    """
    검색 결과에서 정확히 k개의 중복 없는 chunk_id를 선택하는 클래스.

    핵심 규칙:
    - chunk_id 중복 제거
    - score 기준 정렬
    - 정확히 k개 반환
    - 부족하면 명확한 에러 발생
    - chunk_id를 새로 만들지 않음
    """

    def __init__(self, k: int = 5):
        self.k = k

    def select(
        self,
        results: List[Dict[str, Any]],
        score_key: str = "boosted_score",
        chunk_id_key: str = "chunk_id",
        valid_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        검색 결과에서 정확히 k개의 chunk를 선택한다.

        Args:
            results:
                검색 결과 리스트.

            score_key:
                정렬에 사용할 점수 key.
                보통 boosted_score 또는 score 사용.

            chunk_id_key:
                chunk_id가 들어 있는 key.

            valid_chunk_ids:
                공개본 데이터에 존재하는 chunk_id 집합.
                None이면 존재 검증은 생략한다.

        Returns:
            List[Dict[str, Any]]:
                정확히 k개의 중복 없는 검색 결과.

        Raises:
            ValueError:
                k개를 만들 수 없거나 chunk_id가 유효하지 않은 경우.
        """

        if not results:
            raise ValueError("검색 결과가 비어 있습니다.")

        sorted_results = sorted(
            results,
            key=lambda x: float(x.get(score_key, 0.0)),
            reverse=True,
        )

        selected = []
        seen_chunk_ids = set()

        for item in sorted_results:
            chunk_id = item.get(chunk_id_key)

            if not chunk_id:
                continue

            if chunk_id in seen_chunk_ids:
                continue

            if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
                continue

            selected.append(item)
            seen_chunk_ids.add(chunk_id)

            if len(selected) == self.k:
                break

        chunk_ids = [item[chunk_id_key] for item in selected]

        if len(chunk_ids) != self.k:
            raise ValueError(
                f"Top-{self.k} 선택 실패: "
                f"현재 {len(chunk_ids)}개만 선택됨. "
                f"검색 후보를 더 많이 가져와야 합니다."
            )

        if len(set(chunk_ids)) != self.k:
            raise ValueError("Top-K 결과에 중복 chunk_id가 포함되어 있습니다.")

        return selected


if __name__ == "__main__":
    sample_results = [
        {
            "chunk_id": "case_001_chunk_001",
            "boosted_score": 9.2,
            "text": "사실관계 내용"
        },
        {
            "chunk_id": "case_001_chunk_002",
            "boosted_score": 8.7,
            "text": "판단 근거 내용"
        },
        {
            "chunk_id": "case_001_chunk_002",
            "boosted_score": 8.5,
            "text": "중복 chunk"
        },
        {
            "chunk_id": "case_001_chunk_003",
            "boosted_score": 8.1,
            "text": "법조항 내용"
        },
        {
            "chunk_id": "case_001_chunk_004",
            "boosted_score": 7.9,
            "text": "시정명령 내용"
        },
        {
            "chunk_id": "case_001_chunk_005",
            "boosted_score": 7.5,
            "text": "과징금 내용"
        },
        {
            "chunk_id": "case_001_chunk_006",
            "boosted_score": 7.2,
            "text": "추가 후보"
        },
    ]

    selector = TopKSelector(k=5)
    top5 = selector.select(sample_results)

    print("Top-5 결과")
    for rank, item in enumerate(top5, start=1):
        print(
            f"{rank}. {item['chunk_id']} | "
            f"score={item['boosted_score']} | "
            f"text={item['text']}"
        )

    chunk_ids = [item["chunk_id"] for item in top5]

    assert len(chunk_ids) == 5
    assert len(set(chunk_ids)) == 5

    print("\n검증 완료: 정확히 5개, 중복 없음")