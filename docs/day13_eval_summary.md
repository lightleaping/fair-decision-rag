# Day 13 Evaluation Summary

## 1. 평가 대상

- 입력 결과 파일: `outputs/results/day12_hybrid_top5.json`
- 평가 결과 파일: `outputs/results/day13_eval_report.json`
- QA 평가셋: `data/processed/qa_eval_set.jsonl`
- 검색 대상 chunk 파일: `data/chunks.jsonl`
- 평가 질의 수: 10개

## 2. 현재 검색기 상태

현재 Day 12 검색기는 `HybridRetriever` 인터페이스를 사용하지만 Dense Retriever는 아직 연결되지 않았다.  
따라서 현재 결과는 `hybrid_bm25_fallback` 상태이며, BM25 검색 결과를 기반으로 Top-5 chunk_id를 반환한다.

## 3. 자동 평가 결과

- 전체 질의 수: 10개
- 형식 검증 통과: 10개
- 형식 검증 실패: 0개
- 정확히 5개 chunk_id 반환: 10개
- 중복 없는 chunk_id 반환: 10개
- 30초 이내 응답: 10개
- 평균 응답 시간: 0.0359초
- 최대 응답 시간: 0.0562초
- 최소 응답 시간: 0.0265초

## 4. 검증 기준

각 질의 결과는 다음 조건을 만족해야 한다.

- 정확히 5개의 chunk_id 반환
- 중복 chunk_id 없음
- 30초 이내 응답
- 공개본 데이터에 존재하는 chunk_id만 사용
- 외부 API 사용 없음

## 5. Recall@5 / MRR 상태

현재 `qa_eval_set.jsonl`의 `gold_chunk_ids`가 비어 있어 Recall@5와 MRR은 계산하지 않았다.

현재 상태:

- `recall_at_5_available`: false
- `avg_recall_at_5`: null
- `mrr_available`: false
- `avg_mrr`: null

A가 수동 평가를 통해 정답 `gold_chunk_ids`를 채운 후 같은 evaluator를 다시 실행하면 Recall@5와 MRR을 계산할 수 있다.

## 6. Day 13 B 작업 결과

구현 파일:

- `src/retrieval/evaluator.py`

생성 결과:

- `outputs/results/day13_eval_report.json`
- `docs/day13_eval_summary.md`

## 7. 다음 작업

A가 수행할 작업:

- 각 query별 Top-5 결과 수동 평가
- O / △ / X 판정
- 실패 사례 분석
- query_type 또는 section boost 보정 의견 작성
- 필요한 경우 `gold_chunk_ids` 채우기

B가 이후 수행할 작업:

- A가 채운 `gold_chunk_ids` 기준으로 Recall@5 / MRR 재계산
- Day 14 answer_generator / evidence_trace 구현 준비