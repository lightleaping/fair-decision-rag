# Day 7 — 1주차 통합 점검 결과

## 1. Day 7 목표

Day 1~6에서 만든 결과물을 하나의 흐름으로 연결하여, 질문 입력 시 BM25 검색 결과와 Dense Retrieval 검색 결과가 모두 출력되는지 확인한다.

## 2. 통합 대상

| 항목 | 파일 |
|---|---|
| 데이터 로딩 | `src/preprocess.py` |
| BM25 검색 | `src/bm25_retriever.py` |
| Dense Retrieval | `src/dense_retriever.py` |
| QA 검증셋 | `data/qa_eval_set.jsonl` |
| 통합 테스트 | `src/test_week1_pipeline.py` |

## 3. 통합 테스트 흐름

```text
질문 입력
→ chunk 데이터 로딩
→ QA셋 로딩
→ BM25 검색
→ Dense Retrieval 검색
→ BM25 Top-10 출력
→ Dense Top-10 출력
→ chunk_id, score, section_type, preview 확인