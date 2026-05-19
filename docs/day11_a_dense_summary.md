# Day 11-A 작업 요약: Dense Retrieval 구현

## 1. 목표

Day 11-A의 목표는 공정거래 의결서 RAG 시스템에 1GB 이하 경량 임베딩 모델 기반 Dense Retrieval을 추가하는 것이다.

Day 10에서는 BM25 기반 검색 baseline을 구현하였다.  
Day 11-A에서는 사용자의 질문과 chunk 본문의 의미적 유사도를 계산하여 관련 chunk를 검색하는 Dense Retrieval을 구현하였다.

## 2. 핵심 구현 조건

본 프로젝트의 구현 조건은 다음과 같다.

- 외부 API 호출 금지
- 모델 학습 불필요
- 1GB 이하 경량 모델 사용
- 한 요청 30초 이내 응답
- 정확히 5개의 chunk_id 반환
- 중복 chunk_id 금지
- 공개본 데이터에 존재하는 기존 chunk_id 유지
- 검색된 근거 chunk에 기반한 답변 생성

## 3. 사용 모델

Dense Retrieval에는 다음 임베딩 모델을 사용한다.

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2