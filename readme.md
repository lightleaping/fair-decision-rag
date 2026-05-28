# 공정거래 의결서 기반 BM25 Baseline Retrieval

## 1. 프로젝트 개요

공정거래 의결서 데이터를 chunk 단위로 나누고, 사용자의 자연어 질문에 대해 관련성이 높은 근거 chunk를 검색하는 baseline retrieval 시스템입니다.

1주차 목표는 BM25 기반 검색 파이프라인을 구현하고, 이후 Dense Retrieval 및 Hybrid Retrieval로 확장할 수 있는 기반을 만드는 것입니다.

## 2. 주요 기능

- chunk 데이터 로딩
- 텍스트 정제
- chunk_id 무결성 검사
- section_type 자동 보정
- BM25 검색
- Top-5 chunk_id 반환
- 검색 결과 JSON 저장
- QA 검증셋 JSONL 구조 제공

## 3. 폴더 구조

```text
project/
  data/
    raw/
    processed/
  src/
    preprocess.py
    bm25_retriever.py
    utils.py
    evaluator.py
    dense_retriever.py
  outputs/
    results/
  main.py
  requirements.txt
  README.md

  ## Retrieval Pipeline Status

본 프로젝트는 공정거래위원회 공개본 의결서 chunk 데이터를 대상으로 외부 API 호출 없이 Retrieval 기반 질의응답 파이프라인을 구현한다.

현재 구현 상태는 다음과 같다.

- BM25 baseline retrieval
- Query classification
- Section-aware score boost
- Top-5 chunk_id selection
- Chunk_id validation
- HybridRetriever interface
- BM25 fallback mode
- Evaluation automation
- Extractive answer fallback
- Evidence trace

현재 Dense Retriever는 아직 연결되지 않았으며, `HybridRetriever`는 BM25 fallback 방식으로 동작한다.

## Data Preparation

원본 `*_hybrid.json` 파일을 `data/chunks.jsonl`로 변환한다.

```bash
python scripts/build_chunks_jsonl.py