# 공정거래 의결서 기반 Section-aware Hybrid Retrieval RAG

## 1. 프로젝트 개요

본 프로젝트는 공정거래위원회 공개본 의결서 데이터를 기반으로, 사용자의 자연어 질문에 대해 관련성이 높은 근거 chunk를 검색하고, 검색된 근거를 바탕으로 답변과 evidence trace를 제공하는 Retrieval 기반 질의응답 시스템입니다.

공정거래 의결서는 사건 개요, 사실관계, 법 위반 판단, 시정명령, 과징금 등 중요한 정보가 긴 문서 안에 분산되어 있습니다. 본 프로젝트는 의결서 데이터를 chunk 단위로 정리하고, 질문 유형과 section_type을 함께 고려하여 관련 근거를 검색하는 것을 목표로 합니다.

최종 구현 브랜치: `toastcoding-working`

## 2. 완료 조건

사용자가 공정거래 의결서 관련 질문을 입력하면, 시스템은 공개본 데이터에 존재하는 중복 없는 관련 `chunk_id` 5개와 해당 근거에 기반한 답변을 반환합니다.

본 프로젝트의 구현 조건은 다음과 같습니다.

* 외부 API를 호출하지 않습니다.
* 모델 학습은 수행하지 않습니다.
* Retrieval 및 inference 중심으로 동작합니다.
* 최종 반환 `chunk_id`는 공개본 데이터에 존재하는 기존 ID를 유지합니다.
* 한 요청에서 중복 없는 Top-5 `chunk_id`를 반환합니다.
* 사용자 질문을 영구 저장하지 않습니다.
* 검색 결과와 답변은 근거 chunk에 기반하여 생성합니다.

## 3. 주요 기능

* 공정거래 의결서 chunk 데이터 로딩
* 텍스트 정제 및 전처리
* `chunk_id` 무결성 검사
* `section_type` 자동 보정
* 질문 유형 분류
* Section-aware score boost
* BM25 기반 검색
* Hybrid Retriever interface
* BM25 fallback mode
* Top-5 `chunk_id` 선택
* 중복 chunk 제거
* Evidence trace 생성
* Extractive answer fallback
* 검색 및 검증 결과 JSON 저장
* 평가 자동화 구조 제공

## 4. 검색 파이프라인

전체 검색 흐름은 다음과 같습니다.

```text
사용자 질문 입력
→ 질문 유형 분류
→ section 우선순위 결정
→ BM25 검색
→ Hybrid Retriever 호출
→ section_type 기반 score boost
→ 후보 chunk 정렬
→ chunk_id 존재 여부 및 중복 검증
→ Top-5 chunk_id 반환
→ 근거 기반 답변 생성
→ Evidence trace 출력
```

현재 `HybridRetriever`는 BM25 fallback 방식으로 동작합니다. Dense Retriever 확장을 고려한 인터페이스는 포함되어 있으나, 최종 검증 기준에서는 외부 API 호출 없이 로컬 검색 파이프라인을 우선 사용합니다.

## 5. 폴더 구조

```text
project/
  data/
    raw/
    processed/
    chunks.jsonl

  docs/
    day1_pipeline_design.md
    day2_data_policy.md
    day3_section_policy.md
    day4_bm25_policy.md
    day5_dense_retrieval.md
    day7_week1_integration.md
    day8_hybrid_retrieval.md

  outputs/
    results/

  scripts/
    build_chunks_jsonl.py

  src/
    retrieval/
      answer_generator.py
      bm25_retriever.py
      chunk_validator.py
      day10_a_runner.py
      day10_bm25_pipeline.py
      evaluator.py
      evidence_trace.py
      hybrid_retriever.py
      query_classifier.py
      score_fusion.py
      section_boost.py
      topk_selector.py

    dense_retriever.py
    hybrid_retriever.py
    preprocess.py
    utils.py

  main.py
  main_hybrid.py
  main_day12_hybrid.py
  main_day14_answer_trace.py
  requirements.txt
  readme.md
```

## 6. 데이터 준비

원본 `*_hybrid.json` 파일을 `data/chunks.jsonl` 형식으로 변환합니다.

```bash
python scripts/build_chunks_jsonl.py
```

변환된 데이터는 검색 파이프라인의 입력으로 사용됩니다.

## 7. 실행 방법

A 파트 검증 실행:

```bash
python -m src.retrieval.day10_a_runner
```

BM25 기반 검색 파이프라인 실행:

```bash
python main.py
```

Hybrid 검색 파이프라인 실행:

```bash
python main_hybrid.py
```

Day 12 Hybrid 파이프라인 실행:

```bash
python main_day12_hybrid.py
```

Day 14 답변 및 Evidence Trace 실행:

```bash
python main_day14_answer_trace.py
```

## 8. A 파트 검증 결과

A 파트에서는 질문 유형 분류, section boost, Top-5 chunk_id 반환 조건을 검증했습니다.

검증 명령:

```bash
python -m src.retrieval.day10_a_runner
```

검증 예시 질문:

```text
이 사건 과징금은 얼마야?
```

검증 결과:

```text
query_type: penalty
matched_keywords: 과징금, 얼마
top5_chunk_ids: 5개 반환
중복 chunk_id 없음
section_boost 적용
결과 저장: outputs/results/day10_a_sample_result.json
```

예시 결과에서는 `penalty`, `order`, `legal_reasoning`, `fact`, `law_article` section이 검색 후보로 반환되며, 과징금 질문에 대해 `penalty` section이 가장 높은 boost를 받아 1순위로 선택됩니다.

## 9. 역할별 구현 범위

### A 파트

* 질문 유형 분류
* section 우선순위 판단
* section_type 기반 boost 적용
* Top-5 chunk_id 선택 검증
* 중복 없는 chunk_id 반환 확인
* 실행 결과 JSON 저장 확인

### B 파트

* BM25 검색 파이프라인 구현
* chunk 데이터 로딩 및 전처리
* chunk_id 검증 구조 구현
* Hybrid Retriever interface 구성
* BM25 fallback mode 구성
* 평가 자동화 구조 구현
* 근거 기반 답변 및 evidence trace 구조 구현

## 10. 검증 기준

최종 검색 결과는 다음 조건을 만족해야 합니다.

```text
1. chunk_id가 정확히 5개 반환된다.
2. 반환된 chunk_id에 중복이 없다.
3. 반환된 chunk_id는 공개본 데이터에 존재하는 기존 ID를 사용한다.
4. 질문 유형에 따라 section_type boost가 적용된다.
5. 답변은 검색된 근거 chunk에 기반한다.
6. 외부 API를 호출하지 않는다.
7. 모델 학습을 수행하지 않는다.
```

## 11. 현재 상태

최종 기준 브랜치: `toastcoding-working`

현재 브랜치에는 Day 10~15 retrieval pipeline이 포함되어 있으며, A 파트 검증 명령을 통해 질문 유형 분류, section-aware boost, Top-5 chunk_id 반환 조건이 정상 동작함을 확인했습니다.
