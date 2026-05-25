# Day 10-A 작업 요약

## 1. 목표

Day 10-A의 목표는 공정거래 의결서 RAG 시스템에서 검색 결과의 품질을 높이기 위한 질문 분류 및 section boost 모듈을 구현하는 것이다.

본 프로젝트의 핵심 조건은 다음과 같다.

- 외부 API 호출 금지
- 모델 학습 불필요
- 1GB 이하 경량 모델 또는 로컬 포함 모델 사용
- 한 요청 30초 이내 응답
- 정확히 5개의 chunk_id 반환
- 중복 chunk_id 금지
- 공개본 데이터에 존재하는 기존 chunk_id 유지
- 검색된 근거에 기반한 답변 생성

## 2. 구현한 파일

### 2.1 query_classifier.py

사용자 질문을 다음 유형 중 하나로 분류한다.

- penalty
- legal_reasoning
- fact_pattern
- law_article
- corrective_order
- summary
- general

질문 유형에 따라 section priority를 반환한다.

### 2.2 section_boost.py

검색 결과의 section_type을 확인하고, 질문 유형별 section priority에 따라 검색 점수를 보정한다.

예를 들어 사용자가 과징금을 질문하면 penalty section에 더 높은 boost를 적용한다.

### 2.3 topk_selector.py

검색 결과에서 정확히 5개의 중복 없는 chunk_id를 선택한다.

핵심 검증 규칙은 다음과 같다.

```python
assert len(chunk_ids) == 5
assert len(set(chunk_ids)) == 5
2.4 day10_a_runner.py

질문 분류, section boost, Top-5 선택을 하나의 함수로 통합한다.

입력 형식:

{
    "chunk_id": "...",
    "section_type": "...",
    "score": 0.0,
    "text": "..."
}

출력 형식:

{
  "query": "...",
  "query_type": "...",
  "matched_keywords": [],
  "top5_chunk_ids": [],
  "top5_results": []
}
3. Day 10-A 파이프라인
사용자 질문
↓
QueryClassifier
↓
query_type 분류
↓
section_priority 선택
↓
SectionBooster
↓
boosted_score 계산
↓
TopKSelector
↓
정확히 5개 chunk_id 반환
4. 완료 조건

Day 10-A는 다음 조건을 만족한다.

질문 유형 분류 가능
section_type 기반 boost 적용 가능
검색 결과 중복 제거 가능
정확히 5개 chunk_id 반환 가능
결과 JSON 저장 가능
외부 API 호출 없음
모델 학습 없음
chunk_id 새로 생성 없음
5. 다음 연결 작업

Day 10-B에서 BM25 검색 결과를 아래 형식으로 넘기면 A 모듈과 바로 연결할 수 있다.

{
    "chunk_id": "case_001_chunk_003",
    "section_type": "penalty",
    "score": 6.8,
    "text": "피심인에게 과징금 1억 원을 부과한다.",
    "doc_id": "case_001",
    "title": "OOO의 불공정거래행위에 대한 건"
}

최종적으로 Day 10 목표는 BM25 기반 검색 결과가 A 모듈을 거쳐 30초 이내에 중복 없는 Top-5 chunk_id를 반환하는 것이다.