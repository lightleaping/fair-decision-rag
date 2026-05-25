# Day 1 — A 작업 결과: AI 파이프라인 설계

## 1. 프로젝트 목표

본 프로젝트는 공정거래위원회 의결서 chunk 데이터에서 사용자의 질문과 관련된 근거 chunk를 검색하고, Top-5 chunk_id와 근거 기반 답변을 함께 반환하는 Section-aware Hybrid Retrieval 기반 Legal RAG 시스템을 개발하는 것을 목표로 한다.

## 2. 전체 검색 및 답변 생성 흐름

사용자 질문 입력
→ 질문 유형 분류
→ 우선 검색할 section_type 결정
→ BM25 검색
→ Dense Retrieval 검색
→ BM25 + Dense 결과 결합
→ Section-aware 점수 보정
→ Top-5 chunk_id 후처리 검증
→ Grounded Answer 생성
→ Evidence Trace 출력

## 3. 질문 유형 분류 기준

### 과징금 질문
- 과징금
- 금액
- 얼마
- 부과

### 처분 결과 질문
- 조치
- 처분
- 명령
- 고발
- 시정명령

### 위반 이유 질문
- 왜
- 이유
- 근거
- 판단

### 행위 패턴 질문
- 어떤 행위
- 방식
- 담합
- 합의
- 거래상지위남용
- 하도급

### 법 조항 질문
- 법
- 조항
- 위반
- 공정거래법
- 하도급법

### 사건 요약 질문
- 요약
- 핵심
- 정리
- 무슨 사건

## 4. 질문 유형별 우선 section_type

| 질문 유형 | 우선 section_type |
|---|---|
| 과징금 질문 | 주문, 별지 |
| 처분 결과 질문 | 주문 |
| 위반 이유 질문 | 이유 |
| 행위 패턴 질문 | 이유 |
| 법 조항 질문 | 이유 |
| 사건 요약 질문 | 전체 |

## 5. Retrieval 모듈 역할

### BM25 Retriever

BM25는 키워드 기반 검색을 담당한다.  
법 조항, 기업명, 금액, 과징금처럼 정확한 단어가 중요한 질문에 강하다.

### Dense Retriever

Dense Retrieval은 의미 기반 검색을 담당한다.  
질문 표현과 문서 표현이 달라도 의미가 유사한 chunk를 찾는 데 사용한다.

### Hybrid Retriever

Hybrid Retrieval은 BM25와 Dense Retrieval 결과를 결합한다.  
초기 결합 방식은 다음과 같다.

Hybrid Score = 0.5 × BM25 Score + 0.5 × Dense Score

## 6. Top-5 chunk_id 반환 조건

최종 검색 결과는 반드시 다음 조건을 만족해야 한다.

- 정확히 5개 chunk_id를 반환한다.
- 중복 chunk_id는 제거한다.
- 실제 데이터에 존재하는 chunk_id만 반환한다.
- 관련성이 높은 순서대로 정렬한다.
- 검색 근거로 사용된 chunk_text와 section_type을 함께 보존한다.

## 7. Grounded Answer 원칙

답변 생성은 반드시 검색된 Top-5 chunk 안의 정보만 사용한다.

답변 규칙은 다음과 같다.

- 제공된 근거 chunk 안에서만 답변한다.
- 근거가 부족하면 부족하다고 말한다.
- 새로운 법률 판단을 추측하지 않는다.
- 사용한 chunk_id를 함께 출력한다.
- 가능하면 evidence_sentence를 함께 제시한다.