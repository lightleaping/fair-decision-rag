# Day 8 — Hybrid Retrieval 설계

## 1. Day 8 목표

BM25 검색 결과와 Dense Retrieval 검색 결과를 하나의 후보 리스트로 합치고, 정규화된 점수를 결합하여 Hybrid 검색 결과 Top-10을 출력한다.

## 2. Hybrid Retrieval이 필요한 이유

BM25는 법 조항, 기업명, 금액, 과징금처럼 정확한 키워드가 중요한 질문에 강하다.  
Dense Retrieval은 질문 표현과 원문 표현이 달라도 의미가 비슷한 chunk를 찾는 데 강하다.

따라서 두 검색 방식을 결합하면 키워드 검색과 의미 검색의 장점을 함께 사용할 수 있다.

## 3. 기본 점수 결합 방식

초기 MVP에서는 다음 결합식을 사용한다.

```text
Hybrid Score = 0.5 × BM25 정규화 점수 + 0.5 × Dense 정규화 점수