# Day 4 — A 작업 결과: BM25 검색 기준

## 1. Day 4 목표

BM25 검색이 공정거래 의결서 RAG 시스템에서 왜 필요한지 정리하고, 검색 결과 확인 기준을 만든다.

## 2. BM25의 역할

BM25는 키워드 기반 검색을 담당한다.

공정거래 의결서에서는 법 조항, 기업명, 금액, 처분명처럼 정확한 표현이 중요한 정보가 많다.  
따라서 의미 기반 Dense Retrieval만 사용하는 것보다 BM25를 함께 사용하는 것이 안정적이다.

## 3. BM25가 유리한 질문 유형

| 질문 유형 | 핵심 키워드 | 이유 |
|---|---|---|
| 법 조항 질문 | 제19조, 제23조, 하도급법 | 조항 번호가 정확히 일치해야 한다 |
| 과징금 질문 | 과징금, 원, 억, 부과 | 금액과 처분 표현이 중요하다 |
| 처분 결과 질문 | 시정명령, 고발, 납부명령 | 공정위 조치명이 직접 등장한다 |
| 기업명 질문 | 주식회사, 피심인, 기업명 | 회사명은 정확한 문자열 검색이 중요하다 |
| 위반 유형 질문 | 담합, 부당한 공동행위, 표시광고 | 위반 유형 표현이 직접 등장한다 |

## 4. 테스트 질문

BM25 검색 테스트에는 다음 질문을 사용한다.

1. 기업들이 가격을 같이 정한 사례가 있나요?
2. 과징금이 부과된 사건은?
3. 하도급 대금을 지급하지 않은 사례가 있나요?
4. 입찰담합 사건의 조치는 무엇인가요?
5. 어떤 법 조항을 위반했나요?
6. 시정명령이 내려진 사건은?
7. 고발 조치가 있었나요?
8. 부당한 공동행위가 인정된 이유는 무엇인가요?
9. 관련매출액은 어떻게 산정되었나요?
10. 표시광고법 위반 사례가 있나요?

## 5. 검색 결과 출력 형식

BM25 검색 결과는 다음 정보를 포함해야 한다.

- chunk_id
- score
- section_type
- preview

예시:

```text
Query: 과징금이 부과된 사건은?

[1]
chunk_id: chunk_000152
score: 13.42
section_type: 주문
preview: 피심인에게 과징금 1억 원을 부과한다...

# src/bm25_retriever.py

from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks, tokenizer):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.corpus = [chunk["clean_text"] for chunk in chunks]
        self.tokenized_corpus = [self.tokenizer(text) for text in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[idx]),
                "section_type": chunk.get("section_type", "기타"),
                "preview": chunk.get("chunk_text", "")[:150],
                "chunk_text": chunk.get("chunk_text", "")
            })

        return results


1. 토크나이저를 바꾼다
2. 임베딩 모델을 바꾼다
