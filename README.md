## Day 10-A: 질문 분류 및 Section Boost 모듈

### 목표

공정거래 의결서 질의에 대해 질문 유형을 분류하고, 검색 결과의 section_type에 따라 점수를 보정하여 더 적절한 chunk_id를 Top-5로 반환한다.

### 구현 파일

```text
src/retrieval/query_classifier.py
src/retrieval/section_boost.py
src/retrieval/topk_selector.py
src/retrieval/day10_a_runner.py
```

### 실행 방법
python src/retrieval/day10_a_runner.py

또는 import 오류가 있을 경우:

python -m src.retrieval.day10_a_runner

### 출력 결과

outputs/results/day10_a_sample_result.json


### 핵심 규칙
assert len(chunk_ids) == 5
assert len(set(chunk_ids)) == 5

최종 검색 결과는 반드시 정확히 5개의 중복 없는 chunk_id를 반환해야 한다.

외부 API 사용 여부 : 사용하지 않음.

모델 학습 여부 : 학습하지 않음.

chunk_id 처리 원칙 : 공개본 데이터에 존재하는 기존 chunk_id만 사용하며, 새 chunk_id를 생성하지 않는다.


## Day 11-A: Dense Retrieval 구현

### 목표

1GB 이하 경량 임베딩 모델을 사용하여 공정거래 의결서 chunk에 대한 Dense Retrieval을 구현한다.

### 사용 모델

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```
