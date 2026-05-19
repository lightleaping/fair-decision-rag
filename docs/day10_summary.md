# Day 10 작업 요약

## 1. 목표

Day 10의 목표는 공정거래 의결서 RAG 시스템에서 BM25 기반 검색 baseline을 만들고, 사용자 질문에 대해 30초 이내에 중복 없는 Top-5 chunk_id를 반환하는 것이다.

## 2. 구현 내용

### 2.1 A 작업

A는 질문 분류 및 section boost 모듈을 구현하였다.

구현 파일:

- `src/retrieval/query_classifier.py`
- `src/retrieval/section_boost.py`
- `src/retrieval/topk_selector.py`
- `src/retrieval/day10_a_runner.py`

주요 기능:

- 사용자 질문을 `penalty`, `legal_reasoning`, `fact_pattern`, `law_article`, `corrective_order`, `summary`, `general`로 분류
- 질문 유형별 section priority 적용
- 검색 결과의 `section_type`에 따라 score 보정
- 중복 없는 Top-5 chunk_id 선택

### 2.2 B 작업

B는 BM25 baseline 검색기와 실행 파이프라인을 구현하였다.

구현 파일:

- `src/retrieval/bm25_retriever.py`
- `src/retrieval/day10_bm25_pipeline.py`
- `src/data/build_chunks_jsonl.py`
- `main.py`
- `src/retrieval/day10_verify_results.py`

주요 기능:

- `_hybrid.json` 파일을 `data/chunks.jsonl`로 변환
- BM25 검색 후보 생성
- A 모듈과 연결하여 section boost 적용
- 정확히 5개의 chunk_id 반환
- 실행 시간 측정
- 검증 결과 JSON 저장

## 3. 실행 명령어

```powershell
python main.py --query "이 사건 과징금은 얼마야?"
```
4. 검증 명령어
python src/retrieval/day10_verify_results.py

5. 산출물
data/chunks.jsonl
outputs/results/day10_bm25_top5.json
outputs/results/day10_verification_report.json
docs/day10_summary.md

6. 완료 조건

Day 10 완료 조건은 다음과 같다.

BM25 검색기가 실행된다.
사용자 질문에 대해 검색 후보를 반환한다.
질문 유형 분류가 적용된다.
section boost가 적용된다.
최종 결과는 정확히 5개의 chunk_id를 포함한다.
chunk_id 중복이 없다.
30초 이내에 응답한다.
외부 API를 사용하지 않는다.
모델 학습을 수행하지 않는다.
기존 chunk_id를 유지한다.

7. Day 10 결론

Day 10에서는 BM25 baseline 검색 파이프라인을 완성하였다.
이제 Day 11에서는 1GB 이하 경량 임베딩 모델을 이용한 Dense Retrieval을 구현하고, BM25와 비교할 수 있는 검색 결과를 생성한다.

---

# 6. Git 커밋

검증까지 끝나면 커밋해.

```powershell
git add .
git commit -m "feat: complete day10 bm25 retrieval baseline"
```

이미 원격 저장소 연결되어 있으면:

git push origin main