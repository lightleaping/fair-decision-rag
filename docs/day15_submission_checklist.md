# Track 2 제출 체크리스트

## 코드와 데이터

- [x] 공식 ZIP 직접 로딩
- [x] 500개 문서, 31,877개 청크 변환
- [x] 기존 `chunk_id` 보존 및 중복 검증
- [x] BM25 검색
- [x] Dense 검색과 저장형 인덱스
- [x] Hybrid 점수 융합
- [x] 질문 유형별 section boost
- [x] 검색 근거로 제한한 추출형 답변 생성
- [x] Evidence trace
- [x] Recall@5, MRR, token F1 평가
- [x] 500건 silver QA 생성 및 전체 평가
- [x] 단일 실행 CLI
- [x] Python 3.11 로컬 실행 환경
- [x] Dockerfile과 Compose 구성
- [x] 50개 문서 샘플 데이터와 대응 Dense 인덱스

## 검증 결과

- [x] 전체 Dense 인덱스: 31,877 × 384
- [x] 전체 데이터 로컬 Hybrid 실행
- [x] 결과의 상위 5개 chunk ID 중복 없음
- [x] `dense_available=true`
- [x] `trace_valid=true`
- [x] 단위 테스트 3건 통과
- [x] 500건 silver QA 평가: Recall@5 0.985, MRR 0.981
- [ ] 공식 또는 수작업 검수 QA로 최종 평가
- [ ] 대회 제출 스키마 확정 후 출력 어댑터 반영
- [ ] BERTScore 평가 모델을 오프라인 제출 패키지에 포함
- [ ] 공모전 모델개발 결과서 작성
- [ ] 10분 발표자료와 시연 화면 준비

Silver QA 점수는 로컬 회귀 테스트용이며 공모전 공식 성능으로 제시하지 않습니다.

## 전체 데이터 로컬 실행

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe run.py `
  --mode hybrid `
  --chunks data/chunks.jsonl `
  --dense-index indexes/dense_embeddings.npy `
  --dense-metadata indexes/dense_chunks.jsonl `
  --query "시장지배적 지위 남용행위의 판단 기준은 무엇인가?"
```

## 샘플 Docker 데이터 생성

```powershell
.\.venv\Scripts\python.exe scripts\build_sample_dataset.py --documents 50
docker compose build
docker compose run --rm fair-decision-rag
```
