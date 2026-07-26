# 공정거래 의결서 NLP 검색

**Section-aware Hybrid Retrieval, Grounded Answer, Offline FastAPI**

> 공정거래위원회 공개 의결서에서 질문과 관련된 근거 청크를 검색하고, 순위가 있는
> Top-5 `chunk_id`와 근거 기반 답변을 반환하는 로컬 RAG 프로젝트입니다.

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3561D8?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense-21AFC4?style=flat-square" alt="Hybrid Retrieval">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Offline-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Offline Docker">
  <img src="https://img.shields.io/badge/External%20API-None-2FA66A?style=flat-square" alt="No External API">
</p>

---

## Why This Project

공정거래 의결서는 사건 개요, 사실관계, 적용 법조, 위법성 판단, 시정명령과 과징금
산정 근거가 하나의 긴 문서에 분산되어 있습니다. 단순 키워드 검색만으로는 질문의
의도와 다른 구간이 상위에 노출되거나, 답변이 어떤 원문에 근거했는지 추적하기
어렵습니다.

이 프로젝트는 다음 문제를 해결하는 데 초점을 맞췄습니다.

- 법률 용어의 정확한 일치와 의미적으로 유사한 표현을 함께 검색
- 질문 유형에 따라 `주문`, `법리`, `사실관계`, `과징금` 등의 섹션 우선순위 조정
- 공정거래위원회가 제공한 원본 `chunk_id`를 유지해 검색 근거 추적
- 답변에 사용한 근거를 Top-5 검색 결과 안으로 제한
- 외부 API와 인터넷 연결이 없는 Docker 환경에서 동일하게 실행

이 프로젝트는 공정거래위원회 제2회 「공정거래 데이터」 활용 공모전 Track 2의
RAG 기반 의결서 질의응답 과제를 목표로 개발했습니다.

[공모전 Track 2 안내](https://www.fairdata.go.kr/aic/contestInfo.do?#tab-track-nav3)

---

## Project Overview

| 항목 | 내용 |
|---|---|
| **개발 기간** | 2026.05–2026.07 |
| **상태** | 공모전용 모델 구현 및 로컬 검증 완료 |
| **대상 데이터** | 공정거래위원회 공개 의결서 청크 31,877개 |
| **목표** | 질문별 관련 근거 Top-5 검색과 근거 기반 답변 생성 |
| **검색 방식** | BM25 + 다국어 MiniLM Dense Retrieval + Section Boost |
| **답변 방식** | 검색 청크 안에서 구성하는 보수적 추출형 답변 |
| **서비스** | FastAPI `GET /health`, `POST /predict` |
| **실행 환경** | Python 3.11, 오프라인 Docker, CPU |
| **기술** | Python, NumPy, PyTorch, Transformers, FastAPI, Uvicorn, Docker |
| **주의** | 법률 자문 또는 법적 판단을 제공하는 시스템이 아님 |

---

## Problem → Implementation → Result

| Problem | Implementation | Result |
|---|---|---|
| 법률 용어와 사건명이 정확히 일치하는 문서를 찾아야 함 | BM25 역색인과 필드 가중치 | 고유명사·법률 표현 중심 검색 |
| 같은 의미를 다른 표현으로 묻는 질의를 보완해야 함 | 다국어 MiniLM 평균 풀링 Dense Retriever | 의미 유사도 기반 후보 보완 |
| 질문 목적과 다른 섹션이 상위에 노출됨 | Query Classification + Section Boost | 주문·과징금·법리 등 질문 유형별 순위 보정 |
| 공식 ID가 아닌 임의 청크를 반환할 위험 | 원본 `chunk_id` 보존과 유효성 검사 | 존재하는 ID만 정확히 5개 반환 |
| 검색 결과와 답변 근거가 분리될 수 있음 | Extractive Answer + Evidence Trace | 답변 근거를 검색된 Top-5 안으로 제한 |
| 평가 환경에서 인터넷을 사용할 수 없음 | 모델·데이터·인덱스를 Docker 이미지에 포함 | `network_mode=none`에서 200/200 요청 성공 |
| 요청마다 인덱스를 다시 읽으면 느림 | FastAPI lifespan에서 모델과 인덱스 1회 로드 | 장기 실행 프로세스에서 반복 추론 |

---

## System Overview

<img src="./docs/assets/fair-decision-system-overview.png" alt="공정거래 의결서 검색 시스템 구성" width="100%">

```text
공개 의결서 청크
├─ BM25 Index
└─ Dense Embedding Index
          │
질문 ── Query Classification
          │
          ├─ BM25 Retrieval
          └─ Dense Retrieval
                  │
          Weighted Score Fusion
                  │
             Section Boost
                  │
      Unique and Valid Top-5
                  │
       Grounded Extractive Answer
                  │
 FastAPI: id + chunk_ids + answer
```

### Layer Responsibilities

| Layer | Responsibility |
|---|---|
| **Query Classifier** | 질문을 과징금, 법리, 사실관계, 법조문, 시정명령, 요약 등으로 분류 |
| **BM25 Retriever** | 사건명·기업명·법률 용어의 어휘 일치 기반 후보 검색 |
| **Dense Retriever** | 다국어 문장 임베딩의 코사인 유사도로 의미 기반 후보 검색 |
| **Score Fusion** | BM25와 Dense 점수를 정규화하고 가중 결합 |
| **Section Boost** | 질문 유형과 관련성이 높은 의결서 섹션의 순위 보정 |
| **Top-5 Validator** | 정확히 5개, 중복 없음, 공개 데이터에 존재하는 ID인지 검증 |
| **Answer Generator** | 검색된 근거 문장을 조합하고 출처 `chunk_id`를 함께 표시 |
| **Submission Service** | 모델과 인덱스를 1회 로드하고 공식 HTTP 스키마로 응답 |

---

## Competition Requirement Mapping

공모전 Track 2는 Retrieval 50%와 Generation 50%로 모델을 평가하며, Retrieval에는
Recall@5와 MRR, Generation에는 BERTScore와 Token F1을 사용합니다. 또한 Top-5
청크 ID 형식과 응답시간에 강제 규칙을 둡니다.

| Official Requirement | Project Implementation | Local Verification |
|---|---|---|
| 의결서 Chunking과 원본 ID 유지 | 제공 청크 31,877개와 원본 `chunk_id` 사용 | **PASS** |
| 임베딩 생성 | 384차원 다국어 MiniLM Dense Index | **PASS** |
| Hybrid Retrieval | BM25 + Dense weighted fusion | **PASS** |
| 근거 기반 생성 | Top-5 범위 안의 추출형 답변과 Evidence Trace | **PASS** |
| 정확히 5개 `chunk_id` | 응답 직전 길이 검증과 안전 fallback | **PASS** |
| 중복 ID 금지 | ID 기준 중복 제거 | **PASS** |
| 공개 데이터에 존재하는 ID | 전체 유효 ID 집합으로 검증 | **PASS** |
| 배열 순서가 검색 순위 | 최종 점수 내림차순으로 반환 | **PASS** |
| 문항당 30초 이하 | 오프라인 Docker 200회 최대 13.3913초 | **PASS** |
| 외부 LLM API 금지 | 네트워크 호출 없는 로컬 추론 | **PASS** |
| 8B 이하 생성 모델 | 별도 생성 LLM 없이 추출형 생성 | **PASS** |
| 인터넷 없는 평가 환경 | 모델·인덱스·데이터를 이미지에 포함 | **PASS** |
| `GET /health` | `{"status":"ok"}` 반환 | **PASS** |
| `POST /predict` | `id`, `retrieved_chunk_ids`, `answer` 반환 | **PASS** |

> 위 표의 PASS는 공개된 실행 규격에 대한 로컬 검증 결과입니다. 공모전 비공개
> 평가 데이터의 품질 점수나 심사 결과를 의미하지 않습니다.

---

## Evaluation

### 1. Silver QA Retrieval Evaluation

공개 의결서 메타데이터에서 자동 구성한 500개 silver QA로 측정한 결과입니다.
정답 라벨이 사람이 검수한 공식 gold set이 아니므로, 모델 간 개발 비교와 회귀
검증 용도로만 해석해야 합니다.

| Metric | Result |
|---|---:|
| Questions | **500** |
| Recall@5 | **0.9850** |
| MRR | **0.9810** |
| Token F1 | **0.0595** |
| BERTScore | 미산출 |
| Total Evaluation Time | **613.33 sec** |

결과 파일:
[`outputs/results/silver_hybrid_eval_500.json`](./outputs/results/silver_hybrid_eval_500.json)

### 2. Offline Docker Stability

전체 데이터, 검색 인덱스와 임베딩 모델을 포함한 이미지에 네트워크와 호스트
볼륨을 연결하지 않고 HTTP 요청 200개를 순차 전송했습니다.

| Metric | Result |
|---|---:|
| Requests | **200** |
| Passed / Failed | **200 / 0** |
| Mean Latency | **1.4503 sec** |
| P95 Latency | **4.2112 sec** |
| Maximum Latency | **13.3913 sec** |
| Network | `none` |
| Host Volume | 사용하지 않음 |

결과 파일:
[`outputs/results/docker_offline_stability_200.json`](./outputs/results/docker_offline_stability_200.json)

### 3. Result Interpretation

- 자체 silver set에서는 검색 대상 사건을 Top-5 안에 포함시키는 성능이 높았습니다.
- Token F1은 0.0595로 낮아, 추출형 답변 선택과 문장 압축에는 개선 여지가 큽니다.
- BERTScore는 선택 의존성과 공식 reference answer 부재로 산출하지 않았습니다.
- 200회 오프라인 HTTP 검증에서는 형식 오류와 제한시간 초과가 없었습니다.
- 공식 비공개 평가 점수는 없으며, 위 수치를 공모전 최종 성능으로 일반화할 수 없습니다.

---

## Retrieval and Answer Pipeline

### Query Types

| Query Type | Example Intent | Priority Sections |
|---|---|---|
| `penalty` | 과징금 금액과 산정 근거 | `penalty`, `order`, `legal_reasoning` |
| `legal_reasoning` | 위법성 판단 이유 | `legal_reasoning`, `law_article`, `fact` |
| `fact_pattern` | 문제된 행위와 사실관계 | `fact`, `legal_reasoning`, `summary` |
| `law_article` | 적용 법령과 조항 | `law_article`, `legal_reasoning` |
| `corrective_order` | 시정명령과 처분 내용 | `order`, `conclusion`, `penalty` |
| `summary` | 사건 개요와 핵심 내용 | `summary`, `fact`, `legal_reasoning` |
| `general` | 특정 유형에 속하지 않는 질문 | `summary`, `fact`, `legal_reasoning` |

### Ranking

```text
BM25 candidates
       +
Dense candidates
       ↓
score normalization
       ↓
weighted fusion
       ↓
query-aware section boost
       ↓
deduplicate → validate → Top-5
```

### Response Contract

Request:

```json
{
  "id": "question-001",
  "question": "과징금 산정 시 고려하는 요소는 무엇인가?"
}
```

Response:

```json
{
  "id": "question-001",
  "retrieved_chunk_ids": [
    "DOC-...-CH-030",
    "DOC-...-CH-041",
    "DOC-...-CH-032",
    "DOC-...-CH-049",
    "DOC-...-CH-085"
  ],
  "answer": "검색된 의결서 근거를 바탕으로 구성한 답변..."
}
```

---

## Run and Verify

### Prerequisites

- Python 3.11
- 전체 실행: 공개 의결서 `data/chunks.jsonl`
- Hybrid 실행: 로컬 임베딩 모델과 생성된 `indexes/`

저장소에는 즉시 검증할 수 있는 sample 데이터와 대응하는 BM25·Dense 인덱스가
포함되어 있습니다. 전체 데이터와 전체 인덱스는 용량 때문에 Git 저장 대상에서
제외되며, 공개 의결서 데이터를 내려받은 뒤 아래 빌드 스크립트로 준비합니다.

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

PyTorch CPU 패키지가 필요합니다.

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Build Indexes

```powershell
python .\scripts\build_bm25_index.py
python .\scripts\build_dense_index.py
```

### 3. CLI Query

```powershell
python .\run.py `
  --mode hybrid `
  --query "시장지배적 지위 남용행위의 판단 기준은 무엇인가?"
```

BM25만 확인하려면:

```powershell
python .\run.py `
  --mode bm25 `
  --query "부당한 공동행위에 대한 시정명령은 무엇인가?"
```

### 4. FastAPI

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Prediction:

```powershell
$body = @{
  id = "demo-001"
  question = "과징금 산정 시 고려하는 요소는 무엇인가?"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body $body
```

### 5. Docker

fresh clone에서는 sample 데이터와 인덱스를 기본으로 사용합니다. Docker 빌드 중
공개 임베딩 모델을 내려받아 이미지에 저장하므로, 빌드 후 실행에는 인터넷이
필요하지 않습니다.

```powershell
docker build -t rag-submission:latest .
docker run --rm `
  -p 8000:8000 `
  rag-submission:latest
```

또는:

```powershell
docker compose up --build
```

인터넷 차단 상태 자체를 검증할 때는 `--network none`으로 실행한 뒤 컨테이너
내부에서 `/health`와 `/predict`를 호출합니다. `none` 네트워크에서는 호스트 포트
공개가 되지 않으므로 일반적인 API 사용 시에는 위 명령처럼 포트만 연결합니다.

전체 데이터로 실행하려면 `data/chunks.jsonl`과 `indexes/`를 준비한 뒤 컨테이너의
`CHUNKS_PATH`, `BM25_INDEX_PATH`, `DENSE_INDEX_PATH`,
`DENSE_METADATA_PATH`를 전체 파일 경로로 지정합니다. 공모전 제출용 이미지에는
전체 데이터와 인덱스를 포함해 별도로 빌드했습니다.

다른 터미널에서:

```powershell
python .\scripts\benchmark_api.py `
  --base-url http://127.0.0.1:8000 `
  --requests 200
```

### 6. Tests

```powershell
python -m unittest discover -s tests -v
```

---

## Project Structure

```text
fair-decision-rag/
├─ data/
│  ├─ processed/             # silver QA와 평가용 데이터
│  ├─ sample/                # 경량 실행용 청크
│  └─ chunks.jsonl           # 전체 공개 청크, Git 제외
├─ docs/
│  └─ assets/                # README 시스템 이미지
├─ indexes/                  # BM25와 Dense Index, Git 제외
│  └─ sample/                # fresh clone 실행용 경량 인덱스
├─ models/                   # 로컬 임베딩 모델, Git 제외
├─ outputs/
│  └─ results/               # 평가 및 벤치마크 결과
├─ scripts/
│  ├─ build_bm25_index.py
│  ├─ build_dense_index.py
│  ├─ build_sample_dataset.py
│  ├─ build_silver_qa.py
│  ├─ evaluate_pipeline.py
│  ├─ benchmark_submission.py
│  └─ benchmark_api.py
├─ src/
│  ├─ retrieval/
│  │  ├─ bm25_retriever.py
│  │  ├─ dense_retriever.py
│  │  ├─ hybrid_retriever.py
│  │  ├─ query_classifier.py
│  │  ├─ section_boost.py
│  │  ├─ answer_generator.py
│  │  ├─ evidence_trace.py
│  │  └─ evaluator.py
│  ├─ runtime.py
│  └─ submission_service.py
├─ tests/
│  └─ test_core.py
├─ Dockerfile
├─ docker-compose.yml
├─ run.py
├─ server.py
└─ requirements.txt
```

---

## Current Scope and Limitations

### Current Scope

- 공정거래위원회 공개 의결서 청크 31,877개
- BM25와 다국어 Dense Embedding을 결합한 Hybrid Retrieval
- 질문 유형 분류와 의결서 Section Boost
- 원본 ID 기반 중복 없는 Top-5 검증
- 검색 근거 범위 안의 추출형 답변과 Evidence Trace
- Python 3.11 FastAPI 서비스
- 외부 API 없는 오프라인 Docker 실행
- silver QA 검색 평가와 HTTP 안정성 벤치마크

### Limitations

- 공모전 비공개 gold set의 공식 평가 결과가 없습니다.
- 500개 QA는 메타데이터에서 자동 생성한 silver label이며 질문 유형이 편중될 수 있습니다.
- 추출형 답변은 관련 청크를 찾더라도 질문에 맞는 핵심 문장을 충분히 압축하지 못할 수 있습니다.
- Token F1이 낮고 BERTScore를 산출하지 않아 Generation 품질 검증이 제한적입니다.
- Cross-encoder re-ranker와 별도 생성 LLM은 포함하지 않았습니다.
- OCR 오류, 표, 각주와 긴 법률 문장에 검색·답변 오류가 전파될 수 있습니다.
- 결과는 의결서 탐색 보조 자료이며 법률 자문이나 최종 판단으로 사용할 수 없습니다.

### Possible Improvements

1. 사람이 검수한 다양한 유형의 gold QA 구축
2. BM25, Dense, Section Boost 각각의 ablation 평가
3. 경량 cross-encoder re-ranking
4. 주문·과징금·법리별 answer span selector 개선
5. 공식 지표와 동일한 BERTScore·Token F1 평가 환경 구성
6. 사건 단위 다양성과 인접 청크 문맥을 고려한 재정렬
7. 검색 실패와 근거 부족을 감지하는 abstention 정책

---

## What This Project Demonstrates

- 긴 공공 법률 문서를 검색 가능한 청크와 인덱스로 구성
- 어휘 검색과 의미 검색을 결합한 Hybrid Retrieval 구현
- 질문 의도를 의결서 구조에 연결하는 Section-aware ranking 설계
- 원본 `chunk_id` 무결성과 Top-5 출력 규칙을 코드로 강제
- 검색 결과와 답변 근거를 연결하는 grounded answer 구조
- 모델, 데이터와 인덱스를 포함한 오프라인 Docker serving
- Recall@5, MRR, Token F1과 HTTP latency를 분리해 평가
- 자체 평가의 한계와 공모전 공식 평가의 차이를 명시

---

## References

- [공정거래위원회 제2회 「공정거래 데이터」 활용 공모전](https://www.fairdata.go.kr/aic/contestInfo.do?#tab-track-nav3)
- [공정거래위원회 심결·법령 의결서 검색](https://case.ftc.go.kr/)
- [Sentence Transformers: paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

---

## Contact

- Developer: 김수진
- GitHub: [lightleaping](https://github.com/lightleaping)
- Email: workingskyroad@gmail.com
