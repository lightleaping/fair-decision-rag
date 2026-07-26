# 공정거래 의결서 근거 기반 RAG

공정거래위원회 제2회 AI·데이터 활용 공모전 Track 2를 위한 오프라인
질의응답 파이프라인입니다. 공정위가 제공한 공개 의결서의 기존
`chunk_id`를 보존하고, 검색·답변·평가 결과를 끝까지 추적합니다.

## 구현 범위

- 공식 공개본 500개 문서, 31,877개 청크 전처리
- 한국어 word/bi-gram BM25
- 다국어 MiniLM sentence embedding Dense Retrieval
- BM25 + Dense score fusion
- 질문 유형 분류와 의결서 section boost
- 중복 없는 Top-5 공식 `chunk_id`
- 근거 문장만 사용하는 인용형 extractive generation
- Recall@5, MRR, token-level F1
- 답변이 Top-5 밖의 근거를 사용하지 않는 Evidence Trace
- 외부 API 없는 단일 CLI
- 오프라인 Docker 실행 구조

## 빠른 실행

PowerShell에서:

```powershell
cd C:\Users\kflow\Downloads\fair-decision-rag
py scripts\build_chunks_jsonl.py
py run.py --mode bm25 --query "과징금은 어떻게 산정되었나요?"
```

`data/chunks.jsonl`이 이미 있다면 첫 번째 명령은 반복할 필요가 없습니다.

## Dense / Hybrid 준비

Python 3.11 환경에서 프로젝트 전용 의존성을 설치합니다.

```powershell
py -m pip install --target .vendor -r requirements.txt
```

임베딩 모델은 평가 중 다운로드되지 않도록 로컬에 준비해야 합니다. 기본
모델은 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`이며
환경변수로 로컬 디렉터리를 지정할 수 있습니다.

```powershell
$env:EMBEDDING_MODEL_PATH="C:\path\to\paraphrase-multilingual-MiniLM-L12-v2"
py -m scripts.build_dense_index --batch-size 32 --max-seq-length 64
```

인덱스 생성 후:

```powershell
py run.py --mode hybrid --query "피심인의 위반 행위와 위법성 판단 근거는 무엇인가요?"
```

생성 파일:

```text
indexes/dense_embeddings.npy
indexes/dense_chunks.jsonl
```

## 평가

공식 청크에서 문서당 하나의 질문을 만든 500건 silver QA는 로컬 회귀검증
전용입니다. 숨겨진 공모전 평가 점수로 간주하지 않습니다.

```powershell
py -m scripts.build_silver_qa
py -m scripts.evaluate_pipeline --mode bm25
py -m scripts.evaluate_pipeline --mode hybrid `
  --output outputs/results/silver_hybrid_eval.json
```

공식 또는 수작업 검수 QA는 다음 JSONL 형식으로 교체할 수 있습니다.

```json
{
  "query": "질문",
  "gold_chunk_ids": ["DOC-...-CH-001"],
  "answer_reference": "정답 답변"
}
```

테스트:

```powershell
py -m unittest discover -v
```

## Windows 로컬 실행 (권장)

Python 3.11 환경을 만들고 CPU 전용 PyTorch를 설치합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install `
  --index-url https://download.pytorch.org/whl/cpu `
  torch==2.13.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

임베딩 모델을 한 번 내려받아 Hugging Face 캐시에 보관한 뒤에는 아래처럼
오프라인 모드로 전체 500개 의결서와 31,877개 청크를 검색할 수 있습니다.

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe run.py `
  --mode hybrid `
  --chunks data/chunks.jsonl `
  --dense-index indexes/dense_embeddings.npy `
  --dense-metadata indexes/dense_chunks.jsonl `
  --query "시장지배적 지위 남용행위의 판단 기준은 무엇인가?" `
  --output outputs/results/local_full_hybrid_query.json
```

모델이 기본 캐시 위치가 아닌 곳에 있으면
`EMBEDDING_MODEL_PATH`에 로컬 모델 디렉터리를 지정합니다. 실행 결과의
`dense_available`과 `trace_valid`가 모두 `true`인지 확인합니다.

## Docker (샘플 재현용)

저장 공간이 제한된 개발 환경에서는 먼저 문서 단위 샘플을 만듭니다. 기본값은
50개 의결서이며, 한 문서에 속한 청크는 모두 유지하고 기존 Dense 인덱스의 같은
행만 추출하므로 공식 chunk ID와 인덱스 정렬이 보존됩니다.

```powershell
python scripts/build_sample_dataset.py --documents 50
```

`docker-compose.yml`은 기본적으로 `data/sample/chunks.jsonl`과
`indexes/sample`을 사용합니다. 전체 데이터로 실행하려면 다음 환경 변수를
지정합니다.

```powershell
$env:CHUNKS_SOURCE="./data/chunks.jsonl"
$env:INDEXES_SOURCE="./indexes"
docker compose run --rm fair-decision-rag
```

## Track 2 공식 API

제출 이미지의 기본 명령은 장기 실행 FastAPI 서버를 시작합니다. 전체 BM25, Dense
인덱스와 임베딩 모델은 시작할 때 한 번만 로드되고 이후 질의에서는 재사용됩니다.

```text
GET  /health
POST /predict
```

```json
{
  "id": "eval_0001",
  "question": "시장지배적 지위 남용행위의 판단 기준은 무엇인가?"
}
```

응답은 공식 가이드가 요구하는 세 필드만 포함합니다.

```json
{
  "id": "eval_0001",
  "retrieved_chunk_ids": ["정확히 5개의 고유한 공개본 chunk_id"],
  "answer": "검색된 공개본 의결서에 근거한 답변"
}
```

로컬 서버 실행과 검증:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app `
  --host 127.0.0.1 --port 8000 --workers 1

Invoke-RestMethod http://127.0.0.1:8000/health
```

전체 데이터 응답시간 벤치마크:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_submission
```

제출 이미지는 전체 데이터, 인덱스 및 `models/embedding`을 포함합니다.

```powershell
docker build -t rag-submission:latest .
docker save rag-submission:latest -o submission.tar
```

컨테이너는 네트워크 없이 실행되도록 설정되어 있습니다. 다음 세 경로를
준비합니다.

```text
data/chunks.jsonl
indexes/dense_embeddings.npy
indexes/dense_chunks.jsonl
models/embedding/...
```

`models/embedding`에는 SentenceTransformer 모델 파일 전체를 복사합니다.
공간이 부족한 개발 환경에서는 `EMBEDDING_MODEL_SOURCE`로 기존 모델
디렉터리를 직접 마운트할 수 있습니다.

```powershell
docker compose build
docker compose run --rm fair-decision-rag
```

```powershell
$env:EMBEDDING_MODEL_SOURCE="C:\path\to\paraphrase-multilingual-MiniLM-L12-v2"
docker compose run --rm fair-decision-rag
```

## 출력 계약

`run.py`는 다음 정보를 JSON으로 반환합니다.

- `top5_chunk_ids`: 중복 없는 공식 청크 ID 5개
- `answer`: 각 문장 끝에 근거 `chunk_id`가 붙은 답변
- `evidence_chunk_ids`: 답변에 실제 사용된 근거
- `trace_rules`: Top-5 외 근거 사용 여부와 외부 API 사용 여부
- `dense_available`: 실제 Dense index 사용 여부
- `elapsed_seconds`: 모델·인덱스 로딩을 포함한 총 실행시간

## 공모전 준수 원칙

- 생성 모델은 8B 이하
- 외부 LLM API 호출 금지
- 평가 환경에서 모델 다운로드 금지
- 공정위 제공 `chunk_id` 유지
- 공개본 의결서 청크에 근거한 답변
- Retrieval: Recall@5, MRR
- Generation: BERTScore, token-level F1

현재 기본 생성기는 별도의 LLM 없이 동작하는 grounded extractive
generator입니다. 따라서 8B 제한과 외부 API 금지를 충족합니다.
BERTScore는 공식 reference answer와 평가용 모델이 제공되는 환경에서
추가 산출하며, 값이 없을 때 임의 수치를 만들지 않습니다.

## 검증 결과

공식 데이터 전체로 생성한 500개 silver QA 회귀평가 결과:

| 항목 | 결과 |
|---|---:|
| 문서 | 500 |
| 청크 | 31,877 |
| Dense index | 31,877 × 384 |
| Recall@5 | 0.985 |
| MRR | 0.981 |
| token F1 | 0.0595 |
| Evidence Trace | 통과 |

silver 질문은 사건명과 주문 청크로 자동 구성되어 실제 사용자의 다양한
질문보다 쉽습니다. 따라서 위 수치는 파이프라인 회귀검증에만 사용하며
공식 성능으로 제출하지 않습니다. 낮은 token F1은 extractive generator와
자동 reference 사이의 표현 차이가 크다는 사실을 보여주며, 임의로
보정하지 않았습니다.

이 프로젝트는 공개 의결서 검색·설명 도구이며 법률 자문을 제공하지
않습니다.
