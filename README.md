# 공정거래 의결서 NLP 검색

**Hybrid Retrieval, Query Classification, Section Boost, and Evidence Trace**

> 공정거래위원회 공개 의결서에서 질문과 관련된 근거 구간을 검색하고, 중복 없는 Top-5 `chunk_id`로 원문 위치를 추적하도록 설계한 문서 검색형 RAG 프로젝트입니다.

<p>
  <img src="https://img.shields.io/badge/Python-NLP-3561D8?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense-21AFC4?style=flat-square" alt="Hybrid Retrieval">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-151F32?style=flat-square" alt="FAISS">
  <img src="https://img.shields.io/badge/Evidence-Top--5%20chunk__id-3561D8?style=flat-square" alt="Evidence Trace">
</p>

---

## Why This Project

공정거래 의결서는 사실관계, 판단 이유, 주문, 법적 근거처럼 서로 다른 역할의 Section으로 구성된 긴 공공문서입니다.

단순 키워드 검색만 적용하면 다음 문제가 생길 수 있습니다.

- 질문의 목적과 다른 Section이 상위 결과에 노출될 수 있습니다.
- 비슷한 문장이 반복되어 동일한 근거가 여러 번 반환될 수 있습니다.
- 검색 결과가 어느 문서의 어느 구간인지 다시 찾기 어렵습니다.
- 생성된 설명과 실제 원문 근거 사이의 연결이 불명확해질 수 있습니다.

따라서 이 프로젝트는 답변 문장을 먼저 만드는 것보다, 질문과 관련된 근거 구간을 검색하고 원문 ID로 추적 가능하게 만드는 것을 우선 목표로 두었습니다.

---

## Project Overview

| 항목 | 내용 |
|---|---|
| **기간** | 2026.05 |
| **형태** | 팀 프로젝트 |
| **대상 문서** | 공정거래위원회 공개 의결서 |
| **목표** | 질문 목적에 맞는 근거 구간을 검색하고 중복 없는 Top-5 `chunk_id` 반환 |
| **전체 범위** | 문서 전처리, Section-aware Chunking, BM25, Dense Retrieval, FAISS, Hybrid Fusion, Query Classification, Section Boost, Top-K Selection, 평가, Answer와 Evidence 연결 |
| **기술** | Python, pandas, BM25, Sentence Embedding, FAISS, JSON |
| **핵심 출력** | Query Type, Top-5 Chunk IDs, Section Type, Score, Evidence |
| **주의** | 법률 판단이나 법률 자문을 제공하는 시스템이 아님 |

---

## Project Scope and Contribution

### Overall Team Scope

```text
Public Decision Documents
→ Preprocessing and Section-aware Chunking
→ BM25 and Dense Retrieval
→ Hybrid Score Fusion
→ Query Classification and Section Boost
→ Duplicate Removal and Top-5 Selection
→ Retrieval Evaluation
→ Answer and Evidence Output
```

### My Contribution

- 질문 목적을 구분하는 Query Classification 기준 설계
- 질문 유형별 우선 Section 정의
- Dense Retrieval Baseline 구성
- Section Boost Score 보정 로직 구현
- 중복 없는 Top-K 선택 규칙과 `chunk_id` 검증
- 수동 평가, 검색 실패 유형 분석과 Gold Evidence 검토
- 팀 결과물 통합과 README 정리
- 팀 리드로 일정 지연 가능성을 조기에 공유하는 협업 기준 설정
- 일정 변동 발생 시 일부 구현 범위를 재분배해 MVP 일정 영향 축소

> 전처리와 Chunking, BM25, Fusion, 자동 평가와 Answer 구성은 팀 전체 범위에 포함되며, 위 기여 항목은 제가 직접 담당하거나 통합한 부분을 구분해 적었습니다.

---

## Problem → Implementation → Result

| Problem | Implementation | Result |
|---|---|---|
| 긴 문서에서 관련 단어는 찾지만 질문 목적과 다른 Section이 노출됨 | Query Type을 분류하고 Section별 가중치 적용 | 사실관계, 판단, 주문, 법적 근거의 우선순위를 질문에 맞게 보정 |
| Keyword Search만으로 표현이 다른 관련 문장을 놓칠 수 있음 | BM25와 Dense Retrieval 후보를 함께 사용 | 어휘 일치와 의미 유사도를 함께 고려하는 구조 구성 |
| 유사한 근거가 반복되어 Top-K가 단조로워짐 | `chunk_id` 기준 중복 제거 후 Top-5 선택 | 동일 근거의 반복 노출 방지 |
| 검색 결과를 원문에서 다시 찾기 어려움 | 기존 공개 데이터의 `chunk_id`를 유지 | Evidence Trace가 가능한 출력 구조 설계 |
| 생성 답변만 평가하면 검색 실패 원인을 찾기 어려움 | Retrieval Metric과 구조 검증을 분리 | 관련성, Section 적합성, 중복, ID 유효성을 각각 확인 |
| 팀 일정 변화가 전체 통합을 지연시킬 수 있음 | 지연 공유 원칙과 범위 재분배 | MVP 통합 범위를 유지하며 작업 연결 |

---

## System Overview

<img src="./docs/assets/fair-decision-system-overview.png" alt="공정거래 의결서 NLP 검색 시스템 구성도" width="100%">

### Search Responsibilities

| Layer | Responsibility |
|---|---|
| **Document** | 공개 의결서의 문서와 Section 구조 유지 |
| **Chunking** | 검색 단위 생성, `section_type`과 `chunk_id` 부여 또는 보존 |
| **BM25** | 질문과 문서의 Keyword 일치 기반 후보 검색 |
| **Dense Retrieval** | Sentence Embedding과 FAISS 기반 의미 유사도 검색 |
| **Hybrid Fusion** | 두 검색 결과의 Score를 정규화하고 병합 |
| **Query Classification** | 질문이 사실관계, 판단 이유, 제재 결과, 법적 근거 중 무엇을 묻는지 구분 |
| **Section Boost** | 질문 목적에 맞는 Section의 순위를 보정 |
| **Top-K Selector** | 중복을 제거하고 Top-5 `chunk_id` 선택 |
| **Evidence Trace** | 선택한 근거를 원문 구간과 연결 |

---

## Query Classification and Section Boost

질문의 목적에 따라 우선 확인할 Section이 다릅니다.

| Query Purpose | Priority Section Example |
|---|---|
| 위반 행위와 사실관계 | 사실관계, 행위 내용 |
| 판단 이유 | 판단, 법리 |
| 제재 결과 | 주문, 조치 내용 |
| 적용 법령 | 법적 근거 |

```text
Question
→ Query Type Classification
→ BM25 and Dense Candidates
→ section_type 확인
→ Query-aware Section Boost
→ Re-ranking
```

Section Boost는 새로운 근거를 생성하지 않습니다. 검색된 후보의 순위를 질문 목적에 맞게 보정하는 단계입니다.

---

## Top-5 Evidence Rule

최종 검색 결과는 다음 규칙을 목표로 합니다.

```python
assert len(chunk_ids) == 5
assert len(set(chunk_ids)) == 5
assert all(chunk_id in source_chunk_ids for chunk_id in chunk_ids)
```

추가 원칙:

- 새 `chunk_id`를 임의로 만들지 않음
- 원문 데이터에 존재하는 ID만 반환
- 동일 근거 반복을 줄이기 위해 중복 제거
- Answer보다 Evidence 목록을 먼저 확정
- Query Type, Section Type, Score와 Chunk ID를 함께 저장

Example Structure:

```json
{
  "question": "시장지배적 지위 남용 판단 이유는 무엇인가?",
  "query_type": "decision_reason",
  "chunk_ids": [
    "decision_001_judgment_03",
    "decision_001_judgment_05",
    "decision_014_legal_02",
    "decision_014_judgment_04",
    "decision_027_judgment_01"
  ],
  "evidence": [
    {
      "chunk_id": "decision_001_judgment_03",
      "section_type": "judgment",
      "score": 0.82
    }
  ]
}
```

> 위 ID와 Score는 Response 구조를 설명하는 예시입니다. 실제 결과에는 실행 Artifact의 값을 사용해야 합니다.

---

## Practical Evaluation Criteria

<img src="./docs/assets/fair-decision-evaluation-flow.png" alt="공정거래 의결서 검색 평가 흐름" width="100%">

| 실무 관점 | 평가 기준 | 해석 |
|---|---|---|
| **관련 근거 포함** | Recall@5 | 정답 Evidence가 Top-5 안에 포함되는지 확인 |
| **정답 순위** | MRR | 첫 번째 관련 근거가 얼마나 위에 나타나는지 확인 |
| **Section 적합성** | Section Hit Rate | 질문 목적에 맞는 Section이 검색되는지 확인 |
| **중복 관리** | Duplicate Rate | 동일 Chunk가 반복되는 비율 확인 |
| **추적 가능성** | Valid Chunk ID Rate | 반환 ID가 실제 원문 데이터에 존재하는지 확인 |
| **검색 방식 비교** | BM25, Dense, Hybrid, Boost Ablation | 어떤 단계가 검색 품질에 기여했는지 비교 |
| **실패 분석** | 누락, Section 불일치, 표현 차이, 유사 근거 중복 | 단일 평균 점수로 보이지 않는 오류 유형 확인 |

> 현재 공개 README에서 확인 가능한 검증은 **정확히 5개, 중복 없는 `chunk_id`, 원문 ID 보존, 질문 유형에 따른 Section Boost**입니다. Recall@5와 MRR의 확정 수치는 평가 Artifact가 공개된 뒤 기재해야 합니다.

---

## Technical Details

<details open>
<summary><b>01 | Document and Chunk Structure</b></summary>

<br>

권장 Chunk Schema:

```json
{
  "document_id": "decision_001",
  "chunk_id": "decision_001_judgment_03",
  "section_type": "judgment",
  "text": "의결서 원문 구간",
  "source": "공개 의결서"
}
```

검색과 재정렬 단계에서 `chunk_id`와 `section_type`을 잃지 않는 것이 중요합니다.

</details>

<details>
<summary><b>02 | Hybrid Retrieval</b></summary>

<br>

```text
Question
├── BM25 Keyword Retrieval
└── Dense Retrieval with FAISS
        ↓
Score Normalization
        ↓
Candidate Merge
        ↓
Query-aware Section Boost
```

BM25는 명확한 법령명, 행위명, 기업명처럼 단어 일치가 중요한 질문에 유리합니다. Dense Retrieval은 질문과 문서가 다른 표현을 사용해도 의미가 비슷한 후보를 찾는 데 사용합니다.

</details>

<details>
<summary><b>03 | Top-K Selection</b></summary>

<br>

Top-K Selector는 단순히 Score 상위 5개를 자르는 것이 아니라 다음을 확인합니다.

1. Score 순서 정렬
2. `chunk_id` 기준 중복 제거
3. 원문 ID 존재 여부 확인
4. 최대 5개까지 선택
5. 부족할 경우 다음 후보에서 보충
6. 결과 JSON 저장

</details>

<details>
<summary><b>04 | Failure Analysis</b></summary>

<br>

| Failure Type | Description | Improvement |
|---|---|---|
| 검색 누락 | 정답 근거가 후보에 포함되지 않음 | Query Rewrite, Dense Model, Chunk 범위 검토 |
| Section 불일치 | 관련 단어는 있으나 질문 목적과 다른 Section | Query Classification과 Boost 규칙 개선 |
| 표현 차이 | 법률 용어와 일반 질문 표현이 다름 | 동의어와 Dense Retrieval 보완 |
| 유사 근거 중복 | 같은 문맥의 Chunk가 여러 번 노출 | Document와 Section 기준 Diversity 적용 |
| 긴 근거 분할 | 하나의 판단 근거가 여러 Chunk로 끊김 | Overlap과 Section-aware Chunking 조정 |
| Gold 불일치 | 평가자별 정답 근거 판단이 다름 | Gold Review 기준과 근거 범위 명시 |

</details>

<details>
<summary><b>05 | Current Public Status</b></summary>

<br>

현재 공개 저장소에서 바로 확인되는 범위는 README와 `docs/assets/`입니다.

따라서 Source Code와 실행 Artifact가 공개되기 전에는 다음을 성능 완료 사실로 단정하지 않습니다.

- 전체 Hybrid Retrieval 재실행
- Recall@5와 MRR 확정 수치
- 자동 평가 결과
- Grounded Answer 정량 평가

Source를 공개할 때는 README의 경로, 실행 명령과 평가 수치를 실제 Repository 상태에 맞춰 갱신해야 합니다.

</details>

<details>
<summary><b>06 | Next Steps</b></summary>

<br>

1. 전처리, Retrieval과 Evaluation Source 공개
2. Gold Question과 Evidence 평가 Dataset 공개
3. BM25, Dense, Hybrid, Section Boost Ablation
4. Recall@5, MRR, nDCG@5, Section Hit Rate 산출
5. Cross-encoder Re-ranking 비교
6. Evidence Highlight와 원문 Section Link
7. Answer와 Evidence의 Citation Consistency 검사
8. OCR과 문서 형식 변화에 대한 Robustness 검증

</details>

---

## Suggested Repository Structure

```text
fair-decision-rag/
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── docs/
│   └── assets/
├── outputs/
│   └── results/
├── src/
│   ├── preprocessing/
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── query_classifier.py
│   │   ├── section_boost.py
│   │   ├── hybrid_retriever.py
│   │   └── topk_selector.py
│   ├── evaluation/
│   └── answer/
├── tests/
├── README.md
└── requirements.txt
```

> 위 구조는 전체 프로젝트 범위를 공개할 때 사용할 권장안입니다. 현재 공개 저장소에 없는 경로를 이미 존재하는 것처럼 표시하지 않습니다.

---

## What This Project Demonstrates

- 긴 공공문서를 Section과 Chunk 단위로 검색하는 NLP 문제 정의 경험
- BM25와 Dense Retrieval의 역할을 구분하고 Hybrid Search로 연결한 경험
- 질문 목적에 맞는 Query Classification과 Section Boost 구현 경험
- 중복 없는 Top-K와 원문 `chunk_id` 보존 규칙 설계 경험
- Recall@5와 MRR을 포함한 검색 평가 기준 이해
- 정량 지표와 수동 실패 분석을 함께 수행한 경험
- 팀 리드로 역할 연결, 일정 공유와 구현 범위 재분배를 수행한 경험
- 본인 기여와 팀 전체 구현 범위를 구분해 문서화한 경험

---

## Contact

- Developer: 김수진
- GitHub: [github.com/lightleaping](https://github.com/lightleaping)
- Email: workingskyroad@gmail.com
