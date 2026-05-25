# Day 3 — A 작업 결과: section_type 정리 기준

## 1. Day 3 목표

공정거래 의결서 chunk의 section_type을 검색에 활용할 수 있도록 정리하고, 질문 유형별로 어떤 section을 우선 검색해야 하는지 규칙을 만든다.

## 2. 사용할 section_type

본 프로젝트에서는 section_type을 다음 다섯 가지로 정리한다.

- 주문
- 이유
- 별지
- 결론
- 기타

## 3. section_type별 의미

| section_type | 의미 | 주로 담기는 정보 |
|---|---|---|
| 주문 | 공정위가 최종적으로 내린 결정 | 시정명령, 과징금, 고발, 처분 결과 |
| 이유 | 왜 그런 판단을 했는지 설명 | 사실관계, 위반 행위, 법적 판단 근거 |
| 별지 | 세부 산정표나 부가 자료 | 과징금 산정, 업체 목록, 세부 금액 |
| 결론 | 판단의 최종 정리 | 위반 인정 여부, 종합 판단 |
| 기타 | 위 항목으로 분류하기 어려운 부분 | 표지, 목차, 절차 설명 등 |

## 4. 질문 유형별 우선 section_type

| 질문 유형 | 예시 질문 | 우선 section_type |
|---|---|---|
| 과징금 질문 | 과징금은 얼마인가요? | 주문, 별지 |
| 처분 결과 질문 | 공정위는 어떤 조치를 내렸나요? | 주문 |
| 위반 이유 질문 | 왜 위반이라고 판단했나요? | 이유 |
| 행위 패턴 질문 | 어떤 행위가 문제였나요? | 이유 |
| 법 조항 질문 | 어떤 법 조항을 위반했나요? | 이유 |
| 사건 요약 질문 | 이 사건을 요약해줘. | 주문, 이유 |

## 5. section_type 추정 우선순위

section_type이 없는 chunk는 다음 순서로 추정한다.

1. 제목 패턴
2. 문단 시작 표현
3. 키워드 밀도
4. 판단 불가 시 기타

## 6. 제목 기반 규칙

다음 표현이 chunk 앞부분에 등장하면 해당 section으로 분류한다.

| 표현 | section_type |
|---|---|
| 주문, 주 문 | 주문 |
| 이유, 이 유 | 이유 |
| 별지 | 별지 |
| 결론 | 결론 |

## 7. 키워드 기반 보조 규칙

제목만으로 판단하기 어려운 경우 다음 키워드를 참고한다.

### 주문 관련 키워드

- 시정명령
- 과징금
- 고발
- 납부명령
- 부과한다

### 이유 관련 키워드

- 사실관계
- 판단
- 인정된다
- 위반
- 부당한 공동행위
- 담합

### 별지 관련 키워드

- 산정기준
- 산정내역
- 부과기준율
- 관련매출액

### 결론 관련 키워드

- 종합하면
- 결론적으로
- 이상과 같이

## 8. section_type 추정 로직 초안

```python
def detect_section_type(text: str) -> str:
    head = text[:100]

    if "주문" in head or "주 문" in head:
        return "주문"

    if "이유" in head or "이 유" in head:
        return "이유"

    if "별지" in head:
        return "별지"

    if "결론" in head:
        return "결론"

    order_keywords = ["시정명령", "과징금", "고발", "납부명령", "부과한다"]
    reason_keywords = ["사실관계", "판단", "인정된다", "위반", "부당한 공동행위", "담합"]
    appendix_keywords = ["산정기준", "산정내역", "부과기준율", "관련매출액"]
    conclusion_keywords = ["종합하면", "결론적으로", "이상과 같이"]

    order_score = sum(1 for kw in order_keywords if kw in text)
    reason_score = sum(1 for kw in reason_keywords if kw in text)
    appendix_score = sum(1 for kw in appendix_keywords if kw in text)
    conclusion_score = sum(1 for kw in conclusion_keywords if kw in text)

    scores = {
        "주문": order_score,
        "이유": reason_score,
        "별지": appendix_score,
        "결론": conclusion_score
    }

    best_section = max(scores, key=scores.get)

    if scores[best_section] == 0:
        return "기타"

    return best_section