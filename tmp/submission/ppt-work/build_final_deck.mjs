import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "C:/Users/kflow/Downloads/fair-decision-rag/tmp/submission/ppt-work";
const source = `${workspace}/template-starter.pptx`;
const output = "C:/Users/kflow/Downloads/fair-decision-rag/docs/submission/fair_decision_rag_presentation.pptx";
const renderDir = `${workspace}/final-render`;

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const edits = [
  ["sh/wvixo3et", "팀명: lightleaping\nFair Decision RAG"],
  ["sh/xwrex8vy", "\n\n\n발표자: 팀 대표자"],
  ["sh/zu9gnahs", "발표 순서"],
  ["sh/epk3ulw3", "1. 문제와 목표\n\n2. 사용자와 활용범위\n\n3. 전체 아키텍처\n\n4. Hybrid Retrieval"],
  ["sh/udsfedkr", "5. 근거 기반 답변\n\n6. 공식 API와 오프라인 실행\n\n7. 성능·안정성 검증\n\n8. 기대효과와 확장"],
  ["sh/sjah8bax", "의결서를 찾는 데서 끝나지 않고\n판단 근거까지 확인해야 합니다"],
  ["sh/3ytgz21k", "문제\n\n• 의결서는 길고 법률 문맥이 복잡함\n• 키워드 검색만으로 관련 근거를 놓침\n• 범용 AI 답변은 출처와 정확성 확인이 어려움"],
  ["sh/jypc3ehg", "목표\n\n• 공개본 의결서만 근거로 답변\n• 관련 chunk_id 5개를 순위대로 제공\n• 국민·기업·정책 담당자가 검증 가능한 결과\n• 문항당 30초 이내 오프라인 응답"],
  ["sh/54vahsfi", "전체 의결서를 한 번 로드하고\n질의마다 검색과 답변만 수행합니다"],
  ["sh/0fupwrad", "사전 처리\n\n500개 공개본 의결서\n→ 원본 chunk_id 보존\n→ 31,877개 청크\n→ BM25 + 384차원 Dense 인덱스"],
  ["sh/4r610jm5", "실시간 처리\n\n질문 유형 분류\n→ BM25·Dense 후보 검색\n→ 가중 결합·구역 보정\n→ 상위 5개 근거\n→ 추출형 답변과 Evidence Trace"],
  ["sh/3adgnadw", "Hybrid Retrieval은 정확한 근거를\n상위 순위에 배치합니다"],
  ["sh/ax4nudkn", "BM25 70%\n\n• 법률 용어와 정확한 표현에 강함\n• 사건명·주문·이유 필드 가중\n• 사전 저장 인덱스로 빠른 기동"],
  ["sh/q9sryl4z", "Dense 30%\n\n• 다국어 MiniLM, 384차원\n• 의미가 비슷한 표현을 보완\n• 질문 유형별 section boost\n• 결과 ID 중복·존재 여부 최종 검증"],
  ["sh/q9w325c3", "답변은 검색된 공개본 근거를\n벗어나지 않습니다"],
  ["sh/xkje5gvq", "근거 기반 생성\n\n• 상위 5개 청크에서 관련 문장 선택\n• 핵심 법률 표현을 유지\n• 답변에 근거 chunk_id 연결\n• 근거가 부족하면 제한적으로 응답"],
  ["sh/p0rmp8jm", "공식 API\n\nGET /health\nPOST /predict\n\n응답 필드\nid\nretrieved_chunk_ids: 정확히 5개\nanswer"],
  ["sh/xwnap87u", "전체 데이터와 오프라인 Docker에서\n응답 제한을 충족했습니다"],
  ["sh/0fe98v2l", "검색 회귀 평가\n\nSilver QA 500문항\nRecall@5  0.985\nMRR         0.981\n\n※ 비공개 공식 평가 점수가 아닌\n로컬 회귀 검증 결과"],
  ["sh/4fqtg3a1", "Docker 안정성 평가\n\n200/200 요청 성공\n평균 1.45초\np95 4.21초\n최대 13.39초\n\n네트워크 none·호스트 볼륨 없음"],
  ["sh/lw76xo3m", "공식 Track 2 실행 규칙을\n제출 이미지 안에서 충족했습니다"],
  ["sh/w7m9wb65", "재현성\n\n• Python 3.11\n• 외부 API 없음\n• 모델·청크·인덱스 이미지 내장\n• submission.tar 로드 성공\n• HTTP UTF-8 스키마 검증"],
  ["sh/gn698be5", "안전장치\n\n• 고유한 공개본 ID 5개 강제\n• 순위 순서 유지\n• 결정적 추론\n• 잘못된 입력은 422 응답\n• 120초 health start period"],
  ["sh/x0vqtgru", "공정거래 판단을 더 빠르게 찾고,\n더 쉽게 검증할 수 있습니다"],
  ["sh/sf6lwjmh", "국민·소상공인\n\n• 유사 의결서와 판단 기준 탐색\n• 답변에서 원문 근거 즉시 확인\n• 전문 용어 접근성 개선"],
  ["sh/0ne5wnu5", "정책·행정·연구\n\n• 반복 조사 시간 절감\n• 동일 질문에 일관된 근거 제공\n• 향후 reranker·경량 생성모델로 확장\n\n핵심: 빠른 답보다 검증 가능한 답"],
];
for (const [id, value] of edits) {
  presentation.resolve(id).text = value;
}

const sourceBlock =
  "[Sources]\n" +
  "https://www.fairdata.go.kr/aic/contestInfo.do#tab-track-nav3\n" +
  "Official FairData model submission guide\n" +
  "Project local evaluation artifacts";
for (const id of [
  "nt/weivbf", "nt/8w88jw", "nt/otc3q9", "nt/0c6qhp", "nt/s2txc0",
  "nt/8x8m73", "nt/kjojgi", "nt/tifcth", "nt/rsuc4y",
]) {
  presentation.resolve(id).setText(sourceBlock);
}

await fs.mkdir(renderDir, { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(
    `${renderDir}/${stem}.png`,
    await presentation.export({ slide, format: "png", scale: 1 }),
  );
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${renderDir}/${stem}.layout.json`, await layout.text());
}
await writeBlob(
  `${renderDir}/montage.webp`,
  await presentation.export({ format: "webp", montage: true, scale: 1 }),
);
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(output);
