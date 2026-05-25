# src/retrieval/day10_bm25_pipeline.py

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.day10_a_runner import run_day10_a, save_result_json


def load_valid_chunk_ids(jsonl_path: str) -> Set[str]:
    """
    怨듦컻蹂??곗씠?곗뿉 議댁옱?섎뒗 chunk_id 吏묓빀??濡쒕뱶?쒕떎.
    Top-5 寃곌낵媛 ?ㅼ젣 chunk_id?몄? 寃利앺븯湲??꾪빐 ?ъ슜?쒕떎.
    """

    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"chunk ?뚯씪??李얠쓣 ???놁뒿?덈떎: {jsonl_path}")

    valid_chunk_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            chunk_id = item.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"{line_idx}踰덉㎏ 以꾩뿉 chunk_id媛 ?놁뒿?덈떎.")

            valid_chunk_ids.add(chunk_id)

    return valid_chunk_ids


def run_bm25_day10_pipeline(
    query: str,
    chunk_jsonl_path: str,
    output_path: str = "outputs/results/day10_bm25_top5.json",
    bm25_top_n: int = 20,
    max_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Day 10 BM25 + A 紐⑤뱢 ?듯빀 ?뚯씠?꾨씪??

    ?낅젰:
    - query: ?ъ슜??吏덈Ц
    - chunk_jsonl_path: chunk JSONL ?뚯씪 寃쎈줈
    - output_path: 寃곌낵 ???寃쎈줈
    - bm25_top_n: BM25 ?꾨낫 媛쒖닔
    - max_seconds: ?쒗븳 ?쒓컙

    異쒕젰:
    - query
    - query_type
    - top5_chunk_ids
    - top5_results
    - elapsed_seconds
    """

    start_time = time.perf_counter()

    # 1. BM25 ?몃뜳???앹꽦
    retriever = BM25Retriever.from_jsonl(chunk_jsonl_path)

    # 2. ?좏슚 chunk_id 濡쒕뱶
    valid_chunk_ids = load_valid_chunk_ids(chunk_jsonl_path)

    # 3. BM25 寃??
    bm25_results = retriever.search(
        query=query,
        top_n=bm25_top_n,
    )

    # 4. A 紐⑤뱢 ?곸슜
    final_result = run_day10_a(
        query=query,
        search_results=bm25_results,
        valid_chunk_ids=valid_chunk_ids,
    )

    elapsed_seconds = time.perf_counter() - start_time

    final_result["retriever"] = "bm25"
    final_result["bm25_top_n"] = bm25_top_n
    final_result["elapsed_seconds"] = round(elapsed_seconds, 4)
    final_result["within_30_seconds"] = elapsed_seconds <= max_seconds

    if elapsed_seconds > max_seconds:
        raise TimeoutError(
            f"?묐떟 ?쒓컙??{elapsed_seconds:.2f}珥덈줈 "
            f"{max_seconds:.2f}珥??쒗븳??珥덇낵?덉뒿?덈떎."
        )

    save_result_json(
        result=final_result,
        output_path=output_path,
    )

    return final_result


def make_sample_chunk_jsonl(path: str) -> None:
    """
    ?ㅼ젣 data/chunks.jsonl???놁쓣 ???뚯뒪?몄슜 sample chunk ?뚯씪???앹꽦?쒕떎.
    """

    sample_chunks = [
        {
            "chunk_id": "case_001_chunk_001",
            "section_type": "fact",
            "text": "?쇱떖?몄? 嫄곕옒?곷?諛⑹뿉寃??뱀젙 議곌굔???붽뎄?섏???",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_002",
            "section_type": "legal_reasoning",
            "text": "???됱쐞??嫄곕옒??吏?꾨? ?댁슜??遺덉씠???쒓났?됱쐞濡??먮떒?쒕떎.",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_003",
            "section_type": "penalty",
            "text": "?쇱떖?몄뿉寃?怨쇱쭠湲?1???먯쓣 遺怨쇳븳??",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_004",
            "section_type": "law_article",
            "text": "???ш굔?먮뒗 怨듭젙嫄곕옒踰???3議?????씠 ?곸슜?쒕떎.",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_005",
            "section_type": "order",
            "text": "?쇱떖?몄? ?ν썑 ?숈씪???됱쐞瑜?諛섎났?섏뿬?쒕뒗 ?꾨땲 ?쒕떎.",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_006",
            "section_type": "summary",
            "text": "???ш굔? 嫄곕옒??吏???⑥슜?됱쐞??愿??嫄댁씠??",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
        {
            "chunk_id": "case_001_chunk_007",
            "section_type": "conclusion",
            "text": "?꾩썝?뚮뒗 ?쇱떖?몄쓽 ?됱쐞??????쒖젙紐낅졊怨?怨쇱쭠湲?遺怨쇰? 寃곗젙?섏???",
            "doc_id": "case_001",
            "title": "?섑뵆 ?ш굔"
        },
    ]

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for item in sample_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    chunk_jsonl_path = "data/chunks.jsonl"

    if not Path(chunk_jsonl_path).exists():
        print("data/chunks.jsonl???놁뼱 sample ?뚯씪???앹꽦?⑸땲??")
        make_sample_chunk_jsonl(chunk_jsonl_path)

    query = "???ш굔 怨쇱쭠湲덉? ?쇰쭏??"

    result = run_bm25_day10_pipeline(
        query=query,
        chunk_jsonl_path=chunk_jsonl_path,
        output_path="outputs/results/day10_bm25_top5.json",
        bm25_top_n=20,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n????꾨즺: outputs/results/day10_bm25_top5.json")
