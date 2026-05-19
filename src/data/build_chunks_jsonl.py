# src/data/build_chunks_jsonl.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    """
    JSON 파일을 로드한다.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_section_type(item: Dict[str, Any]) -> str:
    """
    chunk에 section_type이 없을 때 텍스트와 기존 필드를 기반으로 대략 추정한다.

    단, 이 값은 검색 boost용 보조 정보일 뿐이고,
    chunk_id는 절대 새로 만들지 않는다.
    """

    for key in ["section_type", "section", "type"]:
        value = item.get(key)
        if value:
            return normalize_section_type(str(value))

    text = str(item.get("text", ""))

    if any(word in text for word in ["과징금", "부과금", "산정", "감경", "가중"]):
        return "penalty"

    if any(word in text for word in ["시정명령", "재발방지", "공표명령", "명령한다"]):
        return "order"

    if any(word in text for word in ["제", "조", "법", "시행령", "고시"]):
        if any(law in text for law in ["공정거래법", "하도급법", "가맹사업법", "표시광고법"]):
            return "law_article"

    if any(word in text for word in ["판단", "인정된다", "해당한다", "위반", "부당"]):
        return "legal_reasoning"

    if any(word in text for word in ["피심인", "거래", "행위", "요구", "제한"]):
        return "fact"

    return "default"


def normalize_section_type(section_type: str) -> str:
    """
    다양한 section 이름을 프로젝트 표준 section_type으로 정규화한다.
    """

    value = section_type.strip().lower()

    mapping = {
        "사실관계": "fact",
        "사실": "fact",
        "fact": "fact",

        "판단": "legal_reasoning",
        "판단근거": "legal_reasoning",
        "법적판단": "legal_reasoning",
        "legal_reasoning": "legal_reasoning",

        "법조항": "law_article",
        "적용법조": "law_article",
        "관련법령": "law_article",
        "law_article": "law_article",

        "과징금": "penalty",
        "제재": "penalty",
        "penalty": "penalty",

        "시정명령": "order",
        "조치": "order",
        "order": "order",

        "결론": "conclusion",
        "의결": "conclusion",
        "conclusion": "conclusion",

        "요약": "summary",
        "summary": "summary",
    }

    return mapping.get(value, value if value else "default")


def extract_chunks_from_hybrid_json(
    hybrid_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    *_hybrid.json 파일에서 chunk 목록을 추출한다.

    가능한 구조를 최대한 넓게 허용한다.
    - {"chunks": [...]}
    - {"documents": [{"chunks": [...]}]}
    - [{...}, {...}]
    """

    data = load_json(hybrid_path)
    metadata = metadata or {}

    if isinstance(data, list):
        raw_chunks = data
    elif isinstance(data, dict):
        if isinstance(data.get("chunks"), list):
            raw_chunks = data["chunks"]
        elif isinstance(data.get("documents"), list):
            raw_chunks = []
            for doc in data["documents"]:
                if isinstance(doc, dict) and isinstance(doc.get("chunks"), list):
                    raw_chunks.extend(doc["chunks"])
        else:
            raise ValueError(
                f"chunk 목록을 찾을 수 없습니다: {hybrid_path}"
            )
    else:
        raise ValueError(f"지원하지 않는 JSON 구조입니다: {hybrid_path}")

    converted = []

    title = (
        metadata.get("title")
        or metadata.get("case_name")
        or metadata.get("document_title")
        or hybrid_path.stem.replace("_hybrid", "")
    )

    doc_id = (
        metadata.get("doc_id")
        or metadata.get("case_id")
        or hybrid_path.stem.replace("_hybrid", "")
    )

    for idx, item in enumerate(raw_chunks, start=1):
        if not isinstance(item, dict):
            continue

        chunk_id = (
                item.get("chunk_id")
                or item.get("id")
                or item.get("chunkId")
                or item.get("chunkID")
    )
        if not chunk_id and isinstance(item.get("metadata"), dict):
            chunk_id = (
                item["metadata"].get("chunk_id")
                or item["metadata"].get("id")
                or item["metadata"].get("chunkId")
                or item["metadata"].get("chunkID")
            )

        if not chunk_id:
            print(
                f"[SKIP] {hybrid_path.name}의 {idx}번째 chunk에 chunk_id가 없어 건너뜁니다. "
                "기존 chunk_id를 확인해야 합니다."
            )
            continue

        text = (
            item.get("text")
            or item.get("content")
            or item.get("chunk_text")
            or item.get("page_content")
            or item.get("body")
            or ""
        )

        if not text and isinstance(item.get("metadata"), dict):
            text = (
                item["metadata"].get("text")
                or item["metadata"].get("content")
                or item["metadata"].get("chunk_text")
                or ""
            )

        if not text:
            continue

        section_type = infer_section_type(item)

        converted.append(
            {
                "chunk_id": chunk_id,
                "doc_id": item.get("doc_id") or doc_id,
                "title": item.get("title") or title,
                "section_type": section_type,
                "text": text,
                "source_file": hybrid_path.name,
                "page": item.get("page"),
            }
        )

    return converted


def find_matching_metadata(hybrid_path: Path) -> Optional[Path]:
    """
    *_hybrid.json에 대응하는 *_metadata.json 파일을 찾는다.
    """

    candidates = [
        hybrid_path.with_name(hybrid_path.name.replace("_hybrid.json", "_metadata.json")),
        hybrid_path.with_name(hybrid_path.name.replace("hybrid.json", "metadata.json")),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def build_chunks_jsonl(
    input_dir: str = "data/raw",
    output_path: str = "data/chunks.jsonl",
) -> Dict[str, Any]:
    """
    input_dir 아래의 *_hybrid.json 파일들을 모아 data/chunks.jsonl을 생성한다.
    """

    input_path = Path(input_dir)
    output = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")

    hybrid_files = sorted(input_path.glob("*_hybrid.json"))

    if not hybrid_files:
        raise FileNotFoundError(
            f"{input_dir} 안에서 *_hybrid.json 파일을 찾지 못했습니다."
        )

    all_chunks = []
    seen_chunk_ids = set()
    duplicate_chunk_ids = []

    for hybrid_file in hybrid_files:
        metadata_path = find_matching_metadata(hybrid_file)

        metadata = {}
        if metadata_path and metadata_path.exists():
            metadata = load_json(metadata_path)

        chunks = extract_chunks_from_hybrid_json(
            hybrid_path=hybrid_file,
            metadata=metadata,
        )

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            if chunk_id in seen_chunk_ids:
                duplicate_chunk_ids.append(chunk_id)
                continue

            seen_chunk_ids.add(chunk_id)
            all_chunks.append(chunk)

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    summary = {
        "input_dir": input_dir,
        "output_path": output_path,
        "hybrid_file_count": len(hybrid_files),
        "chunk_count": len(all_chunks),
        "duplicate_chunk_id_count": len(duplicate_chunk_ids),
        "duplicate_chunk_ids_sample": duplicate_chunk_ids[:20],
    }

    return summary


if __name__ == "__main__":
    summary = build_chunks_jsonl(
        input_dir="data/raw",
        output_path="data/chunks.jsonl",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))