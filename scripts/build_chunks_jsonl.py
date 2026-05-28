import json
from pathlib import Path


RAW_CHUNKS_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/chunks.jsonl")


def normalize_section(section: str | None) -> str:
    if not section:
        return "default"

    section = str(section).strip()

    mapping = {
        "주문": "order",
        "주 문": "order",
        "이유": "legal_reasoning",
        "이 유": "legal_reasoning",
        "별지": "penalty",
        "결론": "conclusion",
        "기타": "default",
    }

    return mapping.get(section, section)


def extract_text(item: dict) -> str:
    return (
        item.get("text")
        or item.get("chunk_text")
        or item.get("page_content")
        or item.get("content")
        or ""
    )


def build_chunks_jsonl():
    if not RAW_CHUNKS_DIR.exists():
        raise FileNotFoundError(f"원본 chunk 폴더가 없습니다: {RAW_CHUNKS_DIR}")

    target_files = sorted(RAW_CHUNKS_DIR.glob("*_hybrid.json"))

    if not target_files:
        raise FileNotFoundError(f"*_hybrid.json 파일이 없습니다: {RAW_CHUNKS_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    skipped_chunks = 0
    seen_chunk_ids = set()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for file_path in target_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"[SKIP FILE] list 형식이 아님: {file_path.name}")
                continue

            for item in data:
                metadata = item.get("metadata") or {}

                chunk_id = (
                    item.get("chunk_id")
                    or metadata.get("chunk_id")
                )

                text = extract_text(item).strip()

                if not chunk_id or not text:
                    skipped_chunks += 1
                    continue

                if chunk_id in seen_chunk_ids:
                    skipped_chunks += 1
                    continue

                seen_chunk_ids.add(chunk_id)

                if "-CH-" in chunk_id:
                    doc_id = chunk_id.split("-CH-")[0]
                else:
                    doc_id = (
                        item.get("doc_id")
                        or item.get("document_id")
                        or metadata.get("doc_id")
                        or metadata.get("document_id")
                    )

                raw_section = (
                    item.get("section_type")
                    or metadata.get("section")
                    or metadata.get("Header")
                )

                section_type = normalize_section(raw_section)

                output_item = {
                    "chunk_id": chunk_id,
                    "section_type": section_type,
                    "text": text,
                    "doc_id": doc_id,
                    "title": (
                        item.get("title")
                        or metadata.get("title")
                        or metadata.get("case_name")
                        or file_path.stem
                    ),
                    "source_file": file_path.name,
                    "page": (
                        item.get("page")
                        or metadata.get("page")
                        or metadata.get("page_number")
                    ),
                }

                out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                total_chunks += 1

    print("=== chunks.jsonl 생성 완료 ===")
    print(f"입력 파일 수: {len(target_files)}")
    print(f"저장 경로: {OUTPUT_PATH}")
    print(f"저장 chunk 수: {total_chunks}")
    print(f"스킵 chunk 수: {skipped_chunks}")
    print(f"중복 제거 후 chunk_id 수: {len(seen_chunk_ids)}")


if __name__ == "__main__":
    build_chunks_jsonl()
