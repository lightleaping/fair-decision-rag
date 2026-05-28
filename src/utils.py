import json
from pathlib import Path


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(file_path: str) -> list[dict]:
    items = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def save_jsonl(items: list[dict], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_valid_chunk_ids(chunks: list[dict]) -> set[str]:
    return {
        chunk["chunk_id"]
        for chunk in chunks
        if chunk.get("chunk_id")
    }


def validate_top5(candidates: list[dict], valid_chunk_ids: set[str]) -> list[dict]:
    """
    Top-5 chunk 반환 안정성 검증.
    - score 순 정렬
    - 중복 제거
    - 실제 존재하는 chunk_id만 유지
    - 최대 5개 반환
    """
    seen = set()
    top5 = []

    sorted_candidates = sorted(
        candidates,
        key=lambda item: item.get("score", 0),
        reverse=True
    )

    for candidate in sorted_candidates:
        chunk_id = candidate.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id in seen:
            continue

        if chunk_id not in valid_chunk_ids:
            continue

        top5.append(candidate)
        seen.add(chunk_id)

        if len(top5) == 5:
            break

    return top5