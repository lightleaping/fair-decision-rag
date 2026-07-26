"""Build the canonical chunk corpus from the contest archive.

The official ``*_hybrid.json`` files already contain the chunk identifiers used
by the evaluator.  This script preserves them verbatim and can stream directly
from the ZIP, so the 500 PDFs do not need to be extracted.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Tuple


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT = Path("data/chunks.jsonl")


def normalize_section(section: object) -> str:
    value = str(section or "").strip()
    compact = value.replace(" ", "")
    if not compact:
        return "default"
    if "주문" in compact:
        return "order"
    if "이유" in compact or "판단" in compact:
        return "legal_reasoning"
    if "과징금" in compact or "과태료" in compact:
        return "penalty"
    if "법령" in compact or "법조" in compact:
        return "law_article"
    if "사실" in compact or "행위" in compact:
        return "fact"
    if "결론" in compact:
        return "conclusion"
    return value


def iter_sources(raw_dir: Path) -> Iterator[Tuple[str, bytes]]:
    archives = sorted(raw_dir.glob("*.zip"))
    if archives:
        for archive in archives:
            with zipfile.ZipFile(archive) as bundle:
                for name in sorted(bundle.namelist()):
                    if name.endswith("_hybrid.json"):
                        yield name, bundle.read(name)
        return

    for path in sorted(raw_dir.glob("*_hybrid.json")):
        yield path.name, path.read_bytes()


def iter_chunks(raw_dir: Path) -> Iterable[dict]:
    seen: set[str] = set()
    for source_name, payload in iter_sources(raw_dir):
        data = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(data, list):
            continue
        for item in data:
            metadata = item.get("metadata") or {}
            chunk_id = item.get("chunk_id") or metadata.get("chunk_id")
            text = (
                item.get("text")
                or item.get("chunk_text")
                or item.get("page_content")
                or item.get("content")
                or ""
            )
            if not chunk_id or not str(text).strip() or chunk_id in seen:
                continue
            seen.add(str(chunk_id))
            doc_id = str(chunk_id).split("-CH-", 1)[0]
            raw_section = (
                item.get("section_type")
                or metadata.get("section")
                or metadata.get("Header")
            )
            yield {
                "chunk_id": str(chunk_id),
                "doc_id": doc_id,
                "section_type": normalize_section(raw_section),
                "section": raw_section or "default",
                "text": str(text).strip(),
                "title": metadata.get("title")
                or metadata.get("case_name")
                or source_name.removesuffix("_hybrid.json"),
                "source_file": source_name,
                "page": item.get("page")
                or metadata.get("page")
                or metadata.get("page_number"),
            }


def build_chunks_jsonl(raw_dir: Path, output: Path) -> dict:
    if not raw_dir.exists():
        raise FileNotFoundError(f"원본 데이터 폴더가 없습니다: {raw_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    document_ids: set[str] = set()
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in iter_chunks(raw_dir):
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
            document_ids.add(chunk["doc_id"])
    if not count:
        raise ValueError(f"공식 청크를 찾지 못했습니다: {raw_dir}")
    return {"documents": len(document_ids), "chunks": count, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_chunks_jsonl(args.raw_dir, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
