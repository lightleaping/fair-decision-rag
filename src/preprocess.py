import json
import re
from pathlib import Path

import pandas as pd


def load_chunks(file_path: str) -> list[dict]:
    """
    JSON, JSONL, CSV 파일 또는 폴더를 입력받아 chunk 데이터를 불러온다.

    현재 데이터 폴더 구조:
    - *_hybrid.json: 검색 대상 chunk 파일
    - *_metadata.json: 문서 메타데이터 파일, 검색에서는 제외
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {file_path}")

    # 폴더인 경우: 검색용 chunk 파일만 로딩
    if path.is_dir():
        all_chunks = []

        target_files = []
        target_files.extend(sorted(path.glob("*_hybrid.json")))
        target_files.extend(sorted(path.glob("*.jsonl")))
        target_files.extend(sorted(path.glob("*.csv")))

        if not target_files:
            raise FileNotFoundError(
                f"폴더 안에 검색용 chunk 파일(*_hybrid.json)이 없습니다: {file_path}"
            )

        print(f"총 {len(target_files)}개 검색용 데이터 파일을 로딩합니다.")

        for target_file in target_files:
            print(f"- 로딩 중: {target_file.name}")
            chunks = load_chunks(str(target_file))
            all_chunks.extend(chunks)

        print(f"전체 chunk 수: {len(all_chunks)}")
        return all_chunks

    # 단일 JSON 파일
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "chunks" in data:
                return data["chunks"]
            if "data" in data:
                return data["data"]

        # metadata.json 같은 파일은 여기로 들어오면 검색용이 아니므로 오류 처리
        raise ValueError(f"JSON 구조에서 chunk 리스트를 찾을 수 없습니다: {file_path}")

    # 단일 JSONL 파일
    if path.suffix == ".jsonl":
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    # 단일 CSV 파일
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict("records")

    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")


def clean_text(text: str) -> str:
    """
    법률 문서의 핵심 정보는 유지하면서 불필요한 공백만 정리한다.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_section_type(text: str, current_section: str | None = None) -> str:
    """
    section_type이 이미 있으면 사용하고,
    없으면 텍스트 패턴으로 추정한다.
    """
    if current_section and str(current_section).strip():
        return str(current_section).strip()

    text = text or ""

    if "주문" in text:
        return "주문"
    if "이유" in text:
        return "이유"
    if "별지" in text:
        return "별지"
    if "결론" in text:
        return "결론"
    if "과징금" in text:
        return "주문/별지"

    return "기타"


def normalize_chunks(chunks: list[dict]) -> list[dict]:
    """
    다양한 chunk 형식을 프로젝트 표준 형식으로 변환한다.

    지원 형식:
    1. {"chunk_id": "...", "chunk_text": "...", "section_type": "..."}
    2. {"page_content": "...", "metadata": {"chunk_id": "...", "section": "..."}}
    """
    normalized = []

    for idx, chunk in enumerate(chunks):
        metadata = chunk.get("metadata") or {}

        raw_text = (
            chunk.get("chunk_text")
            or chunk.get("text")
            or chunk.get("content")
            or chunk.get("page_content")
            or ""
        )

        cleaned_text = clean_text(raw_text)

        chunk_id = (
            chunk.get("chunk_id")
            or metadata.get("chunk_id")
            or f"chunk_{idx:05d}"
        )

        section_type = (
            chunk.get("section_type")
            or metadata.get("section")
            or metadata.get("Header")
        )

        section_type = detect_section_type(
            text=cleaned_text,
            current_section=section_type
        )

        if isinstance(chunk_id, str) and "-CH-" in chunk_id:
            document_id = chunk_id.split("-CH-")[0]
        else:
            document_id = (
                chunk.get("document_id")
                or metadata.get("document_id")
                or metadata.get("source")
                or f"document_{idx:05d}"
            )

        page = (
            chunk.get("page")
            or metadata.get("page")
            or metadata.get("page_number")
            or None
        )

        normalized.append({
            **chunk,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_text": cleaned_text,
            "page": page,
            "section_type": section_type,
            "metadata": metadata,
        })

    return normalized


def check_chunk_integrity(chunks: list[dict]) -> dict:
    """
    chunk_id 누락, 중복, 빈 텍스트를 검사한다.
    """
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]

    missing_chunk_id = sum(1 for chunk_id in chunk_ids if not chunk_id)
    duplicated_chunk_id = len(chunk_ids) - len(set(chunk_ids))

    empty_text = sum(
        1 for chunk in chunks
        if not str(chunk.get("chunk_text", "")).strip()
    )

    return {
        "total_chunks": len(chunks),
        "missing_chunk_id": missing_chunk_id,
        "duplicated_chunk_id": duplicated_chunk_id,
        "empty_text": empty_text,
    }


def get_section_statistics(chunks: list[dict]) -> dict:
    """
    section_type별 chunk 개수를 계산한다.
    """
    stats = {}

    for chunk in chunks:
        section_type = chunk.get("section_type", "기타")
        stats[section_type] = stats.get(section_type, 0) + 1

    return stats