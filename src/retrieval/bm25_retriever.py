"""Dependency-free Korean BM25 retriever."""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


TOKEN_PATTERN = re.compile(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?")


def simple_tokenize(text: str) -> List[str]:
    """Tokenize Korean text with word and overlapping bi-gram features."""
    words = TOKEN_PATTERN.findall(str(text).lower())
    features = list(words)
    for word in words:
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 2:
            features.extend(word[i : i + 2] for i in range(len(word) - 1))
    return features


class BM25Retriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not chunks:
            raise ValueError("청크가 비어 있습니다.")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies: List[Counter[str]] = []
        self.document_lengths: List[int] = []
        document_frequency: Counter[str] = Counter()
        self.postings: Dict[str, List[int]] = defaultdict(list)

        for index, chunk in enumerate(chunks):
            title = str(chunk.get("title") or "")
            section = str(chunk.get("section") or chunk.get("section_type") or "")
            # Metadata is a strong structural signal in long legal documents.
            # Repetition provides explicit field weighting without changing BM25.
            searchable_text = " ".join([title] * 2 + [section] * 8 + [chunk["text"]])
            tokens = simple_tokenize(searchable_text)
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))
            for token in frequencies:
                document_frequency[token] += 1
                self.postings[token].append(index)

        corpus_size = len(chunks)
        self.average_length = sum(self.document_lengths) / corpus_size
        self.idf = {
            token: math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    @classmethod
    def from_jsonl(cls, jsonl_path: str) -> "BM25Retriever":
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"청크 파일이 없습니다: {path}")
        chunks = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("chunk_id") or not item.get("text"):
                    raise ValueError(f"{line_number}번째 청크의 필수 필드가 없습니다.")
                chunks.append(item)
        return cls(chunks)

    def save(self, path: str | Path) -> None:
        """Persist the tokenized index for fast server startup."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        """Load a trusted index produced by :meth:`save`."""
        with Path(path).open("rb") as stream:
            retriever = pickle.load(stream)
        if not isinstance(retriever, cls):
            raise TypeError("The persisted object is not a BM25Retriever.")
        return retriever

    def search(
        self,
        query: str,
        top_n: int = 20,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        limit = top_k if top_k is not None else top_n
        query_terms = Counter(simple_tokenize(query))
        if not query_terms:
            raise ValueError("검색어가 비어 있습니다.")
        scores: Dict[int, float] = defaultdict(float)
        for term, query_frequency in query_terms.items():
            for index in self.postings.get(term, []):
                term_frequency = self.term_frequencies[index][term]
                length_norm = 1 - self.b + self.b * (
                    self.document_lengths[index] / self.average_length
                )
                scores[index] += (
                    self.idf[term]
                    * term_frequency
                    * (self.k1 + 1)
                    / (term_frequency + self.k1 * length_norm)
                    * query_frequency
                )
        ranked = sorted(scores, key=scores.get, reverse=True)[:limit]
        results = []
        for index in ranked:
            chunk = self.chunks[index]
            results.append(
                {
                    **chunk,
                    "score": float(scores[index]),
                    "retriever": "bm25",
                    "chunk_text": chunk["text"],
                    "preview": chunk["text"][:250],
                }
            )
        return results
