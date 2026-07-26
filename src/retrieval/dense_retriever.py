"""Lightweight offline dense retriever backed by Transformers and NumPy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.runtime import enable_vendored_dependencies

enable_vendored_dependencies()

try:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as error:  # pragma: no cover - optional runtime
    np = None
    torch = None
    AutoModel = None
    AutoTokenizer = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


def load_chunks(path: str | Path) -> List[Dict[str, Any]]:
    chunks = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                chunks.append(json.loads(line))
    if not chunks:
        raise ValueError(f"Chunk file is empty: {path}")
    return chunks


class DenseRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: Optional[str] = None,
        offline: bool = True,
        max_seq_length: int = 64,
        cpu_threads: int = 8,
    ):
        if IMPORT_ERROR:
            raise RuntimeError(
                "Dense retrieval dependencies are unavailable. "
                "Install numpy, torch, and transformers."
            ) from IMPORT_ERROR
        self.model_name = model_name
        self.device = torch.device(device or "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(cpu_threads)
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=offline
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=offline
        ).to(self.device)
        self.model.eval()
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings = None

    def _encode(self, texts: List[str], batch_size: int = 64):
        vectors = []
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[offset : offset + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).expand(output.size())
                summed = (output * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.concatenate(vectors, axis=0)

    def build_index(self, chunk_path: str, batch_size: int = 64) -> None:
        self.chunks = load_chunks(chunk_path)
        texts = []
        for chunk in self.chunks:
            title = str(chunk.get("title") or "")
            section = str(chunk.get("section") or chunk.get("section_type") or "")
            texts.append(f"사건명: {title}\n문서구역: {section}\n{chunk['text']}")
        self.embeddings = self._encode(texts, batch_size=batch_size)

    def save_index(
        self,
        index_path: str = "indexes/dense_embeddings.npy",
        metadata_path: str = "indexes/dense_chunks.jsonl",
    ) -> None:
        if self.embeddings is None:
            raise ValueError("Build the dense index before saving it.")
        index = Path(index_path)
        metadata = Path(metadata_path)
        index.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        np.save(index, self.embeddings)
        with metadata.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in self.chunks:
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def load_index(
        self,
        index_path: str = "indexes/dense_embeddings.npy",
        metadata_path: str = "indexes/dense_chunks.jsonl",
    ) -> None:
        self.embeddings = np.load(index_path, mmap_mode="r")
        self.chunks = load_chunks(metadata_path)
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("Dense embeddings and metadata have different lengths.")

    def search(
        self,
        query: str,
        top_n: int = 20,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.chunks:
            raise ValueError("Load or build the dense index before searching.")
        limit = top_k if top_k is not None else top_n
        query_vector = self._encode([query], batch_size=1)[0]
        scores = np.asarray(self.embeddings @ query_vector)
        if limit >= len(scores):
            indices = np.argsort(scores)[::-1]
        else:
            candidates = np.argpartition(scores, -limit)[-limit:]
            indices = candidates[np.argsort(scores[candidates])[::-1]]
        return [
            {
                **self.chunks[int(index)],
                "score": float(scores[int(index)]),
                "retriever": "dense",
                "chunk_text": self.chunks[int(index)]["text"],
                "preview": self.chunks[int(index)]["text"][:250],
            }
            for index in indices
        ]
