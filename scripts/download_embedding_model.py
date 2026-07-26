"""Download and save the embedding model during the Docker build."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="models/embedding")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    tokenizer.save_pretrained(output)
    model.save_pretrained(output, safe_serialization=True)
    print(f"Saved {args.model} to {output}")


if __name__ == "__main__":
    main()
