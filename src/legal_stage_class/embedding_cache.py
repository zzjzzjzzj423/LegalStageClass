"""Local, content-addressed corpus embedding cache (never a Git artifact)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from array import array
from pathlib import Path
from typing import Any, Sequence

from .util import digest, write_json

CACHE_SCHEMA_VERSION = 1


def corpus_cache_identity(
    dataset_hash: str, corpus: Sequence[dict[str, str]], *, endpoint: str, model: str, provider: str = "siliconflow",
) -> tuple[str, dict[str, Any]]:
    texts = [f"{article['name']}\n{article['content']}" for article in corpus]
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "corpus_text_hash": digest(texts),
        "provider": provider,
        "endpoint": endpoint,
        "model": model,
        "encoding_format": "float",
        "input_template": "{name}\\n{content}",
    }
    return digest(identity), identity


class EmbeddingCache:
    def __init__(self, root: Path):
        self.root = root

    def _paths(self, cache_key: str) -> tuple[Path, Path]:
        return self.root / f"{cache_key}.json", self.root / f"{cache_key}.f32"

    def load(self, cache_key: str, identity: dict[str, Any], expected_count: int) -> list[array] | None:
        metadata_path, vectors_path = self._paths(cache_key)
        if not metadata_path.is_file() or not vectors_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dimension = int(metadata["dimension"])
            if metadata.get("identity") != identity or metadata.get("count") != expected_count or dimension < 1:
                return None
            expected_bytes = expected_count * dimension * 4
            if vectors_path.stat().st_size != expected_bytes:
                return None
            values = array("f")
            with vectors_path.open("rb") as handle:
                values.fromfile(handle, expected_count * dimension)
            if len(values) != expected_count * dimension:
                return None
            print(f"[cache] corpus embeddings hit: {metadata_path}", file=sys.stderr, flush=True)
            return [values[offset : offset + dimension] for offset in range(0, len(values), dimension)]
        except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError):
            return None

    def store(self, cache_key: str, identity: dict[str, Any], vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            raise ValueError("cannot cache an empty corpus embedding set")
        dimension = len(vectors[0])
        if dimension < 1 or any(len(vector) != dimension for vector in vectors):
            raise ValueError("corpus embedding vectors must share a non-zero dimension")
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_path, vectors_path = self._paths(cache_key)
        vector_fd, vector_tmp = tempfile.mkstemp(prefix=f".{cache_key}.", suffix=".f32", dir=self.root)
        os.close(vector_fd)
        metadata_fd, metadata_tmp_raw = tempfile.mkstemp(prefix=f".{cache_key}.", suffix=".json", dir=self.root)
        os.close(metadata_fd)
        metadata_tmp = Path(metadata_tmp_raw)
        try:
            with Path(vector_tmp).open("wb") as handle:
                for vector in vectors:
                    array("f", vector).tofile(handle)
            write_json(metadata_tmp, {"identity": identity, "count": len(vectors), "dimension": dimension, "format": "float32"})
            os.replace(vector_tmp, vectors_path)
            os.replace(metadata_tmp, metadata_path)
            print(f"[cache] stored corpus embeddings: {metadata_path}", file=sys.stderr, flush=True)
        finally:
            Path(vector_tmp).unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)
