"""Content-addressed cache for remote cross-encoder reranker responses."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .util import digest, write_json


CACHE_SCHEMA_VERSION = 1


def rerank_cache_identity(
    query: str,
    candidates: Sequence[dict[str, str]],
    *,
    endpoint: str,
    model: str,
    instruction: str,
) -> tuple[str, dict[str, Any]]:
    """Bind a reranker result to the exact provider and ordered candidate text."""
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "siliconflow_cross_encoder_rerank",
        "provider": "siliconflow",
        "endpoint": endpoint,
        "model": model,
        "query": query,
        "instruction": instruction,
        "candidates": [
            {"article_id": candidate["article_id"], "name": candidate["name"], "content": candidate["content"]}
            for candidate in candidates
        ],
    }
    return digest(identity), identity


class RerankDecisionCache:
    """Atomic JSON cache for complete cross-encoder rankings."""

    def __init__(self, root: Path):
        self.root = root / "rerank"

    def _path(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.json"

    def load(
        self,
        cache_key: str,
        identity: dict[str, Any],
        article_ids: Sequence[str],
    ) -> tuple[list[tuple[str, float]], dict[str, Any]] | None:
        path = self._path(cache_key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("identity") != identity or not isinstance(payload.get("value"), dict):
            return None
        value = payload["value"]
        ranking, metadata = value.get("ranking"), value.get("metadata")
        if not isinstance(ranking, list) or not isinstance(metadata, dict):
            return None
        try:
            parsed = [(str(article_id), float(score)) for article_id, score in ranking]
        except (TypeError, ValueError):
            return None
        if len(parsed) != len(article_ids) or {article_id for article_id, _ in parsed} != set(article_ids):
            return None
        if not all(score == score and abs(score) != float("inf") for _, score in parsed):
            return None
        print(f"[cache] cross-encoder rerank hit: {path}", file=sys.stderr, flush=True)
        return parsed, metadata

    def store(
        self,
        cache_key: str,
        identity: dict[str, Any],
        ranking: Sequence[tuple[str, float]],
        metadata: dict[str, Any],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(cache_key)
        fd, temporary_raw = tempfile.mkstemp(prefix=f".{cache_key}.", suffix=".json", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_raw)
        try:
            write_json(temporary, {"identity": identity, "value": {"ranking": list(ranking), "metadata": metadata}})
            os.replace(temporary, destination)
            print(f"[cache] stored cross-encoder rerank: {destination}", file=sys.stderr, flush=True)
        finally:
            temporary.unlink(missing_ok=True)
