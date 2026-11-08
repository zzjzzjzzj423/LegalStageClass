"""Local, content-addressed Jev decision cache (never a Git artifact)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .util import digest, write_json


CACHE_SCHEMA_VERSION = 1
PRE_NOUL_INSTRUCTIONS = "该用户咨询是否与《{canonical_law_name}》相关？"
POST_CHOICE_INSTRUCTIONS = "选择最能回答该用户咨询的法条。"


def _provider_identity() -> dict[str, str]:
    return {
        "provider": "typesafe_jev",
        "endpoint": os.environ.get("TYPESAFE_JEV_URL", "https://api.typesafe.ai/v1/systemone"),
        "model": os.environ.get("TYPESAFE_JEV_MODEL", "jev-latest"),
    }


def pre_route_cache_identity(query: str, laws: Sequence[dict[str, Any]], batch_size: int) -> tuple[str, dict[str, Any]]:
    """Bind a cached Noul score to every input that can affect it."""
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "jev_pre_noul_scores",
        **_provider_identity(),
        "query": query,
        "law_catalog_hash": digest(list(laws)),
        # A request contains one batch of questions; preserve its grouping as
        # part of the decision input instead of assuming batch invariance.
        "batch_size": batch_size,
        "question_type": "noul",
        "instructions": PRE_NOUL_INSTRUCTIONS,
        "criteria": {"true": "该法律可能包含适用法条", "false": "该法律不太可能适用"},
    }
    return digest(identity), identity


def post_rerank_cache_identity(query: str, candidates: Sequence[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    """Bind Choice probabilities to the exact, ordered shortlist text."""
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "jev_post_choice",
        **_provider_identity(),
        "query": query,
        "candidates": [
            {"article_id": candidate["article_id"], "name": candidate["name"], "content": candidate["content"]}
            for candidate in candidates
        ],
        "question_type": "choice",
        "instructions": POST_CHOICE_INSTRUCTIONS,
    }
    return digest(identity), identity


class JevDecisionCache:
    """Atomic JSON cache for complete PRE scores and POST probabilities."""

    def __init__(self, root: Path):
        self.root = root / "jev"

    def _path(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.json"

    def _load(self, cache_key: str, identity: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(cache_key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity") != identity or not isinstance(payload.get("value"), dict):
                return None
            return payload["value"]
        except (OSError, json.JSONDecodeError):
            return None

    def _store(self, cache_key: str, identity: dict[str, Any], value: dict[str, Any], label: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(cache_key)
        fd, temporary_raw = tempfile.mkstemp(prefix=f".{cache_key}.", suffix=".json", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_raw)
        try:
            write_json(temporary, {"identity": identity, "value": value})
            os.replace(temporary, destination)
            print(f"[cache] stored Jev {label}: {destination}", file=sys.stderr, flush=True)
        finally:
            temporary.unlink(missing_ok=True)

    def load_pre_scores(self, cache_key: str, identity: dict[str, Any], law_ids: Sequence[str]) -> dict[str, float] | None:
        value = self._load(cache_key, identity)
        scores = value.get("scores") if value else None
        if not isinstance(scores, dict) or set(scores) != set(law_ids):
            return None
        try:
            parsed = {law_id: float(score) for law_id, score in scores.items()}
        except (TypeError, ValueError):
            return None
        if any(not 0 <= score <= 1 for score in parsed.values()):
            return None
        print(f"[cache] Jev PRE scores hit: {self._path(cache_key)}", file=sys.stderr, flush=True)
        return parsed

    def store_pre_scores(self, cache_key: str, identity: dict[str, Any], scores: dict[str, float]) -> None:
        self._store(cache_key, identity, {"scores": scores}, "PRE scores")

    def load_post_rerank(self, cache_key: str, identity: dict[str, Any], article_ids: Sequence[str]) -> tuple[list[tuple[str, float]], dict[str, Any]] | None:
        value = self._load(cache_key, identity)
        ranking = value.get("ranking") if value else None
        metadata = value.get("metadata") if value else None
        if not isinstance(ranking, list) or not isinstance(metadata, dict):
            return None
        try:
            parsed = [(str(article_id), float(probability)) for article_id, probability in ranking]
        except (TypeError, ValueError):
            return None
        if len(parsed) != len(article_ids) or {article_id for article_id, _ in parsed} != set(article_ids):
            return None
        if any(not 0 <= probability <= 1 for _, probability in parsed):
            return None
        print(f"[cache] Jev POST decision hit: {self._path(cache_key)}", file=sys.stderr, flush=True)
        return parsed, metadata

    def store_post_rerank(self, cache_key: str, identity: dict[str, Any], ranking: Sequence[tuple[str, float]], metadata: dict[str, Any]) -> None:
        self._store(cache_key, identity, {"ranking": list(ranking), "metadata": metadata}, "POST decision")
