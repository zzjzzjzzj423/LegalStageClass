"""Thin live adapters for SiliconFlow embeddings and TypeSafe Jev."""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from .util import env_float


@dataclass
class ProviderLedger:
    entries: list[dict[str, Any]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def add(self, *, provider: str, operation: str, elapsed_ms: float, usage: dict[str, Any] | None, request_id: str | None, cost_per_1k: float, reserved_cost_usd: float | None = None) -> None:
        usage = usage or {}
        tokens = float(usage.get("total_tokens", usage.get("input_tokens", 0)) or 0)
        cost_known = cost_per_1k > 0 or reserved_cost_usd is not None
        with self._lock:
            self.entries.append({
                "event_id": str(uuid.uuid4()), "provider": provider, "operation": operation,
                "elapsed_ms": round(elapsed_ms, 3), "request_id": request_id,
                "usage": usage, "cost_usd": round(tokens / 1000 * cost_per_1k if cost_per_1k > 0 else reserved_cost_usd or 0.0, 8) if cost_known else None,
                "cost_status": "estimated_token_rate" if cost_per_1k > 0 else ("reserved" if cost_known else "not_reported"),
            })

    @property
    def known_cost(self) -> float:
        return sum(float(entry["cost_usd"] or 0.0) for entry in self.entries)


def _post_json(endpoint: str, api_key: str, payload: dict[str, Any], timeout_seconds: float = 90) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    tls12 = ssl.create_default_context()
    tls12.minimum_version = ssl.TLSVersion.TLSv1_2
    tls12.maximum_version = ssl.TLSVersion.TLSv1_2
    # Some Anaconda/OpenSSL combinations receive a spurious TLS 1.3 EOF from
    # this endpoint. Retry only that transport failure with verified TLS 1.2.
    for context in (None, tls12):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds, context=context) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body, (time.perf_counter() - start) * 1000
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"provider returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if context is None and isinstance(exc.reason, ssl.SSLEOFError):
                continue
            raise RuntimeError(f"provider request failed: {exc.reason}") from exc
    raise RuntimeError("provider request failed after verified TLS 1.2 retry")


class SiliconFlow:
    def __init__(self, ledger: ProviderLedger, *, batch_size: int = 32, concurrency: int | None = None):
        self.api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for Dense retrieval")
        self.endpoint = os.environ.get("SILICONFLOW_EMBEDDINGS_URL", "https://api.siliconflow.cn/v1/embeddings")
        self.model = os.environ.get("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.ledger, self.batch_size = ledger, batch_size
        self.concurrency = concurrency or int(os.environ.get("SILICONFLOW_EMBEDDING_CONCURRENCY", "4"))
        if self.concurrency < 1:
            raise ValueError("SILICONFLOW_EMBEDDING_CONCURRENCY must be at least 1")

    def _embed_batch(self, batch: list[str], operation: str) -> list[list[float]]:
        response, elapsed = _post_json(self.endpoint, self.api_key, {"model": self.model, "input": batch, "encoding_format": "float"})
        data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        if len(data) != len(batch) or any(not isinstance(item.get("embedding"), list) for item in data):
            raise RuntimeError("SiliconFlow response does not contain one float embedding per input")
        self.ledger.add(
            provider="siliconflow", operation=operation, elapsed_ms=elapsed,
            usage=response.get("usage"), request_id=response.get("id") or response.get("request_id"),
            cost_per_1k=env_float("SILICONFLOW_COST_USD_PER_1K_TOKENS"),
            reserved_cost_usd=env_float("SILICONFLOW_RESERVED_USD_PER_REQUEST", 0.0),
        )
        return [[float(value) for value in item["embedding"]] for item in data]

    def embed(self, texts: list[str], operation: str) -> list[list[float]]:
        batches = [(offset, texts[offset : offset + self.batch_size]) for offset in range(0, len(texts), self.batch_size)]
        vectors: list[list[float] | None] = [None] * len(texts)
        total_batches = max(1, (len(texts) + self.batch_size - 1) // self.batch_size)
        report_every = max(1, total_batches // 20)
        completed = 0
        with ThreadPoolExecutor(max_workers=min(self.concurrency, total_batches), thread_name_prefix="siliconflow") as executor:
            futures = {executor.submit(self._embed_batch, batch, operation): (offset, batch) for offset, batch in batches}
            for future in as_completed(futures):
                offset, batch = futures[future]
                vectors[offset : offset + len(batch)] = future.result()
                completed += 1
                if completed == 1 or completed == total_batches or completed % report_every == 0:
                    print(
                        f"[siliconflow] {operation}: {completed}/{total_batches} batches complete "
                        f"({completed / total_batches:.0%}); concurrency={self.concurrency}",
                        file=sys.stderr,
                        flush=True,
                    )
        if any(vector is None for vector in vectors):
            raise RuntimeError("SiliconFlow did not return every requested embedding")
        return [vector for vector in vectors if vector is not None]


class Jev:
    def __init__(self, ledger: ProviderLedger):
        self.api_key = os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for Jev experiments")
        self.endpoint = os.environ.get("TYPESAFE_JEV_URL", "https://api.typesafe.ai/v1/systemone")
        self.model = os.environ.get("TYPESAFE_JEV_MODEL", "jev-latest")
        self.ledger = ledger

    def _decide(self, state: dict[str, Any], questions: dict[str, Any], operation: str) -> dict[str, Any]:
        response, elapsed = _post_json(self.endpoint, self.api_key, {"model": self.model, "state": state, "questions": questions})
        self.ledger.add(
            provider="typesafe_jev", operation=operation, elapsed_ms=elapsed,
            usage=response.get("usage"), request_id=response.get("id") or response.get("request_id"),
            cost_per_1k=env_float("TYPESAFE_JEV_COST_USD_PER_1K_TOKENS"),
            reserved_cost_usd=env_float("TYPESAFE_JEV_RESERVED_USD_PER_JUDGMENT", 0.00003) * len(questions),
        )
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError("Jev response has no answers object")
        return answers

    def route_laws(self, query: str, laws: list[dict[str, Any]], batch_size: int = 32) -> dict[str, float]:
        scores: dict[str, float] = {}
        total_batches = max(1, (len(laws) + batch_size - 1) // batch_size)
        report_every = max(1, total_batches // 10)
        for offset in range(0, len(laws), batch_size):
            batch = laws[offset : offset + batch_size]
            questions = {
                law["law_id"]: {
                    "type": "noul",
                    "instructions": f"该用户咨询是否与《{law['canonical_law_name']}》相关？",
                    "criteria": {"true": "该法律可能包含适用法条", "false": "该法律不太可能适用"},
                }
                for law in batch
            }
            answers = self._decide({"query": query}, questions, "jev_pre_noul")
            for law in batch:
                answer = answers.get(law["law_id"], {})
                value = answer.get("noul") if isinstance(answer, dict) else None
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise RuntimeError(f"Jev Noul answer missing/invalid for {law['law_id']}")
                scores[law["law_id"]] = float(value)
            batch_number = offset // batch_size + 1
            if batch_number == 1 or batch_number == total_batches or batch_number % report_every == 0:
                print(
                    f"[typesafe_jev] jev_pre_noul: {min(offset + len(batch), len(laws))}/{len(laws)} laws "
                    f"({batch_number}/{total_batches} batches)",
                    file=sys.stderr,
                    flush=True,
                )
        return scores

    def rerank(self, query: str, candidates: list[dict[str, str]]) -> tuple[list[tuple[str, float]], dict[str, Any]]:
        criteria = {f"a_{article['article_id']}": f"{article['name']}\n{article['content']}" for article in candidates}
        answers = self._decide(
            {"query": query},
            {"rerank": {"type": "choice", "instructions": "选择最能回答该用户咨询的法条。", "criteria": criteria}},
            "jev_post_choice",
        )
        answer = answers.get("rerank", {})
        probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise RuntimeError("Jev Choice response must provide a probability for every a_<article_id> option")
        ranking = []
        for option_id, probability in probabilities.items():
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise RuntimeError(f"invalid Jev Choice probability for {option_id}")
            ranking.append((option_id.removeprefix("a_"), float(probability)))
        return sorted(ranking, key=lambda pair: (-pair[1], pair[0])), {"choice": answer.get("choice"), "confidence": answer.get("confidence")}
