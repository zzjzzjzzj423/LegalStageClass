"""Thin live adapters for SiliconFlow embeddings and TypeSafe Jev."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from .util import env_float


@dataclass
class ProviderLedger:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, provider: str, operation: str, elapsed_ms: float, usage: dict[str, Any] | None, request_id: str | None, cost_per_1k: float, reserved_cost_usd: float | None = None) -> None:
        usage = usage or {}
        tokens = float(usage.get("total_tokens", usage.get("input_tokens", 0)) or 0)
        cost_known = cost_per_1k > 0 or reserved_cost_usd is not None
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
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"provider request failed: {exc.reason}") from exc
    return body, (time.perf_counter() - start) * 1000


class SiliconFlow:
    def __init__(self, ledger: ProviderLedger, *, batch_size: int = 32):
        self.api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for Dense retrieval")
        self.endpoint = os.environ.get("SILICONFLOW_EMBEDDINGS_URL", "https://api.siliconflow.cn/v1/embeddings")
        self.model = os.environ.get("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.ledger, self.batch_size = ledger, batch_size

    def embed(self, texts: list[str], operation: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            response, elapsed = _post_json(self.endpoint, self.api_key, {"model": self.model, "input": batch, "encoding_format": "float"})
            data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
            if len(data) != len(batch) or any(not isinstance(item.get("embedding"), list) for item in data):
                raise RuntimeError("SiliconFlow response does not contain one float embedding per input")
            vectors.extend([[float(value) for value in item["embedding"]] for item in data])
            self.ledger.add(
                provider="siliconflow", operation=operation, elapsed_ms=elapsed,
                usage=response.get("usage"), request_id=response.get("id") or response.get("request_id"),
                cost_per_1k=env_float("SILICONFLOW_COST_USD_PER_1K_TOKENS"),
                reserved_cost_usd=env_float("SILICONFLOW_RESERVED_USD_PER_REQUEST", 0.0),
            )
        return vectors


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
