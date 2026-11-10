"""Thin live adapters for SiliconFlow embeddings and TypeSafe Jev."""

from __future__ import annotations

import json
import http.client
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
from typing import Any, Callable

from .jev_budget import REQUEST_TOKEN_LIMIT, JevBudget, cost_usd
from .jev_cache import FILE_CHOICE_INSTRUCTIONS, POST_CHOICE_INSTRUCTIONS
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


def _post_json(endpoint: str, api_key: str, payload: dict[str, Any], timeout_seconds: float = 90,
               *, before_attempt: Callable[[], None] | None = None) -> tuple[dict[str, Any], float]:
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
    last_error: BaseException | None = None
    for context in (None, tls12):
        for attempt in range(3):
            if before_attempt is not None:
                before_attempt()
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds, context=context) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body, (time.perf_counter() - start) * 1000
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"provider returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if context is None and isinstance(exc.reason, ssl.SSLEOFError):
                    break
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"provider request failed: {exc.reason}") from exc
            except (http.client.RemoteDisconnected, ConnectionResetError, TimeoutError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
    raise RuntimeError("provider closed the connection without a response after retries; check the provider endpoint, key, and network, then retry") from last_error


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


class SiliconFlowReranker:
    """SiliconFlow text reranking adapter for an independent cross-encoder arm."""

    def __init__(self, ledger: ProviderLedger):
        self.api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for cross-encoder reranking")
        self.endpoint = os.environ.get("SILICONFLOW_RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
        self.model = os.environ.get("SILICONFLOW_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
        self.instruction = os.environ.get(
            "SILICONFLOW_RERANK_INSTRUCTION",
            "请根据用户问题判断每条法律条文的相关性，并按相关性从高到低排序。",
        )
        self.ledger = ledger

    def rerank(self, query: str, candidates: list[dict[str, str]]) -> tuple[list[tuple[str, float]], dict[str, Any]]:
        if not candidates:
            raise ValueError("cross-encoder reranking requires at least one candidate")
        payload = {
            "model": self.model,
            "query": query,
            "documents": [f"{candidate['name']}\n{candidate['content']}" for candidate in candidates],
            "instruction": self.instruction,
            "top_n": len(candidates),
            "return_documents": False,
        }
        response, elapsed = _post_json(self.endpoint, self.api_key, payload)
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(candidates):
            raise RuntimeError("SiliconFlow rerank response must contain one result per candidate")
        ranking: list[tuple[str, float]] = []
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict) or type(result.get("index")) is not int:
                raise RuntimeError("SiliconFlow rerank response has an invalid result index")
            index = result["index"]
            score = result.get("relevance_score")
            if index < 0 or index >= len(candidates) or index in seen or not isinstance(score, (int, float)):
                raise RuntimeError("SiliconFlow rerank response has invalid indices or scores")
            score = float(score)
            if score != score or abs(score) == float("inf"):
                raise RuntimeError("SiliconFlow rerank response has a non-finite score")
            seen.add(index)
            ranking.append((candidates[index]["article_id"], score))
        self.ledger.add(
            provider="siliconflow", operation="cross_encoder_rerank", elapsed_ms=elapsed,
            usage=response.get("meta", {}).get("tokens") if isinstance(response.get("meta"), dict) else response.get("usage"),
            request_id=response.get("id") or response.get("request_id"),
            cost_per_1k=env_float("SILICONFLOW_RERANK_COST_USD_PER_1K_TOKENS"),
            reserved_cost_usd=env_float("SILICONFLOW_RERANK_RESERVED_USD_PER_REQUEST", 0.0),
        )
        return ranking, {
            "model": response.get("model") or self.model,
            "endpoint": self.endpoint,
            "instruction": self.instruction,
            "result_count": len(ranking),
        }


class Jev:
    def __init__(self, ledger: ProviderLedger, *, run_id: str, query_id: str):
        self.api_key = os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for Jev experiments")
        self.endpoint = os.environ.get("TYPESAFE_JEV_URL", "https://api.typesafe.ai/v1/systemone")
        self.model = os.environ.get("TYPESAFE_JEV_MODEL", "jev-latest")
        self.ledger = ledger
        self.run_id = run_id
        self.query_id = query_id
        self.budget = JevBudget()

    def _decide(self, state: dict[str, Any], questions: dict[str, Any], operation: str) -> dict[str, Any]:
        attempt_id = None

        def reserve_attempt() -> None:
            nonlocal attempt_id
            attempt_id = self.budget.reserve(self.run_id, self.query_id, operation, self.model)

        response, elapsed = _post_json(
            self.endpoint, self.api_key, {"model": self.model, "state": state, "questions": questions},
            before_attempt=reserve_attempt,
        )
        usage = response.get("usage")
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        if type(input_tokens) is not int or input_tokens < 0:
            input_tokens = None
        request_id = response.get("id") or response.get("request_id")
        model = response.get("model") or self.model
        confirmed, accounted = self.budget.settle(attempt_id, input_tokens, request_id, model)
        status = "estimated_input_token_rate" if input_tokens is not None else "reserved"
        cost = cost_usd(input_tokens if input_tokens is not None else REQUEST_TOKEN_LIMIT)
        self.ledger.entries.append({
            "event_id": str(uuid.uuid4()), "provider": "typesafe_jev", "operation": operation,
            "elapsed_ms": round(elapsed, 3), "request_id": request_id, "usage": usage,
            "query_id": self.query_id, "model": model, "cost_usd": cost, "cost_status": status,
            "jev_confirmed_cost_usd": confirmed, "jev_accounted_cost_usd": accounted,
        })
        print(f"[jev] query={self.query_id} {operation}: USD {cost:.8f} ({status}), "
              f"confirmed USD {confirmed:.6f}, including reservations USD {accounted:.6f} / 24",
              file=sys.stderr, flush=True)
        if input_tokens is not None and input_tokens > REQUEST_TOKEN_LIMIT:
            raise RuntimeError("Jev usage exceeded the budgeted request context; verify the provider context limit")
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError("Jev response has no answers object")
        return answers

    @staticmethod
    def _choice_ranking(answer: Any, criteria: dict[str, str], prefix: str) -> list[tuple[str, float]]:
        probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise RuntimeError(f"Jev Choice response must provide a probability for every {prefix}<id> option")
        ranking: list[tuple[str, float]] = []
        for option_id, probability in probabilities.items():
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise RuntimeError(f"invalid Jev Choice probability for {option_id}")
            ranking.append((option_id.removeprefix(prefix), float(probability)))
        return sorted(ranking, key=lambda pair: (-pair[1], pair[0]))

    def choose_files(self, query: str, laws: list[dict[str, Any]]) -> tuple[list[tuple[str, float]], dict[str, Any]]:
        if not laws or len(laws) > 255:
            raise ValueError("Jev file Choice requires 1 to 255 files")
        criteria = {f"f_{law['law_id']}": f"{law['law_id']}\n{law['canonical_law_name']}" for law in laws}
        answers = self._decide(
            {"query": query},
            {"file_choice": {"type": "choice", "instructions": FILE_CHOICE_INSTRUCTIONS, "criteria": criteria}},
            "jev_file_choice",
        )
        answer = answers.get("file_choice", {})
        return self._choice_ranking(answer, criteria, "f_"), {"choice": answer.get("choice"), "confidence": answer.get("confidence")}

    def rerank(self, query: str, candidates: list[dict[str, str]]) -> tuple[list[tuple[str, float]], dict[str, Any]]:
        criteria = {f"a_{article['article_id']}": f"{article['name']}\n{article['content']}" for article in candidates}
        answers = self._decide(
            {"query": query},
            {"rerank": {"type": "choice", "instructions": POST_CHOICE_INSTRUCTIONS, "criteria": criteria}},
            "jev_post_choice",
        )
        answer = answers.get("rerank", {})
        return self._choice_ranking(answer, criteria, "a_"), {"choice": answer.get("choice"), "confidence": answer.get("confidence")}
