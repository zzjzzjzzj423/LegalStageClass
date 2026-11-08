"""Three independent, minimal live retrieval experiment arms."""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import write_run
from .data import load_dataset
from .embedding_cache import EmbeddingCache, corpus_cache_identity
from .metrics import evaluate
from .providers import Jev, ProviderLedger, SiliconFlow
from .retrieval import BM25, cosine_topk, rrf


@dataclass(frozen=True)
class RunOptions:
    strategy: str
    query_ids: list[str]
    run_id: str
    run_root: Path
    candidate_k: int = 1000
    top_k: int = 10
    rrf_constant: int = 60
    route_threshold: float = 0.5
    min_laws: int = 1
    max_laws: int = 20
    jev_batch_size: int = 32
    pre_cost_ceiling_usd: float = 20.0
    cache_root: Path = Path(".cache/legal-stage-class")


def _qrel_map(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if float(row.get("relevance", 1)) > 0:
            output[str(row["query_id"])].add(str(row["article_id"]))
    return output


def _select_laws(scores: dict[str, float], threshold: float, minimum: int, maximum: int) -> list[str]:
    ordered = sorted(scores, key=lambda law_id: (-scores[law_id], law_id))
    selected = [law_id for law_id in ordered if scores[law_id] >= threshold][:maximum]
    if len(selected) < minimum:
        selected = ordered[:minimum]
    return selected[:maximum]


class ExperimentRunner:
    def __init__(self) -> None:
        # This cache exists only for one process (not on disk). It lets the
        # three independent acceptance arms use the same frozen corpus index.
        self._corpus_vectors: dict[str, list[list[float]]] = {}

    def run(self, dataset_path: Path, options: RunOptions) -> Path:
        if options.strategy not in {"baseline_hybrid", "jev_pre_route", "jev_post_rerank"}:
            raise ValueError("strategy must be baseline_hybrid, jev_pre_route, or jev_post_rerank")
        if options.min_laws < 1 or options.max_laws < options.min_laws:
            raise ValueError("route law bounds must satisfy 1 <= min_laws <= max_laws")
        dataset = load_dataset(dataset_path)
        query_by_id = {row["query_id"]: row for row in dataset["queries"]}
        missing = set(options.query_ids) - set(query_by_id)
        if missing:
            raise ValueError(f"unknown query ids: {sorted(missing)}")
        queries = [query_by_id[query_id] for query_id in options.query_ids]
        print(f"[run] {options.strategy}: {len(queries)} queries; loading BM25 and Dense index", file=sys.stderr, flush=True)
        ledger = ProviderLedger()
        allowed: dict[str, set[str]] = {}
        route_info: dict[str, Any] = {}
        route_elapsed_ms: dict[str, float] = {}
        query_ledger: dict[str, list[dict[str, Any]]] = {row["query_id"]: [] for row in queries}
        if options.strategy == "jev_pre_route":
            jev = Jev(ledger)
            # Full STARD runs are cost-calibrated before scoring the remaining queries.
            calibration = queries[: min(10, len(queries))] if len(queries) > 10 else []
            scored: dict[str, dict[str, float]] = {}
            for query in calibration:
                route_start = time.perf_counter()
                ledger_start = len(ledger.entries)
                scored[query["query_id"]] = jev.route_laws(query["text"], dataset["laws"], options.jev_batch_size)
                route_elapsed_ms[query["query_id"]] = (time.perf_counter() - route_start) * 1000
                query_ledger[query["query_id"]].extend(ledger.entries[ledger_start:])
            if calibration:
                known = ledger.known_cost
                reserved_per_judgment = float(os.environ.get("TYPESAFE_JEV_RESERVED_USD_PER_JUDGMENT", "0.00003"))
                predicted = known / len(calibration) * len(queries) if known else reserved_per_judgment * len(dataset["laws"]) * len(queries)
                if predicted > options.pre_cost_ceiling_usd:
                    raise RuntimeError(f"PRE calibration predicts ${predicted:.4f}, above ${options.pre_cost_ceiling_usd:.2f}; refusing full run")
                route_info["pre_calibration"] = {"query_count": len(calibration), "predicted_cost_usd": predicted, "ceiling_usd": options.pre_cost_ceiling_usd, "cost_basis": "provider_usage" if known else "reserved_per_judgment"}
            for query in queries:
                query_id = query["query_id"]
                if query_id in scored:
                    scores = scored[query_id]
                else:
                    route_start = time.perf_counter()
                    ledger_start = len(ledger.entries)
                    scores = jev.route_laws(query["text"], dataset["laws"], options.jev_batch_size)
                    route_elapsed_ms[query_id] = (time.perf_counter() - route_start) * 1000
                    query_ledger[query_id].extend(ledger.entries[ledger_start:])
                law_ids = _select_laws(scores, options.route_threshold, options.min_laws, options.max_laws)
                article_ids = set()
                for law in dataset["laws"]:
                    if law["law_id"] in law_ids:
                        article_ids.update(law["article_ids"])
                allowed[query_id] = article_ids
                route_info[query_id] = {"selected_law_ids": law_ids, "selected_law_scores": {law_id: scores[law_id] for law_id in law_ids}, "allowed_article_count": len(article_ids)}

        bm25 = BM25(dataset["corpus"])
        siliconflow = SiliconFlow(ledger)
        cache_key, cache_identity = corpus_cache_identity(
            dataset["dataset_hash"], dataset["corpus"], endpoint=siliconflow.endpoint, model=siliconflow.model,
        )
        cache = EmbeddingCache(options.cache_root)
        corpus_vectors = self._corpus_vectors.get(cache_key)
        memory_hit = corpus_vectors is not None
        if corpus_vectors is None:
            corpus_vectors = cache.load(cache_key, cache_identity, len(dataset["corpus"]))
        disk_hit = corpus_vectors is not None and not memory_hit
        if corpus_vectors is None:
            corpus_start = time.perf_counter()
            corpus_vectors = siliconflow.embed([f"{row['name']}\n{row['content']}" for row in dataset["corpus"]], "dense_corpus_embedding")
            corpus_embedding_ms = (time.perf_counter() - corpus_start) * 1000
            cache.store(cache_key, cache_identity, corpus_vectors)
            self._corpus_vectors[cache_key] = corpus_vectors
            corpus_embedding_source = "live"
        else:
            self._corpus_vectors[cache_key] = corpus_vectors
            corpus_embedding_ms = 0.0
            corpus_embedding_source = "disk_cache" if disk_hit else "in_memory_reuse"
            print("[run] reusing cached corpus embeddings", file=sys.stderr, flush=True)
        article_by_id = {row["article_id"]: row for row in dataset["corpus"]}
        rankings_rows: list[dict[str, Any]] = []
        latency_rows: list[dict[str, Any]] = []
        ranking_ids: dict[str, list[str]] = {}
        shortlists: dict[str, list[str]] = {}
        for query_number, query in enumerate(queries, 1):
            query_id = query["query_id"]
            print(f"[run] {options.strategy}: query {query_number}/{len(queries)} (id={query_id})", file=sys.stderr, flush=True)
            query_start = time.perf_counter()
            ledger_start = len(ledger.entries)
            vector = siliconflow.embed([query["text"]], "dense_query_embedding")[0]
            permitted = allowed.get(query_id)
            bm25_ranking = bm25.topk(query["text"], options.candidate_k, permitted)
            dense_ranking = cosine_topk(vector, dataset["corpus"], corpus_vectors, options.candidate_k, permitted)
            fused = rrf([bm25_ranking, dense_ranking], options.top_k, options.rrf_constant)
            shortlist = [article_id for article_id, _ in fused]
            shortlists[query_id] = shortlist
            post_metadata: dict[str, Any] = {}
            if options.strategy == "jev_post_rerank":
                before_post = time.perf_counter()
                reranked, post_metadata = Jev(ledger).rerank(query["text"], [article_by_id[article_id] for article_id in shortlist])
                final = reranked[: options.top_k]
                post_metadata["incremental_jev_post_ms"] = round((time.perf_counter() - before_post) * 1000, 3)
            else:
                final = fused
            query_ledger[query_id].extend(ledger.entries[ledger_start:])
            bm25_rank = {article_id: rank for rank, (article_id, _) in enumerate(bm25_ranking, 1)}
            dense_rank = {article_id: rank for rank, (article_id, _) in enumerate(dense_ranking, 1)}
            ranking_ids[query_id] = [article_id for article_id, _ in final]
            rankings_rows.append({
                "query_id": query_id, "strategy": options.strategy, "query": query["text"],
                "shortlist_article_ids": shortlist, "ranking": [
                    {"rank": rank, "article_id": article_id, "score": score, "bm25_rank": bm25_rank.get(article_id), "dense_rank": dense_rank.get(article_id)}
                    for rank, (article_id, score) in enumerate(final, 1)
                ],
                **({"pre_route": route_info[query_id]} if options.strategy == "jev_pre_route" else {}),
                **({"post_rerank": post_metadata} if options.strategy == "jev_post_rerank" else {}),
            })
            query_costs = [entry["cost_usd"] for entry in query_ledger[query_id]]
            latency_rows.append({
                "query_id": query_id, "strategy": options.strategy,
                "query_to_final_topk_ms": round(route_elapsed_ms.get(query_id, 0.0) + (time.perf_counter() - query_start) * 1000, 3),
                "cost_usd": round(sum(float(cost or 0.0) for cost in query_costs), 8) if query_costs else 0.0,
                "cost_status": "reserved_or_estimated",
                **({"incremental_jev_post_ms": post_metadata["incremental_jev_post_ms"]} if options.strategy == "jev_post_rerank" else {}),
            })
        routed = allowed if options.strategy == "jev_pre_route" else None
        metrics = evaluate(ranking_ids, _qrel_map(dataset["qrels"]), shortlists if options.strategy == "jev_post_rerank" else None, routed)
        metrics["known_provider_cost_usd"] = ledger.known_cost
        metrics["corpus_dense_embedding_ms"] = round(corpus_embedding_ms, 3)
        metrics["corpus_embedding_source"] = corpus_embedding_source
        return write_run(options.run_root, options.run_id, manifest={
            "run_id": options.run_id, "strategy": options.strategy, "dataset_hash": dataset["dataset_hash"],
            "query_ids": options.query_ids, "parameters": {
                "bm25_top_k": options.candidate_k, "dense_top_k": options.candidate_k, "rrf_constant": options.rrf_constant,
                "final_top_k": options.top_k, "route_threshold": options.route_threshold, "min_laws": options.min_laws, "max_laws": options.max_laws,
            }, "route_info": route_info, "provider_cost_usd": ledger.known_cost, "corpus_embedding_source": corpus_embedding_source,
            "corpus_embedding_cache_key": cache_key,
        }, rankings=rankings_rows, metrics=metrics, latency_cost=latency_rows, provider_ledger=ledger.entries)
