"""The three domain → file → article retrieval experiment arms.

The arms share every stage except the studied intervention: the baseline ranks
file titles locally, PRE asks Jev to rank that same file directory, and POST
reranks the baseline article shortlist.  PRE and POST are never combined.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.processed.prepare_stard import load_dataset

from .artifacts import write_run
from .embedding_cache import EmbeddingCache, corpus_cache_identity
from .jev_cache import JevDecisionCache, file_choice_cache_identity, post_rerank_cache_identity
from .metrics import evaluate
from .navigation import LocalBgeNavigator
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
    domain_initial_n: int = 4
    choice_max_files: int = 255
    selected_files: int = 2
    post_shortlist_n: int = 20
    dense_backend: str = "local"
    file_domains_path: Path = Path("data/processed/file_domains.jsonl")
    navigation_model: str = "BAAI/bge-small-zh-v1.5"
    cache_root: Path = Path(".cache/legal-stage-class")


def _qrel_map(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if float(row.get("relevance", 1)) > 0:
            output[str(row["query_id"])].add(str(row["article_id"]))
    return output


def _article_ids(laws: list[dict[str, Any]]) -> set[str]:
    return {article_id for law in laws for article_id in law["article_ids"]}


class ExperimentRunner:
    """Run independent arms while reusing only frozen corpus vectors."""

    def __init__(self) -> None:
        self._corpus_vectors: dict[str, list[list[float]]] = {}
        self._navigators: dict[tuple[Path, str], LocalBgeNavigator] = {}

    def _navigator(self, options: RunOptions, laws: list[dict[str, Any]]) -> LocalBgeNavigator:
        key = (options.file_domains_path.resolve(), options.navigation_model)
        navigator = self._navigators.get(key)
        if navigator is None:
            navigator = LocalBgeNavigator(options.file_domains_path, options.navigation_model, laws)
            self._navigators[key] = navigator
        return navigator

    def run(self, dataset_path: Path, options: RunOptions) -> Path:
        if options.strategy not in {"baseline_hybrid", "jev_pre_route", "jev_post_rerank"}:
            raise ValueError("strategy must be baseline_hybrid, jev_pre_route, or jev_post_rerank")
        if options.domain_initial_n < 1 or options.choice_max_files < 1 or options.selected_files < 1:
            raise ValueError("domain_initial_n, choice_max_files, and selected_files must be positive")
        if options.dense_backend not in {"local", "siliconflow"}:
            raise ValueError("dense_backend must be local or siliconflow")
        if options.post_shortlist_n < options.top_k:
            raise ValueError("post_shortlist_n must be at least top_k")

        dataset = load_dataset(dataset_path)
        query_by_id = {row["query_id"]: row for row in dataset["queries"]}
        missing = set(options.query_ids) - set(query_by_id)
        if missing:
            raise ValueError(f"unknown query ids: {sorted(missing)}")
        queries = [query_by_id[query_id] for query_id in options.query_ids]
        navigator = self._navigator(options, dataset["laws"])
        navigation_start = time.perf_counter()
        navigator.warm()
        navigation_warm_ms = (time.perf_counter() - navigation_start) * 1000
        ledger = ProviderLedger()
        jev_cache = JevDecisionCache(options.cache_root)
        article_by_id = {row["article_id"]: row for row in dataset["corpus"]}

        print(f"[run] {options.strategy}: {len(queries)} queries; loading shared indices", file=sys.stderr, flush=True)
        bm25 = BM25(dataset["corpus"])
        if options.dense_backend == "siliconflow":
            dense_client: SiliconFlow | None = SiliconFlow(ledger)
            dense_endpoint, dense_model, dense_provider = dense_client.endpoint, dense_client.model, "siliconflow"
        else:
            dense_client = None
            dense_endpoint, dense_model, dense_provider = "local://transformers", options.navigation_model, "local_bge"
        vector_cache_key, vector_identity = corpus_cache_identity(
            dataset["dataset_hash"], dataset["corpus"], endpoint=dense_endpoint, model=dense_model, provider=dense_provider,
        )
        vector_cache = EmbeddingCache(options.cache_root)
        corpus_vectors = self._corpus_vectors.get(vector_cache_key)
        memory_hit = corpus_vectors is not None
        if corpus_vectors is None:
            corpus_vectors = vector_cache.load(vector_cache_key, vector_identity, len(dataset["corpus"]))
        disk_hit = corpus_vectors is not None and not memory_hit
        if corpus_vectors is None:
            started = time.perf_counter()
            corpus_texts = [f"{row['name']}\n{row['content']}" for row in dataset["corpus"]]
            corpus_vectors = (dense_client.embed(corpus_texts, "dense_corpus_embedding") if dense_client
                              else navigator.embed(corpus_texts, "dense_corpus_embedding"))
            corpus_embedding_ms = (time.perf_counter() - started) * 1000
            vector_cache.store(vector_cache_key, vector_identity, corpus_vectors)
            corpus_embedding_source = "live"
        else:
            corpus_embedding_ms = 0.0
            corpus_embedding_source = "disk_cache" if disk_hit else "in_memory_reuse"
        self._corpus_vectors[vector_cache_key] = corpus_vectors

        rankings_rows: list[dict[str, Any]] = []
        latency_rows: list[dict[str, Any]] = []
        final_rankings: dict[str, list[str]] = {}
        domain_allowed: dict[str, set[str]] = {}
        file_allowed: dict[str, set[str]] = {}
        shortlists: dict[str, list[str]] = {}
        cache_sources: dict[str, dict[str, str]] = {"file_choice": {}, "post_rerank": {}}

        for number, query in enumerate(queries, 1):
            query_id, text = query["query_id"], query["text"]
            print(f"[run] {options.strategy}: query {number}/{len(queries)} (id={query_id})", file=sys.stderr, flush=True)
            started = time.perf_counter()
            provider_start = len(ledger.entries)

            domain_route = navigator.select_domains(text, options.domain_initial_n, options.choice_max_files)
            candidate_laws = domain_route["laws"]
            domain_allowed[query_id] = _article_ids(candidate_laws)

            if options.strategy == "jev_pre_route":
                file_cache_key, file_identity = file_choice_cache_identity(text, candidate_laws)
                cached = jev_cache.load_file_choice(file_cache_key, file_identity, [law["law_id"] for law in candidate_laws])
                if cached is None:
                    ranked_laws, file_metadata = Jev(ledger).choose_files(text, candidate_laws)
                    jev_cache.store_file_choice(file_cache_key, file_identity, ranked_laws, file_metadata)
                    cache_sources["file_choice"][query_id] = "live"
                else:
                    ranked_laws, file_metadata = cached
                    cache_sources["file_choice"][query_id] = "disk_cache"
                selected_laws = ranked_laws[: options.selected_files]
                file_selection = {
                    "method": "jev_choice", "selected_law_ids": [law_id for law_id, _ in selected_laws],
                    "probabilities": {law_id: probability for law_id, probability in ranked_laws}, **file_metadata,
                }
            else:
                ranked_laws = navigator.rank_titles(text, candidate_laws)
                selected_laws = ranked_laws[: options.selected_files]
                file_selection = {
                    "method": "title_similarity", "selected_law_ids": [law_id for law_id, _ in selected_laws],
                    "similarities": {law_id: score for law_id, score in ranked_laws},
                }

            law_by_id = {law["law_id"]: law for law in candidate_laws}
            chosen_laws = [law_by_id[law_id] for law_id, _ in selected_laws]
            permitted = _article_ids(chosen_laws)
            if not permitted:
                raise RuntimeError("file selection produced no allowed articles")
            file_allowed[query_id] = permitted

            query_vector = (dense_client.embed([text], "dense_query_embedding")[0] if dense_client
                            else navigator.embed([text], "dense_query_embedding")[0])
            bm25_ranking = bm25.topk(text, options.candidate_k, permitted)
            dense_ranking = cosine_topk(query_vector, dataset["corpus"], corpus_vectors, options.candidate_k, permitted)
            fusion_depth = options.post_shortlist_n if options.strategy == "jev_post_rerank" else options.top_k
            fused = rrf([bm25_ranking, dense_ranking], fusion_depth, options.rrf_constant)
            shortlist = [article_id for article_id, _ in fused]
            shortlists[query_id] = shortlist

            post_metadata: dict[str, Any] = {}
            if options.strategy == "jev_post_rerank":
                candidates = [article_by_id[article_id] for article_id in shortlist]
                post_cache_key, post_identity = post_rerank_cache_identity(text, candidates)
                cached_post = jev_cache.load_post_rerank(post_cache_key, post_identity, shortlist)
                if cached_post is None:
                    reranked, post_metadata = Jev(ledger).rerank(text, candidates)
                    jev_cache.store_post_rerank(post_cache_key, post_identity, reranked, post_metadata)
                    cache_sources["post_rerank"][query_id] = "live"
                else:
                    reranked, post_metadata = cached_post
                    cache_sources["post_rerank"][query_id] = "disk_cache"
                final = reranked[: options.top_k]
            else:
                final = fused[: options.top_k]

            bm25_rank = {article_id: rank for rank, (article_id, _) in enumerate(bm25_ranking, 1)}
            dense_rank = {article_id: rank for rank, (article_id, _) in enumerate(dense_ranking, 1)}
            final_rankings[query_id] = [article_id for article_id, _ in final]
            query_ledger = ledger.entries[provider_start:]
            rankings_rows.append({
                "query_id": query_id, "strategy": options.strategy, "query": text,
                "domain_route": {key: value for key, value in domain_route.items() if key != "laws"},
                "file_selection": file_selection, "allowed_article_count": len(permitted),
                "shortlist_article_ids": shortlist,
                "ranking": [
                    {"rank": rank, "article_id": article_id, "score": score,
                     "bm25_rank": bm25_rank.get(article_id), "dense_rank": dense_rank.get(article_id)}
                    for rank, (article_id, score) in enumerate(final, 1)
                ],
                **({"post_rerank": post_metadata} if options.strategy == "jev_post_rerank" else {}),
            })
            latency_rows.append({
                "query_id": query_id, "strategy": options.strategy,
                "query_to_final_topk_ms": round((time.perf_counter() - started) * 1000, 3),
                "cost_usd": round(sum(float(entry["cost_usd"] or 0.0) for entry in query_ledger), 8),
                "cost_status": "reserved_or_estimated",
            })

        metrics = evaluate(
            final_rankings, _qrel_map(dataset["qrels"]), domain_allowed=domain_allowed,
            file_allowed=file_allowed, shortlist=shortlists if options.strategy == "jev_post_rerank" else None,
        )
        metrics.update({"known_provider_cost_usd": ledger.known_cost, "navigation_warm_ms": round(navigation_warm_ms, 3),
                        "corpus_dense_embedding_ms": round(corpus_embedding_ms, 3),
                        "corpus_embedding_source": corpus_embedding_source})
        return write_run(options.run_root, options.run_id, manifest={
            "run_id": options.run_id, "strategy": options.strategy, "dataset_hash": dataset["dataset_hash"],
            "query_ids": options.query_ids,
            "parameters": {"domain_initial_n": options.domain_initial_n, "choice_max_files": options.choice_max_files,
                           "selected_files": options.selected_files, "bm25_top_k": options.candidate_k,
                           "dense_top_k": options.candidate_k, "rrf_constant": options.rrf_constant,
                           "post_shortlist_n": options.post_shortlist_n, "final_top_k": options.top_k,
                           "navigation_model": options.navigation_model, "dense_backend": options.dense_backend,
                           "dense_model": dense_model, "file_domains_path": str(options.file_domains_path)},
            "provider_cost_usd": ledger.known_cost, "navigation_warm_ms": round(navigation_warm_ms, 3),
            "corpus_embedding_source": corpus_embedding_source,
            "corpus_embedding_cache_key": vector_cache_key, "jev_decision_cache_sources": cache_sources,
        }, rankings=rankings_rows, metrics=metrics, latency_cost=latency_rows, provider_ledger=ledger.entries)
