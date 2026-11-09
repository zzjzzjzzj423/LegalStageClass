"""Metric definitions used identically by all three experimental arms."""

from __future__ import annotations

import math
from typing import Any


def _per_query(ranking: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    top = ranking[:k]
    hits = [article_id in relevant for article_id in top]
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum((1.0 / math.log2(position + 1)) for position, hit in enumerate(hits, 1) if hit)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))
    return {"recall_at_10": recall, "ndcg_at_10": dcg / ideal if ideal else 0.0, "hit_at_1": float(bool(hits and hits[0]))}


def _retention(rankings: dict[str, list[str]], qrels: dict[str, set[str]], allowed: dict[str, set[str]]) -> tuple[float, float]:
    fractions = [len(allowed.get(query_id, set()) & qrels.get(query_id, set())) / max(1, len(qrels.get(query_id, set()))) for query_id in rankings]
    complete = [qrels.get(query_id, set()).issubset(allowed.get(query_id, set())) for query_id in rankings]
    return sum(fractions) / max(1, len(fractions)), sum(complete) / max(1, len(complete))


def evaluate(
    rankings: dict[str, list[str]], qrels: dict[str, set[str]], *, shortlist: dict[str, list[str]] | None = None,
    domain_allowed: dict[str, set[str]] | None = None, file_allowed: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    per_query = {query_id: _per_query(ranking, qrels.get(query_id, set())) for query_id, ranking in rankings.items()}
    result: dict[str, Any] = {"query_count": len(per_query), "per_query": per_query}
    for metric in ("recall_at_10", "ndcg_at_10", "hit_at_1"):
        result[metric] = sum(row[metric] for row in per_query.values()) / max(1, len(per_query))
    if shortlist is not None:
        result["shortlist_recall"] = sum(
            len(set(shortlist.get(query_id, [])) & qrels.get(query_id, set())) / max(1, len(qrels.get(query_id, set())))
            for query_id in rankings
        ) / max(1, len(rankings))
        result["shortlist_size"] = sum(len(shortlist.get(query_id, [])) for query_id in rankings) / max(1, len(rankings))
    if domain_allowed is not None:
        result["domain_retention"], result["domain_complete_retention"] = _retention(rankings, qrels, domain_allowed)
    if file_allowed is not None:
        result["file_retention"], result["file_complete_retention"] = _retention(rankings, qrels, file_allowed)
    return result
