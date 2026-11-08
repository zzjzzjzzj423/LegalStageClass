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


def evaluate(rankings: dict[str, list[str]], qrels: dict[str, set[str]], shortlist: dict[str, list[str]] | None = None, routed: dict[str, set[str]] | None = None) -> dict[str, Any]:
    per_query = {query_id: _per_query(ranking, qrels.get(query_id, set())) for query_id, ranking in rankings.items()}
    result: dict[str, Any] = {"query_count": len(per_query), "per_query": per_query}
    for metric in ("recall_at_10", "ndcg_at_10", "hit_at_1"):
        result[metric] = sum(row[metric] for row in per_query.values()) / max(1, len(per_query))
    if shortlist is not None:
        result["shortlist_recall_at_10"] = sum(
            len(set(shortlist.get(query_id, [])) & qrels.get(query_id, set())) / max(1, len(qrels.get(query_id, set())))
            for query_id in rankings
        ) / max(1, len(rankings))
    if routed is not None:
        fractions = [len(routed.get(query_id, set()) & qrels.get(query_id, set())) / max(1, len(qrels.get(query_id, set()))) for query_id in rankings]
        result["routing_retention"] = sum(fractions) / max(1, len(fractions))
    return result
