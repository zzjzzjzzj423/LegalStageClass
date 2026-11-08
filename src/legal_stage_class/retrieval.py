"""Local BM25, SiliconFlow BGE-M3 embeddings, and reciprocal-rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, corpus: list[dict[str, str]], k1: float = 1.2, b: float = 0.75):
        self.corpus, self.k1, self.b = corpus, k1, b
        self.docs = [Counter(tokenize(f"{row['name']} {row['content']}")) for row in corpus]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for document in self.docs:
            document_frequency.update(document.keys())
        self.idf = {term: math.log(1 + (len(corpus) - frequency + 0.5) / (frequency + 0.5)) for term, frequency in document_frequency.items()}

    def topk(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        terms = Counter(tokenize(query))
        scored: list[tuple[str, float]] = []
        for index, (row, document) in enumerate(zip(self.corpus, self.docs)):
            if allowed_ids is not None and row["article_id"] not in allowed_ids:
                continue
            score = 0.0
            denominator_norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / max(1.0, self.average_length))
            for term, qtf in terms.items():
                tf = document.get(term, 0)
                if tf:
                    score += qtf * self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + denominator_norm)
            scored.append((row["article_id"], score))
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k]


def cosine_topk(query_vector: list[float], corpus: list[dict[str, str]], vectors: list[list[float]], k: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
    query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
    scored = []
    for row, vector in zip(corpus, vectors):
        if allowed_ids is not None and row["article_id"] not in allowed_ids:
            continue
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        score = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * norm)
        scored.append((row["article_id"], score))
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k]


def rrf(rankings: list[list[tuple[str, float]]], k: int, constant: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (article_id, _) in enumerate(ranking, 1):
            scores[article_id] = scores.get(article_id, 0.0) + 1 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
