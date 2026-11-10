"""Local BGE navigation over the fixed 22-domain and file-title directory."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from data.processed.classify_domains import DEFINITIONS


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class LocalBgeNavigator:
    """Encode only navigation strings locally; it is never the article retriever.

    ``transformers`` and ``torch`` are optional package dependencies because the
    production dense retriever is remote.  They are required only when this
    navigation experiment is executed, and the model is local-only by default.
    """

    def __init__(self, file_domains_path: Path, model_name: str, laws: list[dict[str, Any]]) -> None:
        if not file_domains_path.is_file():
            raise ValueError(f"file domains catalog does not exist: {file_domains_path}")
        rows = [json.loads(line) for line in file_domains_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {str(row["law_id"]): row for row in rows}
        source_ids, dataset_ids = set(by_id), {str(law["law_id"]) for law in laws}
        if source_ids != dataset_ids:
            missing, extra = sorted(dataset_ids - source_ids), sorted(source_ids - dataset_ids)
            raise ValueError(f"file domains catalog does not match dataset laws (missing={missing[:3]}, extra={extra[:3]})")
        self.laws = [dict(law, domain_ids=by_id[str(law["law_id"])]["domain_ids"]) for law in laws]
        self.model_name = model_name
        self._vectors: dict[str, list[float]] = {}
        self._domain_text = {
            domain_id: f"{name}\n关键词：{pattern.replace('|', ' ')}" for domain_id, _parent, name, pattern in DEFINITIONS
        }

    def _encode(self, texts: list[str], operation: str | None = None) -> list[list[float]]:
        unseen = [text for text in texts if text not in self._vectors]
        if unseen:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on local optional setup
                raise RuntimeError("领域导航需要安装 torch 和 transformers，并在本地准备 BAAI/bge-small-zh-v1.5") from exc
            if not hasattr(self, "_tokenizer"):
                local_only = os.environ.get("LEGAL_STAGE_CLASS_NAVIGATION_LOCAL_ONLY", "1") != "0"
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only)
                self._model = AutoModel.from_pretrained(self.model_name, local_files_only=local_only)
                self._model.eval()
            total_batches = max(1, (len(unseen) + 31) // 32)
            report_every = max(1, total_batches // 20)
            with torch.no_grad():
                for start in range(0, len(unseen), 32):
                    batch = unseen[start : start + 32]
                    tokens = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    output = self._model(**tokens).last_hidden_state[:, 0, :]
                    output = torch.nn.functional.normalize(output, p=2, dim=1)
                    for text, vector in zip(batch, output.tolist()):
                        self._vectors[text] = [float(value) for value in vector]
                    batch_number = start // 32 + 1
                    if operation and (batch_number == 1 or batch_number == total_batches or batch_number % report_every == 0):
                        print(f"[local-bge] {operation}: {batch_number}/{total_batches} batches complete ({batch_number / total_batches:.0%})", file=sys.stderr, flush=True)
        return [self._vectors[text] for text in texts]

    def _rank(self, query: str, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        query_prefix = os.environ.get("LEGAL_STAGE_CLASS_NAVIGATION_QUERY_PREFIX", "")
        query_vector = self._encode([query_prefix + query])[0]
        vectors = self._encode([text for _, text in pairs])
        return sorted(((identifier, _cosine(query_vector, vector)) for (identifier, _), vector in zip(pairs, vectors)), key=lambda pair: (-pair[1], pair[0]))

    def warm(self) -> None:
        """Load the local model and encode the fixed domain representations."""
        self._encode(list(self._domain_text.values()))

    def embed(self, texts: list[str], operation: str) -> list[list[float]]:
        """Embed local texts for the Dense arm without a remote provider call."""
        return self._encode(texts, operation)

    def select_domains(self, query: str, initial_n: int, max_files: int) -> dict[str, Any]:
        ranked = self._rank(query, sorted(self._domain_text.items()))
        selected_domains = []
        selected_laws = set()
        for domain_id, _ in ranked[:initial_n]:
            new_laws = [law for law in self.laws
                        if domain_id in law["domain_ids"] and law["law_id"] not in selected_laws]
            remaining = max_files - len(selected_laws)
            if len(new_laws) > remaining:
                ranked_titles = self.rank_titles(query, new_laws)
                selected_laws.update(law_id for law_id, _ in ranked_titles[:remaining])
            else:
                selected_laws.update(law["law_id"] for law in new_laws)
            selected_domains.append(domain_id)
            if len(selected_laws) == max_files:
                break
        laws = [law for law in self.laws if law["law_id"] in selected_laws]
        return {
            "ranked_domains": [{"domain_id": domain_id, "similarity": score} for domain_id, score in ranked],
            "selected_domain_ids": selected_domains,
            "file_count": len(laws), "candidate_law_ids": [law["law_id"] for law in laws], "laws": laws,
        }

    def rank_titles(self, query: str, laws: list[dict[str, Any]]) -> list[tuple[str, float]]:
        return self._rank(query, [(law["law_id"], law["canonical_law_name"]) for law in laws])
