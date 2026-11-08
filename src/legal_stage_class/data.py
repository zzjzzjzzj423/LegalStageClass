"""STARD normalization and the mechanically derived law catalog."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

from .util import digest, read_jsonl, write_json, write_jsonl

ARTICLE_SUFFIX = re.compile(r"^(?P<law>.+?)第[〇零一二三四五六七八九十百千万0-9]+条(?:之[〇零一二三四五六七八九十百千万0-9]+)?$")


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _normalise_queries(source: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("queries") or raw.get("data") or list(raw.values())
    if not isinstance(raw, list):
        raise ValueError("queries.json must contain a list of query records")
    queries, qrels = [], []
    for row in raw:
        query_id = _first(row, ("query_id", "id", "qid"))
        text = _first(row, ("问题", "query", "text", "question"))
        matches = _first(row, ("match_id", "relevant_ids", "article_ids")) or []
        if query_id is None or not text:
            raise ValueError("every query needs query_id and 问题/query text")
        query_id = str(query_id)
        queries.append({"query_id": query_id, "text": str(text)})
        for article_id in matches:
            qrels.append({"query_id": query_id, "article_id": str(article_id), "relevance": 1})
    return queries, qrels


def _normalise_corpus(source: Path) -> list[dict[str, str]]:
    rows = read_jsonl(source)
    corpus: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        article_id = _first(row, ("id", "article_id", "docid"))
        name = _first(row, ("name", "title"))
        content = _first(row, ("content", "text", "contents"))
        if article_id is None or not name or content is None:
            raise ValueError("every corpus record needs id, name, and content")
        article_id = str(article_id)
        if article_id in seen:
            raise ValueError(f"duplicate corpus article id: {article_id}")
        seen.add(article_id)
        corpus.append({"article_id": article_id, "name": str(name).strip(), "content": str(content).strip()})
    return corpus


def derive_law_catalog(corpus: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[str, list[str]] = {}
    article_laws: dict[str, str] = {}
    failures: list[str] = []
    for article in corpus:
        match = ARTICLE_SUFFIX.fullmatch(article["name"].strip())
        if not match:
            failures.append(f"{article['article_id']}:{article['name']}")
            continue
        canonical_name = match.group("law").strip()
        grouped.setdefault(canonical_name, []).append(article["article_id"])
    if failures:
        preview = "; ".join(failures[:5])
        raise ValueError(f"cannot derive law name from {len(failures)} article names: {preview}")
    catalog = []
    for canonical_name in sorted(grouped):
        law_id = "law_" + hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:16]
        article_ids = sorted(grouped[canonical_name], key=lambda value: (len(value), value))
        catalog.append({"law_id": law_id, "canonical_law_name": canonical_name, "article_ids": article_ids})
        article_laws.update({article_id: law_id for article_id in article_ids})
    return catalog, article_laws


def _ids_from_qrel_file(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0].strip() and fields[0].strip().lower() not in {"query-id", "query_id", "qid"}:
                ids.add(fields[0].strip())
    return ids


def _splits(raw_root: Path, query_ids: list[str]) -> dict[str, list[str]]:
    train_qrels, dev_qrels = raw_root / "qrels.train.tsv", raw_root / "qrels.dev.tsv"
    if train_qrels.exists() and dev_qrels.exists():
        train = sorted(_ids_from_qrel_file(train_qrels))
        final = sorted(_ids_from_qrel_file(dev_qrels))
        unknown = (set(train) | set(final)) - set(query_ids)
        if unknown:
            raise ValueError(f"qrels refer to absent queries: {sorted(unknown)[:5]}")
    else:
        ordered = sorted(query_ids)
        random.Random(20261108).shuffle(ordered)
        boundary = max(1, round(len(ordered) * 0.8))
        train, final = sorted(ordered[:boundary]), sorted(ordered[boundary:])
    return {"tune_train": train, "final": final, "all": sorted(query_ids)}


def prepare_stard(raw_root: Path, output_root: Path) -> Path:
    corpus_source, queries_source = raw_root / "corpus.jsonl", raw_root / "queries.json"
    if not corpus_source.is_file() or not queries_source.is_file():
        raise ValueError("raw root must contain corpus.jsonl and queries.json")
    corpus = _normalise_corpus(corpus_source)
    queries, qrels = _normalise_queries(queries_source)
    corpus_ids, query_ids = {row["article_id"] for row in corpus}, [row["query_id"] for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query ids")
    unknown_qrels = {row["article_id"] for row in qrels} - corpus_ids
    if unknown_qrels:
        raise ValueError(f"qrels refer to absent articles: {sorted(unknown_qrels)[:5]}")
    catalog, article_laws = derive_law_catalog(corpus)
    for article in corpus:
        article["law_id"] = article_laws[article["article_id"]]
    splits = _splits(raw_root, query_ids)
    dataset_hash = digest({"corpus": corpus, "queries": queries, "qrels": qrels, "splits": splits})
    destination = output_root / dataset_hash
    if destination.exists():
        return destination
    staging = output_root / f".{dataset_hash}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_jsonl(staging / "queries.jsonl", queries)
    write_jsonl(staging / "corpus.jsonl", corpus)
    write_jsonl(staging / "qrels.jsonl", qrels)
    write_json(staging / "splits.json", splits)
    write_jsonl(staging / "law_catalog.jsonl", catalog)
    staging.rename(destination)
    return destination


def load_dataset(dataset: Path) -> dict[str, Any]:
    required = ["queries.jsonl", "corpus.jsonl", "qrels.jsonl", "splits.json", "law_catalog.jsonl"]
    missing = [name for name in required if not (dataset / name).is_file()]
    if missing:
        raise ValueError(f"not a processed STARD dataset; missing: {', '.join(missing)}")
    corpus = read_jsonl(dataset / "corpus.jsonl")
    return {
        "root": dataset,
        "queries": read_jsonl(dataset / "queries.jsonl"),
        "corpus": corpus,
        "qrels": read_jsonl(dataset / "qrels.jsonl"),
        "splits": json.loads((dataset / "splits.json").read_text(encoding="utf-8")),
        "laws": read_jsonl(dataset / "law_catalog.jsonl"),
        "dataset_hash": dataset.name,
    }
