import json
from pathlib import Path

import legal_stage_class.experiment as experiment
from legal_stage_class.data import prepare_stard
from legal_stage_class.experiment import ExperimentRunner, RunOptions
from legal_stage_class.providers import ProviderLedger, SiliconFlow


def _write_raw(root: Path) -> None:
    root.mkdir()
    (root / "corpus.jsonl").write_text(
        "\n".join([
            json.dumps({"id": 1, "name": "劳动法第一条", "content": "保护劳动者合法权益"}, ensure_ascii=False),
            json.dumps({"id": 2, "name": "劳动法第二条", "content": "劳动关系适用本法"}, ensure_ascii=False),
            json.dumps({"id": 3, "name": "合同法第十条", "content": "合同可以书面订立"}, ensure_ascii=False),
        ]) + "\n", encoding="utf-8")
    (root / "queries.json").write_text(json.dumps([
        {"query_id": 11, "问题": "劳动者权益如何保护", "match_id": [1]},
        {"query_id": 12, "问题": "劳动关系如何认定", "match_id": [2]},
        {"query_id": 13, "问题": "合同要书面吗", "match_id": [3]},
    ], ensure_ascii=False), encoding="utf-8")
    (root / "qrels.train.tsv").write_text("11\t0\t1\t1\n12\t0\t2\t1\n", encoding="utf-8")
    (root / "qrels.dev.tsv").write_text("13\t0\t3\t1\n", encoding="utf-8")


class FakeSiliconFlow:
    operations = []
    endpoint = "https://test.invalid/v1/embeddings"
    model = "test-model"

    def __init__(self, ledger, **_kwargs):
        self.ledger = ledger

    def embed(self, texts, operation):
        self.operations.append(operation)
        self.ledger.add(provider="siliconflow", operation=operation, elapsed_ms=1, usage={"total_tokens": len(texts)}, request_id="sf-test", cost_per_1k=0)
        return [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]


class FakeJev:
    route_calls = 0
    rerank_calls = 0

    def __init__(self, ledger):
        self.ledger = ledger

    def route_laws(self, _query, laws, _batch_size):
        type(self).route_calls += 1
        self.ledger.add(provider="typesafe_jev", operation="jev_pre_noul", elapsed_ms=1, usage={"input_tokens": len(laws)}, request_id="jev-test", cost_per_1k=0)
        return {law["law_id"]: 0.9 if law["canonical_law_name"] == "劳动法" else 0.1 for law in laws}

    def rerank(self, _query, candidates):
        type(self).rerank_calls += 1
        return [(candidate["article_id"], 1.0 - index / 10) for index, candidate in enumerate(candidates)], {"choice": "a_1", "confidence": 0.9}


def test_prepare_derives_the_required_five_dataset_files(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    assert {path.name for path in dataset.iterdir()} == {"queries.jsonl", "corpus.jsonl", "qrels.jsonl", "splits.json", "law_catalog.jsonl"}
    laws = [json.loads(line) for line in (dataset / "law_catalog.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [law["canonical_law_name"] for law in laws] == ["劳动法", "合同法"]


def test_each_independent_arm_writes_only_the_five_run_artifacts(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    monkeypatch.setattr(experiment, "SiliconFlow", FakeSiliconFlow)
    monkeypatch.setattr(experiment, "Jev", FakeJev)
    for strategy in ("baseline_hybrid", "jev_pre_route", "jev_post_rerank"):
        destination = ExperimentRunner().run(dataset, RunOptions(
            strategy=strategy, query_ids=["11", "12", "13"], run_id=strategy,
            run_root=tmp_path / "runs", candidate_k=3, cache_root=tmp_path / "cache",
        ))
        assert {path.name for path in destination.iterdir()} == {"manifest.json", "rankings.jsonl", "metrics.json", "latency_cost.jsonl", "provider_ledger.jsonl"}
        metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
        assert {"recall_at_10", "ndcg_at_10", "hit_at_1"} <= metrics.keys()
    pre_metrics = json.loads((tmp_path / "runs" / "jev_pre_route" / "metrics.json").read_text(encoding="utf-8"))
    post_metrics = json.loads((tmp_path / "runs" / "jev_post_rerank" / "metrics.json").read_text(encoding="utf-8"))
    assert "routing_retention" in pre_metrics
    assert "shortlist_recall_at_10" in post_metrics


def test_one_acceptance_process_reuses_corpus_embeddings(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    FakeSiliconFlow.operations.clear()
    monkeypatch.setattr(experiment, "SiliconFlow", FakeSiliconFlow)
    monkeypatch.setattr(experiment, "Jev", FakeJev)
    runner = ExperimentRunner()
    for strategy in ("baseline_hybrid", "jev_post_rerank"):
        runner.run(dataset, RunOptions(
            strategy=strategy, query_ids=["11", "12", "13"], run_id=strategy,
            run_root=tmp_path / "runs", candidate_k=3, cache_root=tmp_path / "cache",
        ))
    assert FakeSiliconFlow.operations.count("dense_corpus_embedding") == 1
    manifest = json.loads((tmp_path / "runs" / "jev_post_rerank" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_embedding_source"] == "in_memory_reuse"


def test_fresh_runner_uses_verified_local_corpus_cache(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    FakeSiliconFlow.operations.clear()
    monkeypatch.setattr(experiment, "SiliconFlow", FakeSiliconFlow)
    cache_root = tmp_path / "cache"
    for run_id in ("first", "second"):
        ExperimentRunner().run(dataset, RunOptions(
            strategy="baseline_hybrid", query_ids=["11", "12", "13"], run_id=run_id,
            run_root=tmp_path / "runs", candidate_k=3, cache_root=cache_root,
        ))
    assert FakeSiliconFlow.operations.count("dense_corpus_embedding") == 1
    manifest = json.loads((tmp_path / "runs" / "second" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_embedding_source"] == "disk_cache"


def test_parallel_embedding_preserves_input_order(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    client = SiliconFlow(ProviderLedger(), batch_size=2, concurrency=2)
    monkeypatch.setattr(client, "_embed_batch", lambda batch, _operation: [[float(text)] for text in batch])
    assert client.embed(["1", "2", "3", "4"], "test") == [[1.0], [2.0], [3.0], [4.0]]


def test_fresh_runner_reuses_complete_jev_pre_and_post_decisions(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    FakeJev.route_calls = 0
    FakeJev.rerank_calls = 0
    monkeypatch.setattr(experiment, "SiliconFlow", FakeSiliconFlow)
    monkeypatch.setattr(experiment, "Jev", FakeJev)
    cache_root = tmp_path / "cache"
    for run_id in ("first-pre", "second-pre"):
        ExperimentRunner().run(dataset, RunOptions(
            strategy="jev_pre_route", query_ids=["11", "12", "13"], run_id=run_id,
            run_root=tmp_path / "runs", candidate_k=3, cache_root=cache_root,
        ))
    for run_id in ("first-post", "second-post"):
        ExperimentRunner().run(dataset, RunOptions(
            strategy="jev_post_rerank", query_ids=["11", "12", "13"], run_id=run_id,
            run_root=tmp_path / "runs", candidate_k=3, cache_root=cache_root,
        ))
    assert FakeJev.route_calls == 3
    assert FakeJev.rerank_calls == 3
    pre_manifest = json.loads((tmp_path / "runs" / "second-pre" / "manifest.json").read_text(encoding="utf-8"))
    post_manifest = json.loads((tmp_path / "runs" / "second-post" / "manifest.json").read_text(encoding="utf-8"))
    assert set(pre_manifest["jev_decision_cache_sources"]["pre_route"].values()) == {"disk_cache"}
    assert set(post_manifest["jev_decision_cache_sources"]["post_rerank"].values()) == {"disk_cache"}
