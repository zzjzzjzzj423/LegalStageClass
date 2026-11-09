import json
from pathlib import Path

import legal_stage_class.experiment as experiment
from data.processed.prepare_stard import prepare_stard
from legal_stage_class.experiment import ExperimentRunner, RunOptions
from legal_stage_class.navigation import LocalBgeNavigator
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


def _write_file_domains(path: Path, dataset: Path) -> Path:
    laws = [json.loads(line) for line in (dataset / "law_catalog.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    for index, law in enumerate(laws):
        rows.append({**law, "domain_ids": [f"D{index + 1:02d}"]})
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


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
    file_choice_calls = 0
    rerank_calls = 0

    def __init__(self, ledger):
        self.ledger = ledger

    def choose_files(self, _query, laws):
        type(self).file_choice_calls += 1
        self.ledger.add(provider="typesafe_jev", operation="jev_file_choice", elapsed_ms=1, usage={"input_tokens": len(laws)}, request_id="jev-test", cost_per_1k=0)
        return [(law["law_id"], 1.0 - index / 10) for index, law in enumerate(laws)], {"choice": "f_test", "confidence": 0.9}

    def rerank(self, _query, candidates):
        type(self).rerank_calls += 1
        return [(candidate["article_id"], 1.0 - index / 10) for index, candidate in enumerate(candidates)], {"choice": "a_1", "confidence": 0.9}


class FakeNavigator:
    def __init__(self, _path, _model, laws):
        self.laws = laws

    def select_domains(self, _query, initial_n, max_files):
        laws = self.laws[: min(initial_n, len(self.laws))]
        while len(laws) > max_files:
            laws.pop()
        return {"ranked_domains": [{"domain_id": "D01", "similarity": 1.0}], "selected_domain_ids": ["D01"],
                "file_count": len(laws), "candidate_law_ids": [law["law_id"] for law in laws], "laws": laws}

    def warm(self):
        return None

    def embed(self, texts, _operation):
        return [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]

    def rank_titles(self, _query, laws):
        return [(law["law_id"], 1.0 - index / 10) for index, law in enumerate(laws)]


def _options(tmp_path, dataset, strategy, run_id, **kwargs):
    cache_root = kwargs.pop("cache_root", tmp_path / "cache")
    dense_backend = kwargs.pop("dense_backend", "siliconflow")
    return RunOptions(strategy=strategy, query_ids=["11", "12", "13"], run_id=run_id,
                      run_root=tmp_path / "runs", candidate_k=3, cache_root=cache_root,
                      dense_backend=dense_backend, file_domains_path=_write_file_domains(tmp_path / "domains.jsonl", dataset), **kwargs)


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
    monkeypatch.setattr(experiment, "LocalBgeNavigator", FakeNavigator)
    for strategy in ("baseline_hybrid", "jev_pre_route", "jev_post_rerank"):
        destination = ExperimentRunner().run(dataset, _options(tmp_path, dataset, strategy, strategy))
        assert {path.name for path in destination.iterdir()} == {"manifest.json", "rankings.jsonl", "metrics.json", "latency_cost.jsonl", "provider_ledger.jsonl"}
        metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
        assert {"recall_at_10", "ndcg_at_10", "hit_at_1"} <= metrics.keys()
    pre_metrics = json.loads((tmp_path / "runs" / "jev_pre_route" / "metrics.json").read_text(encoding="utf-8"))
    post_metrics = json.loads((tmp_path / "runs" / "jev_post_rerank" / "metrics.json").read_text(encoding="utf-8"))
    assert "file_retention" in pre_metrics
    assert post_metrics["shortlist_size"] == 3


def test_one_acceptance_process_reuses_corpus_embeddings(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    FakeSiliconFlow.operations.clear()
    monkeypatch.setattr(experiment, "SiliconFlow", FakeSiliconFlow)
    monkeypatch.setattr(experiment, "Jev", FakeJev)
    monkeypatch.setattr(experiment, "LocalBgeNavigator", FakeNavigator)
    runner = ExperimentRunner()
    for strategy in ("baseline_hybrid", "jev_post_rerank"):
        runner.run(dataset, _options(tmp_path, dataset, strategy, strategy))
    assert FakeSiliconFlow.operations.count("dense_corpus_embedding") == 1
    manifest = json.loads((tmp_path / "runs" / "jev_post_rerank" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_embedding_source"] == "in_memory_reuse"


def test_fresh_runner_uses_verified_local_corpus_cache(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    FakeSiliconFlow.operations.clear()
    monkeypatch.setattr(experiment, "SiliconFlow", FakeSiliconFlow)
    monkeypatch.setattr(experiment, "LocalBgeNavigator", FakeNavigator)
    cache_root = tmp_path / "cache"
    for run_id in ("first", "second"):
        ExperimentRunner().run(dataset, _options(tmp_path, dataset, "baseline_hybrid", run_id, cache_root=cache_root))
    assert FakeSiliconFlow.operations.count("dense_corpus_embedding") == 1
    manifest = json.loads((tmp_path / "runs" / "second" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_embedding_source"] == "disk_cache"


def test_parallel_embedding_preserves_input_order(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    client = SiliconFlow(ProviderLedger(), batch_size=2, concurrency=2)
    monkeypatch.setattr(client, "_embed_batch", lambda batch, _operation: [[float(text)] for text in batch])
    assert client.embed(["1", "2", "3", "4"], "test") == [[1.0], [2.0], [3.0], [4.0]]


def test_local_dense_backend_never_constructs_siliconflow(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    monkeypatch.setattr(experiment, "LocalBgeNavigator", FakeNavigator)
    monkeypatch.setattr(experiment, "SiliconFlow", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("remote Dense must not be created")))
    destination = ExperimentRunner().run(dataset, _options(tmp_path, dataset, "baseline_hybrid", "local", dense_backend="local"))
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["dense_backend"] == "local"


def test_domain_route_keeps_the_largest_prefix_that_fits_the_file_cap():
    navigator = object.__new__(LocalBgeNavigator)
    navigator.laws = [
        {"law_id": "a", "domain_ids": ["D01"], "article_ids": ["1"]},
        {"law_id": "b", "domain_ids": ["D01"], "article_ids": ["2"]},
        {"law_id": "c", "domain_ids": ["D02"], "article_ids": ["3"]},
        {"law_id": "d", "domain_ids": ["D03"], "article_ids": ["4"]},
    ]
    navigator._domain_text = {"D01": "one", "D02": "two", "D03": "three"}
    navigator._rank = lambda _query, _pairs: [("D01", 0.9), ("D02", 0.8), ("D03", 0.7)]
    route = navigator.select_domains("q", initial_n=3, max_files=3)
    assert route["selected_domain_ids"] == ["D01", "D02"]
    assert route["file_count"] == 3


def test_fresh_runner_reuses_complete_jev_pre_and_post_decisions(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _write_raw(raw)
    dataset = prepare_stard(raw, tmp_path / "processed")
    FakeJev.file_choice_calls = 0
    FakeJev.rerank_calls = 0
    monkeypatch.setattr(experiment, "SiliconFlow", FakeSiliconFlow)
    monkeypatch.setattr(experiment, "Jev", FakeJev)
    monkeypatch.setattr(experiment, "LocalBgeNavigator", FakeNavigator)
    cache_root = tmp_path / "cache"
    for run_id in ("first-pre", "second-pre"):
        ExperimentRunner().run(dataset, _options(tmp_path, dataset, "jev_pre_route", run_id, cache_root=cache_root))
    for run_id in ("first-post", "second-post"):
        ExperimentRunner().run(dataset, _options(tmp_path, dataset, "jev_post_rerank", run_id, cache_root=cache_root))
    assert FakeJev.file_choice_calls == 3
    assert FakeJev.rerank_calls == 3
    pre_manifest = json.loads((tmp_path / "runs" / "second-pre" / "manifest.json").read_text(encoding="utf-8"))
    post_manifest = json.loads((tmp_path / "runs" / "second-post" / "manifest.json").read_text(encoding="utf-8"))
    assert set(pre_manifest["jev_decision_cache_sources"]["file_choice"].values()) == {"disk_cache"}
    assert set(post_manifest["jev_decision_cache_sources"]["post_rerank"].values()) == {"disk_cache"}
