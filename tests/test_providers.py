import http.client
import json
import sqlite3

import pytest

from legal_stage_class import providers


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"answers": {}}).encode("utf-8")


def test_post_json_retries_a_remote_disconnect(monkeypatch):
    calls = []

    def fake_urlopen(*_args, **_kwargs):
        calls.append(None)
        if len(calls) < 3:
            raise http.client.RemoteDisconnected("closed")
        return _Response()

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(providers.time, "sleep", lambda _seconds: None)
    body, _elapsed = providers._post_json("https://test.invalid", "key", {"x": 1})
    assert body == {"answers": {}}
    assert len(calls) == 3


def test_siliconflow_reranker_parses_scores_and_usage(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setattr(
        providers,
        "_post_json",
        lambda *_args, **_kwargs: ({
            "id": "rerank-1",
            "model": "Qwen/Qwen3-Reranker-8B",
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ],
            "meta": {"tokens": {"input_tokens": 12, "output_tokens": 0}},
        }, 3.5),
    )
    ledger = providers.ProviderLedger()
    ranking, metadata = providers.SiliconFlowReranker(ledger).rerank(
        "query",
        [
            {"article_id": "a", "name": "n1", "content": "c1"},
            {"article_id": "b", "name": "n2", "content": "c2"},
        ],
    )
    assert ranking == [("b", 0.9), ("a", 0.2)]
    assert metadata["model"] == "Qwen/Qwen3-Reranker-8B"
    assert ledger.entries[0]["usage"]["input_tokens"] == 12


def test_jev_budget_survives_retries_and_blocks_the_next_attempt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(providers.time, "sleep", lambda _seconds: None)
    calls = []

    class Answer(_Response):
        def read(self):
            return json.dumps({
                "id": "request-1", "model": "jev-1.14.0",
                "usage": {"input_tokens": 123, "output_tokens": 50, "total_tokens": 173},
                "answers": {"file_choice": {"probabilities": {"f_a": 1.0}}},
            }).encode("utf-8")

    def fake_urlopen(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("closed")
        return Answer()

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    from legal_stage_class import jev_budget

    monkeypatch.setattr(jev_budget, "BUDGET_TOKENS", 2 * jev_budget.REQUEST_TOKEN_LIMIT)
    laws = [{"law_id": "a", "canonical_law_name": "劳动法"}]
    ledger = providers.ProviderLedger()
    jev = providers.Jev(ledger, run_id="pre-test", query_id="11")
    jev.choose_files("q", laws)
    assert ledger.entries[0]["cost_usd"] == pytest.approx(123 * 0.042 / 1_000_000)
    assert ledger.entries[0]["query_id"] == "11"
    with pytest.raises(RuntimeError, match="budget"):
        providers.Jev(providers.ProviderLedger(), run_id="post-test", query_id="12").choose_files("q", laws)
    assert len(calls) == 2
    with sqlite3.connect("artifacts/jev-budget.sqlite3") as db:
        rows = db.execute("SELECT input_tokens, request_id FROM jev_requests ORDER BY id").fetchall()
    assert rows == [(None, None), (123, "request-1")]
