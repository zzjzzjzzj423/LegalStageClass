import http.client
import json

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
