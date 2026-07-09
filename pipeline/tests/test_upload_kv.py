import pytest
import requests

from pipeline.upload_kv import put_kv, KvError

NS = "89d5d9ac9b3141ccb40692cec108641e"


class FakeResp:
    def __init__(self, status, body='{"success":true}'):
        self.status_code = status
        self.text = body


def test_put_kv_builds_correct_url_and_auth():
    seen = {}

    def fake_put(url, data=None, headers=None, timeout=None):
        seen.update(url=url, headers=headers, data=data)
        return FakeResp(200)

    put_kv("research:risk:etf-drawdowns:latest", '{"a":1}',
           account_id="acct123", api_token="tok", http_put=fake_put)
    assert seen["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/acct123/storage/kv/"
        f"namespaces/{NS}/values/research%3Arisk%3Aetf-drawdowns%3Alatest"
    )
    assert seen["headers"]["Authorization"] == "Bearer tok"


def test_put_kv_retries_5xx_then_succeeds():
    responses = [FakeResp(500, "flaky"), FakeResp(200)]
    calls = []
    slept = []

    def fake_put(url, **kwargs):
        calls.append(url)
        return responses[len(calls) - 1]

    put_kv("k", "v", account_id="a", api_token="t",
           http_put=fake_put, sleep=slept.append)
    assert len(calls) == 2
    assert slept == [2]


def test_put_kv_4xx_raises_immediately_without_retry():
    calls = []
    slept = []

    def fake_put(url, **kwargs):
        calls.append(url)
        return FakeResp(401, "unauthorized")

    with pytest.raises(KvError, match="401"):
        put_kv("k", "v", account_id="a", api_token="t",
               http_put=fake_put, sleep=slept.append)
    assert len(calls) == 1
    assert slept == []


def test_put_kv_persistent_5xx_raises_after_single_retry():
    calls = []

    def fake_put(url, **kwargs):
        calls.append(url)
        return FakeResp(500, "boom")

    with pytest.raises(KvError, match="boom"):
        put_kv("k", "v", account_id="a", api_token="t",
               http_put=fake_put, sleep=lambda s: None)
    assert len(calls) == 2


def test_put_kv_retries_connection_error_then_succeeds():
    calls = []

    def fake_put(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("reset by peer")
        return FakeResp(204)  # any 2xx counts as success

    put_kv("k", "v", account_id="a", api_token="t",
           http_put=fake_put, sleep=lambda s: None)
    assert len(calls) == 2
