import pytest
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


def test_put_kv_raises_on_http_error():
    with pytest.raises(KvError):
        put_kv("k", "v", account_id="a", api_token="t",
               http_put=lambda *a, **k: FakeResp(500, "boom"))
