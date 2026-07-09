"""PUT values into the ark-movers-dashboard KV namespace via Cloudflare REST API."""
import urllib.parse
import requests

NAMESPACE_ID = "89d5d9ac9b3141ccb40692cec108641e"


class KvError(RuntimeError):
    pass


def put_kv(key: str, value: str, *, account_id: str, api_token: str, http_put=requests.put):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{NAMESPACE_ID}/values/{urllib.parse.quote(key, safe='')}"
    )
    resp = http_put(url, data=value.encode("utf-8"),
                    headers={"Authorization": f"Bearer {api_token}",
                             "Content-Type": "application/json"},
                    timeout=30)
    if resp.status_code != 200:
        raise KvError(f"KV PUT {key} -> HTTP {resp.status_code}: {resp.text[:200]}")
