"""PUT values into the ark-movers-dashboard KV namespace via Cloudflare REST API."""
import time
import urllib.parse

import requests

NAMESPACE_ID = "89d5d9ac9b3141ccb40692cec108641e"
_RETRYABLE_EXCS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
_MAX_ATTEMPTS = 2  # one retry, only for transient failures (5xx / connection)
_RETRY_SLEEP_S = 2


class KvError(RuntimeError):
    pass


def put_kv(key: str, value: str, *, account_id: str, api_token: str,
           http_put=requests.put, sleep=time.sleep):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{NAMESPACE_ID}/values/{urllib.parse.quote(key, safe='')}"
    )
    last_exc = None
    detail = "no attempt made"
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = http_put(url, data=value.encode("utf-8"),
                            headers={"Authorization": f"Bearer {api_token}",
                                     "Content-Type": "application/json"},
                            timeout=30)
        except _RETRYABLE_EXCS as e:
            last_exc = e
            detail = f"connection error: {e}"
        else:
            if 200 <= resp.status_code < 300:
                return
            detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code < 500:  # 4xx (auth, bad request...) won't heal
                raise KvError(f"KV PUT {key} -> {detail}")
        if attempt < _MAX_ATTEMPTS - 1:
            sleep(_RETRY_SLEEP_S)
    raise KvError(f"KV PUT {key} -> {detail}") from last_exc
