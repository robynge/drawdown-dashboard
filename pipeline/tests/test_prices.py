import pandas as pd
import pytest
from pipeline.prices import fetch_closes, FetchError

ETFS = ["ARKK", "ARKQ"]


def _fake_ok(ticker, **kw):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)


def test_fetch_closes_returns_per_etf_series():
    out = fetch_closes(ETFS, downloader=_fake_ok)
    assert set(out) == set(ETFS)
    assert list(out["ARKK"]) == [1.0, 2.0, 3.0]


def test_retries_then_raises():
    calls = {"n": 0}

    def flaky(ticker, **kw):
        calls["n"] += 1
        raise RuntimeError("rate limited")

    with pytest.raises(FetchError):
        fetch_closes(["ARKK"], downloader=flaky, retries=3, backoff_s=0)
    assert calls["n"] == 3


def test_empty_frame_is_error():
    def empty(ticker, **kw):
        return pd.DataFrame()

    with pytest.raises(FetchError):
        fetch_closes(["ARKK"], downloader=empty, retries=1, backoff_s=0)
