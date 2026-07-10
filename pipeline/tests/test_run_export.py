import sys

import pandas as pd
import pytest

import pipeline.run_export as run_export


def _fake_closes(etfs):
    idx = pd.to_datetime(["2026-07-08", "2026-07-09"])
    return {"ARKK": pd.Series([100.0, 90.0], index=idx)}


def _fake_payload(closes, as_of):
    return {"table": [{"etf": "ARKK", "max_dd_pct": -10.0}], "as_of": as_of}


def _wire(monkeypatch, argv, put_spy):
    monkeypatch.setattr(run_export, "fetch_closes", _fake_closes)
    monkeypatch.setattr(run_export, "build_etf_drawdowns_payload", _fake_payload)
    monkeypatch.setattr(run_export, "put_kv", put_spy)
    monkeypatch.setattr(sys, "argv", ["run_export", *argv])


def test_dry_run_makes_no_kv_calls(monkeypatch, capsys):
    calls = []
    _wire(monkeypatch, ["--dry-run"], lambda *a, **k: calls.append((a, k)))
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)  # must not be needed
    monkeypatch.delenv("CF_API_TOKEN", raising=False)

    assert run_export.main() == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "as_of=2026-07-09" in out


def test_normal_path_uploads_dated_key_then_latest(monkeypatch):
    calls = []

    def put_spy(key, value, *, account_id, api_token):
        calls.append({"key": key, "value": value,
                      "account_id": account_id, "api_token": api_token})

    _wire(monkeypatch, [], put_spy)
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_API_TOKEN", "tok")

    assert run_export.main() == 0
    assert len(calls) == 4
    # Series data uploaded first (dated, then latest), so the xlsx-download
    # button never 404s in the window between the page payload landing and
    # the series data being available.
    assert calls[0]["key"] == "research:risk:etf-drawdowns:data:2026-07-09"
    assert calls[1]["key"] == "research:risk:etf-drawdowns:data:latest"
    # Page payload uploaded last -- it's the key the consumer reads.
    assert calls[2]["key"] == "research:risk:etf-drawdowns:2026-07-09"
    assert calls[3]["key"] == "research:risk:etf-drawdowns:latest"
    assert calls[0]["value"] == calls[1]["value"]  # same body for both data keys
    assert calls[2]["value"] == calls[3]["value"]  # same body for both page keys
    assert all(c["account_id"] == "acct" and c["api_token"] == "tok" for c in calls)


@pytest.mark.parametrize("missing", ["CF_ACCOUNT_ID", "CF_API_TOKEN"])
def test_missing_env_var_exits_with_readable_error(monkeypatch, missing):
    calls = []
    _wire(monkeypatch, [], lambda *a, **k: calls.append(a))
    for name in ("CF_ACCOUNT_ID", "CF_API_TOKEN"):
        if name == missing:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "x")

    with pytest.raises(SystemExit) as excinfo:
        run_export.main()
    assert missing in str(excinfo.value.code)
    assert "secret" in str(excinfo.value.code)
    assert calls == []  # nothing uploaded on config error
