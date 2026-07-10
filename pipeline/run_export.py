"""Entry point: fetch → compute → payload → KV. `--dry-run` prints and skips upload."""
import argparse, json, os, sys
from pipeline.prices import fetch_closes
from pipeline.payload import build_etf_drawdowns_payload, build_etf_drawdowns_series
from pipeline.upload_kv import put_kv

ETFS = ["ARKK", "ARKQ", "ARKW", "ARKG", "ARKF", "ARKX"]
KEY = "research:risk:etf-drawdowns"
KEY_DATA = "research:risk:etf-drawdowns:data"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing env {name} — set it as a GitHub Actions secret")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    closes = fetch_closes(ETFS)
    as_of = max(s.index.max() for s in closes.values()).strftime("%Y-%m-%d")
    payload = build_etf_drawdowns_payload(closes, as_of=as_of)
    body = json.dumps(payload, separators=(",", ":"))
    series_payload = build_etf_drawdowns_series(closes, as_of=as_of)
    series_body = json.dumps(series_payload, separators=(",", ":"))
    print(f"as_of={as_of} bytes={len(body)} series_bytes={len(series_body)}")

    if args.dry_run:
        print(json.dumps(payload["table"], indent=2))
        return 0

    account_id = _require_env("CF_ACCOUNT_ID")
    api_token = _require_env("CF_API_TOKEN")
    # Series data first, then the page payload -- the page key is the one the
    # consumer reads, so it must land last to avoid a 404 window on the
    # xlsx-download button.
    put_kv(f"{KEY_DATA}:{as_of}", series_body, account_id=account_id, api_token=api_token)
    put_kv(f"{KEY_DATA}:latest", series_body, account_id=account_id, api_token=api_token)
    put_kv(f"{KEY}:{as_of}", body, account_id=account_id, api_token=api_token)
    put_kv(f"{KEY}:latest", body, account_id=account_id, api_token=api_token)
    print("uploaded: 4 keys (page+data, dated+latest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
