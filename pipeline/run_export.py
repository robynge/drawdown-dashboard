"""Entry point: fetch → compute → payload → KV. `--dry-run` prints and skips upload."""
import argparse, json, os, sys
from pipeline.prices import fetch_closes
from pipeline.payload import build_etf_drawdowns_payload
from pipeline.upload_kv import put_kv

ETFS = ["ARKK", "ARKQ", "ARKW", "ARKG", "ARKF", "ARKX"]
KEY = "research:risk:etf-drawdowns"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    closes = fetch_closes(ETFS)
    as_of = max(s.index.max() for s in closes.values()).strftime("%Y-%m-%d")
    payload = build_etf_drawdowns_payload(closes, as_of=as_of)
    body = json.dumps(payload, separators=(",", ":"))
    print(f"as_of={as_of} bytes={len(body)}")

    if args.dry_run:
        print(json.dumps(payload["table"], indent=2))
        return 0

    account_id = os.environ["CF_ACCOUNT_ID"]
    api_token = os.environ["CF_API_TOKEN"]
    put_kv(f"{KEY}:{as_of}", body, account_id=account_id, api_token=api_token)
    put_kv(f"{KEY}:latest", body, account_id=account_id, api_token=api_token)
    print("uploaded: dated + latest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
