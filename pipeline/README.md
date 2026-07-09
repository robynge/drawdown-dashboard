# KV Export Pipeline

Computes drawdown stats (summary table + inline SVG chart/heatmap) for the six
ARK ETFs and publishes them to the ark-movers-dashboard Cloudflare KV namespace
as `research:risk:etf-drawdowns:latest` plus a dated copy
(`research:risk:etf-drawdowns:<YYYY-MM-DD>`). A weekday GitHub Actions cron
(`.github/workflows/export-kv.yml`, 22:30 UTC) runs the test suite first, so a
failing pipeline never overwrites the previous KV values.

## Local dry run

```
python -m pipeline.run_export --dry-run
```

Fetches real prices, prints `as_of`, payload size, and the summary table, and
skips the upload — no secrets required.

## Required repo secrets

- `CF_ACCOUNT_ID` — Cloudflare account id
- `CF_API_TOKEN` — API token with write access to KV namespace
  `89d5d9ac9b3141ccb40692cec108641e`

## Alerting caveats

GitHub emails scheduled-workflow failures only to the last committer of the
workflow file, so nobody else is notified when the cron fails. GitHub also
auto-disables scheduled workflows in repos with no recent activity, so check
the Actions page occasionally to confirm the cron is still enabled and green.
