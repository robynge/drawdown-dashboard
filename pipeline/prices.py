"""Full-history daily closes via yfinance, with retry. Injectable for tests."""
import time
import pandas as pd


class FetchError(RuntimeError):
    pass


def _yf_download(ticker: str, **kw) -> pd.DataFrame:
    import yfinance as yf
    return yf.Ticker(ticker).history(period="max", auto_adjust=False)


def fetch_closes(etfs, downloader=_yf_download, retries=4, backoff_s=20):
    """Fetch full-history closes for each ticker in ``etfs``.

    Returns a dict of ticker -> Series (DatetimeIndex, tz-naive, float),
    satisfying ``summarize_drawdown``'s input contract: non-empty and free
    of trailing NaNs (guaranteed here via ``.dropna()``).
    """
    out = {}
    for etf in etfs:
        last_err = None
        for attempt in range(retries):
            try:
                df = downloader(etf)
                if df is None or df.empty or "Close" not in df:
                    raise FetchError(f"empty frame for {etf}")
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                out[etf] = s
                break
            except Exception as e:  # noqa: BLE001 — yfinance raises misc types
                last_err = e
                if attempt < retries - 1:
                    time.sleep(backoff_s * (attempt + 1))
        else:
            raise FetchError(f"{etf}: {last_err}")
    return out
