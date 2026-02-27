from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class ReturnsData:
    prices: pd.DataFrame
    returns: pd.DataFrame


def fetch_prices(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    df = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError("No data returned")

    # keep Adj Close
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Adj Close"].copy()
    else:
        # single ticker case
        px = df[["Adj Close"]].copy()
        px.columns = tickers

    px = px.dropna().sort_index()
    return px


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def load_returns(tickers: List[str], start: str, end: Optional[str] = None) -> ReturnsData:
    prices = fetch_prices(tickers, start, end)
    rets = to_returns(prices)
    return ReturnsData(prices=prices, returns=rets)
