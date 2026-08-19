"""
data_utils.py
Download, cache, and load price data for the project universe.

All functions check for a local CSV cache before hitting yfinance, so the
full 10-year, 32-ticker download only happens once. Every later class
(diversification, factor models, momentum, CAPM...) can reuse the same
cached sp500_daily_2016_2025.csv instead of re-downloading.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    TICKERS, START_DATE, END_DATE,
    BENCHMARK_TICKER, RISK_FREE_TICKER,
    PRICES_CACHE, SHARES_CACHE,
)


def download_prices(force=False):
    """
    Download adjusted close prices for the 30 stocks + benchmark + risk-free
    proxy, for the full sample window. Caches to CSV; set force=True to
    re-download even if a cache already exists.
    """
    if PRICES_CACHE.exists() and not force:
        return pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)

    all_tickers = TICKERS + [BENCHMARK_TICKER, RISK_FREE_TICKER]
    print(f"Downloading {len(all_tickers)} tickers from {START_DATE} to {END_DATE}...")

    raw = yf.download(
        all_tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # adjusted close: dividends/splits already baked in
        progress=False,
    )

    # With multiple tickers, yfinance returns MultiIndex columns
    # (PriceType, Ticker). "Close" here IS the adjusted close, since
    # auto_adjust=True.
    prices = raw["Close"].copy()

    dead = prices.columns[prices.isna().all()].tolist()
    if dead:
        print(f"Warning: no data returned for {dead} -- check these tickers.")

    prices.to_csv(PRICES_CACHE)
    print(f"Saved price cache to {PRICES_CACHE}")
    return prices


def download_shares_outstanding(force=False):
    """
    Fetch current shares outstanding for each stock. Used only to build an
    approximate INITIAL value-weighted allocation at the start of the
    sample (Jan 2016).

    Simplification: yfinance does not expose a clean historical shares-
    outstanding series, so we use today's figure as a stand-in for 2016.
    Shares outstanding move far more slowly than prices, so this is a
    reasonable approximation for a starting allocation -- flag it as an
    assumption in your writeup.
    """
    if SHARES_CACHE.exists() and not force:
        return pd.read_csv(SHARES_CACHE, index_col=0).squeeze("columns")

    shares = {}
    for t in TICKERS:
        info = yf.Ticker(t).get_info()
        shares[t] = info.get("sharesOutstanding", np.nan)

    shares = pd.Series(shares, name="shares_outstanding")
    shares.to_csv(SHARES_CACHE)
    print(f"Saved shares-outstanding cache to {SHARES_CACHE}")
    return shares


def compute_returns(prices):
    """
    Simple daily percentage returns from a price DataFrame. This applies to
    every column alike -- stocks, the S&P 500 benchmark, and BIL (the
    risk-free proxy). BIL's daily return already reflects its distributed
    interest income, since auto_adjust=True was used when downloading.
    """
    return prices.pct_change().dropna(how="all")