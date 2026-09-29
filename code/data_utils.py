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
    PRICES_CACHE, SHARES_CACHE, FACTORS_CACHE,
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


def download_factors(force=False):
    """
    Download daily Fama/French 5 factors + momentum from the Ken French Data
    Library (via pandas-datareader) for the full sample window. Caches to
    CSV; set force=True to re-download even if a cache already exists.

    Source files (Ken French Data Library):
      "Fama/French 5 Factors (2x3) [Daily]" -> Mkt-RF, SMB, HML, RMW, CMA, RF
      "Momentum Factor (Mom) [Daily]"       -> Mom

    Ken French's files report returns in PERCENT (e.g. 0.05 means 0.05%),
    so everything is divided by 100 here to match the decimal stock returns
    used everywhere else in this project.
    """
    if FACTORS_CACHE.exists() and not force:
        return pd.read_csv(FACTORS_CACHE, index_col=0, parse_dates=True)

    import pandas_datareader.data as web

    print("Downloading Fama/French 5 factors + momentum (daily) from the "
          "Ken French Data Library...")

    ff5 = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
        start=START_DATE, end=END_DATE,
    )[0]
    mom = web.DataReader(
        "F-F_Momentum_Factor_daily", "famafrench",
        start=START_DATE, end=END_DATE,
    )[0]

    # pandas-datareader labels daily Ken French series with plain dates, but
    # normalize defensively in case a PeriodIndex slips through.
    ff5.index = pd.to_datetime(ff5.index.astype(str))
    mom.index = pd.to_datetime(mom.index.astype(str))

    ff5.columns = [c.strip() for c in ff5.columns]
    mom.columns = [c.strip() for c in mom.columns]

    factors = ff5.join(mom, how="inner") / 100.0   # percent -> decimal
    factors.to_csv(FACTORS_CACHE)
    print(f"Saved factor cache to {FACTORS_CACHE}")
    return factors


def compute_returns(prices):
    """
    Simple daily percentage returns from a price DataFrame. This applies to
    every column alike -- stocks, the S&P 500 benchmark, and BIL (the
    risk-free proxy). BIL's daily return already reflects its distributed
    interest income, since auto_adjust=True was used when downloading.
    """
    return prices.pct_change().dropna(how="all")