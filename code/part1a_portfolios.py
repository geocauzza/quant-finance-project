"""
part1a_portfolios.py
Project 1, Part A.

Builds value-weighted (VW) and equal-weighted (EW) portfolios from the
30-stock universe, rebalanced monthly on the first trading day of each
month using only information through the PREVIOUS day's close (1-day
lag -- no look-ahead bias).

Outputs:
- output/tables/part1a_summary_stats.csv
- output/tables/part1a_turnover.csv
- output/figures/part1a_cumulative_returns.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TICKERS, BENCHMARK_TICKER, RISK_FREE_TICKER,
    OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR,
)
from data_utils import (
    download_prices, download_shares_outstanding,
    compute_returns,
)


def build_initial_vw_weights(prices, shares):
    """Approximate 2016 market-cap weights: shares outstanding x first price."""
    p0 = prices.iloc[0]
    mcap0 = shares.reindex(TICKERS) * p0[TICKERS]
    return mcap0 / mcap0.sum()


def simulate_portfolio(returns, initial_weights, rebalance_to_target):
    """
    Walk forward day by day.

    On the first trading day of each new month: compute turnover (drifted
    "pre-trade" weights vs. the new "target" weights), then reset to
    target. On every day: let weights drift with that day's returns.

    rebalance_to_target(pre_trade_weights) -> target_weights
        EW: always returns 1/N (forces a trade back to equal weights).
        VW: returns pre_trade_weights unchanged (no trade needed --
            value weights are already "correct" after prices move).
    """
    dates = returns.index
    weights = initial_weights.copy()

    port_returns = pd.Series(index=dates, dtype=float)
    turnover = {}

    current_month = None
    for t in dates:
        if current_month is None or t.month != current_month:
            # Rebalance day. "pre_trade" is where drift left us using only
            # information through yesterday's close -- no same-day data
            # is used in this decision, per the 1-day-lag rule.
            pre_trade = weights.copy()
            target = rebalance_to_target(pre_trade)
            turnover[t] = 0.5 * (target - pre_trade).abs().sum()
            weights = target
            current_month = t.month

        day_returns = returns.loc[t, weights.index]
        port_returns[t] = (weights * day_returns).sum()

        # Drift weights forward using today's returns (no trading)
        grown = weights * (1 + day_returns)
        weights = grown / grown.sum()

    return port_returns, pd.Series(turnover, name="turnover")


def annualize_return(daily_returns, periods=252):
    return (1 + daily_returns.mean()) ** periods - 1


def annualize_vol(daily_returns, periods=252):
    return daily_returns.std(ddof=1) * np.sqrt(periods)


def sharpe_ratio(daily_returns, daily_rf, periods=252):
    excess = daily_returns - daily_rf
    return excess.mean() / daily_returns.std(ddof=1) * np.sqrt(periods)


def information_ratio(port_returns, bench_returns, periods=252):
    active = port_returns - bench_returns
    return active.mean() / active.std(ddof=1) * np.sqrt(periods)


def main():
    prices = download_prices()
    shares = download_shares_outstanding()

    returns = compute_returns(prices)
    stock_returns = returns[TICKERS]
    bench_returns = returns[BENCHMARK_TICKER]
    rf_daily = returns[RISK_FREE_TICKER]   # BIL's own daily return = risk-free proxy

    # --- Equal-weighted ---
    ew_initial = pd.Series(1 / len(TICKERS), index=TICKERS)
    ew_returns, ew_turnover = simulate_portfolio(
        stock_returns, ew_initial,
        rebalance_to_target=lambda pre: pd.Series(1 / len(TICKERS), index=pre.index),
    )

    # --- Value-weighted ---
    vw_initial = build_initial_vw_weights(prices, shares)
    vw_returns, vw_turnover = simulate_portfolio(
        stock_returns, vw_initial,
        rebalance_to_target=lambda pre: pre,   # self-rebalancing: no trade
    )

    # --- Summary statistics ---
    summary = pd.DataFrame({
        "EW": {
            "Annualized Return": annualize_return(ew_returns),
            "Annualized Volatility": annualize_vol(ew_returns),
            "Sharpe Ratio": sharpe_ratio(ew_returns, rf_daily),
            "Information Ratio vs S&P 500": information_ratio(ew_returns, bench_returns),
            "Avg Monthly Turnover": ew_turnover.mean(),
        },
        "VW": {
            "Annualized Return": annualize_return(vw_returns),
            "Annualized Volatility": annualize_vol(vw_returns),
            "Sharpe Ratio": sharpe_ratio(vw_returns, rf_daily),
            "Information Ratio vs S&P 500": information_ratio(vw_returns, bench_returns),
            "Avg Monthly Turnover": vw_turnover.mean(),
        },
    })
    print(summary)
    summary.to_csv(OUTPUT_TABLES_DIR / "part1a_summary_stats.csv")

    turnover_compare = pd.DataFrame({"EW": ew_turnover, "VW": vw_turnover})
    turnover_compare.to_csv(OUTPUT_TABLES_DIR / "part1a_turnover.csv")
    higher = "EW" if ew_turnover.mean() > vw_turnover.mean() else "VW"
    print(f"\nHigher average monthly turnover: {higher}")

    # --- Cumulative return plot ---
    cum_ew = (1 + ew_returns).cumprod()
    cum_vw = (1 + vw_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()
    cum_rf = (1 + rf_daily).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cum_ew, label="Equal-Weighted")
    plt.plot(cum_vw, label="Value-Weighted")
    plt.plot(cum_bench, label="S&P 500")
    plt.plot(cum_rf, label="Risk-Free")
    plt.title("Cumulative Returns: EW vs VW vs S&P 500 vs Risk-Free (2016-2025)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "part1a_cumulative_returns.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()