"""
part2_frontier.py
Project 2: The Mean-Variance Frontier in Practice.

Uses the same 30-stock, 2016-2025 panel built for Project 1 (cached in
data/raw/, see data_utils.download_prices).

Tasks:
- Tangency portfolio: risk premium per asset = past 10-year average excess
  return (mean daily excess return, annualized) -- a bad but easy assumption,
  since it uses realized average return as a proxy for expected return with
  essentially no shrinkage, so the estimate is extremely noisy asset by
  asset (flagged again below, next to the weights it produces).
- Minimum-Variance Portfolio (MVP).
- Performance stats for both (monthly rebalanced back to the fixed
  in-sample-optimal weights): annualized return/vol, Sharpe, IR vs S&P 500,
  plus the weights themselves (largest/smallest positions).
- The 30-asset mean-variance frontier (closed form, unconstrained -- shorting
  allowed, which is the point: it lets the weights get as extreme as the
  noisy inputs push them), with the MVP, tangency portfolio, and the capital
  allocation line.
- The two-asset frontier, varying the correlation between two of the 30
  stocks from -1 to 1.

Outputs:
- output/tables/part2_summary_stats.csv
- output/tables/part2_weights.csv
- output/figures/part2_frontier.png
- output/figures/part2_two_asset_frontier.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    TICKERS, BENCHMARK_TICKER, RISK_FREE_TICKER,
    OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR,
)
from data_utils import download_prices, compute_returns

TRADING_DAYS = 252

# Two names used for the two-asset correlation illustration: one tech, one
# health-care name, chosen for a low real-world correlation so the effect of
# varying rho is visually clear.
TWO_ASSET_PAIR = ("AAPL", "JNJ")
CORRELATIONS_TO_PLOT = (-1.0, -0.5, 0.0, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Performance-stat helpers (same definitions as Project 1, Part A)
# ---------------------------------------------------------------------------
def annualize_return(daily_returns, periods=TRADING_DAYS):
    return (1 + daily_returns.mean()) ** periods - 1


def annualize_vol(daily_returns, periods=TRADING_DAYS):
    return daily_returns.std(ddof=1) * np.sqrt(periods)


def sharpe_ratio(daily_returns, daily_rf, periods=TRADING_DAYS):
    excess = daily_returns - daily_rf
    return excess.mean() / daily_returns.std(ddof=1) * np.sqrt(periods)


def information_ratio(port_returns, bench_returns, periods=TRADING_DAYS):
    active = port_returns - bench_returns
    return active.mean() / active.std(ddof=1) * np.sqrt(periods)


# ---------------------------------------------------------------------------
# Markowitz closed-form frontier (Merton, 1972). Allows short sales, so
# weights can be negative or larger than 1 -- intentional, since the point
# of the assignment is to see how extreme the weights get with estimated
# inputs, not to impose a no-shorting constraint that would hide that.
# ---------------------------------------------------------------------------
def markowitz_constants(mu, sigma_inv, ones):
    A = ones @ sigma_inv @ ones
    B = ones @ sigma_inv @ mu
    C = mu @ sigma_inv @ mu
    D = A * C - B ** 2
    return A, B, C, D


def mvp_weights(sigma_inv, ones, A):
    return (sigma_inv @ ones) / A


def tangency_weights(sigma_inv, mu, rf, ones):
    excess = mu - rf * ones
    raw = sigma_inv @ excess
    return raw / raw.sum()


def frontier_variance(m, A, B, C, D):
    """Variance of the minimum-variance portfolio achieving target return m."""
    return (A * m ** 2 - 2 * B * m + C) / D


def fixed_weight_monthly_rebalanced_returns(stock_returns, weights):
    """
    Daily return series for a portfolio held at a FIXED target weight
    vector, drifting with prices intramonth and reset back to that target
    on the first trading day of each month (one-day-lag, no look-ahead --
    same convention as Project 1, Part A).
    """
    dates = stock_returns.index
    w = weights.copy()
    port_returns = pd.Series(index=dates, dtype=float)

    current_month = None
    for t in dates:
        if current_month is None or t.month != current_month:
            w = weights.copy()   # reset to the fixed in-sample-optimal target
            current_month = t.month

        day_returns = stock_returns.loc[t, w.index]
        port_returns[t] = (w * day_returns).sum()

        grown = w * (1 + day_returns)
        w = grown / grown.sum()

    return port_returns


def performance_row(port_returns, rf_daily, bench_returns, weights):
    sorted_w = weights.sort_values(ascending=False)
    return {
        "Annualized Return": annualize_return(port_returns),
        "Annualized Volatility": annualize_vol(port_returns),
        "Sharpe Ratio": sharpe_ratio(port_returns, rf_daily),
        "Information Ratio vs S&P 500": information_ratio(port_returns, bench_returns),
        "Largest Position": f"{sorted_w.index[0]} ({sorted_w.iloc[0]:.1%})",
        "Smallest Position": f"{sorted_w.index[-1]} ({sorted_w.iloc[-1]:.1%})",
    }


def main():
    prices = download_prices()   # cached from Project 1
    returns = compute_returns(prices)
    stock_returns = returns[TICKERS]
    bench_returns = returns[BENCHMARK_TICKER]
    rf_daily = returns[RISK_FREE_TICKER]

    # --- Inputs: annualized mean returns and covariance matrix ---
    mu = stock_returns.mean() * TRADING_DAYS
    cov = stock_returns.cov() * TRADING_DAYS
    sigma_inv = np.linalg.inv(cov.values)
    ones = np.ones(len(TICKERS))

    # Risk-free rate assumption: the sample-average annualized return on
    # BIL (the same proxy used in Project 1). Declared explicitly, as the
    # assignment asks.
    rf = annualize_return(rf_daily)
    print(f"Risk-free rate assumption (avg. annualized BIL return): {rf:.2%}")

    A, B, C, D = markowitz_constants(mu.values, sigma_inv, ones)

    # --- MVP and tangency portfolio weights ---
    w_mvp = pd.Series(mvp_weights(sigma_inv, ones, A), index=TICKERS)
    w_tan = pd.Series(tangency_weights(sigma_inv, mu.values, rf, ones), index=TICKERS)

    print("\nEach asset's risk premium is its own past 10-year average excess "
          "return -- a noisy, unshrunk estimate of expected return, which is "
          "exactly why the tangency weights below come out so extreme.")

    # --- Realized daily return series under monthly rebalancing ---
    mvp_returns = fixed_weight_monthly_rebalanced_returns(stock_returns, w_mvp)
    tan_returns = fixed_weight_monthly_rebalanced_returns(stock_returns, w_tan)

    summary = pd.DataFrame({
        "Tangency": performance_row(tan_returns, rf_daily, bench_returns, w_tan),
        "MVP": performance_row(mvp_returns, rf_daily, bench_returns, w_mvp),
    })
    print("\n", summary)
    summary.to_csv(OUTPUT_TABLES_DIR / "part2_summary_stats.csv")

    weights_table = pd.DataFrame({"Tangency": w_tan, "MVP": w_mvp})
    weights_table.to_csv(OUTPUT_TABLES_DIR / "part2_weights.csv")

    # --- 30-asset mean-variance frontier ---
    mvp_mean, mvp_vol = B / A, np.sqrt(1 / A)
    tan_mean = float(w_tan.values @ mu.values)
    tan_vol = float(np.sqrt(w_tan.values @ cov.values @ w_tan.values))

    m_grid = np.linspace(mvp_mean - 0.15, max(tan_mean, mu.max()) + 0.10, 300)
    vol_grid = np.sqrt(frontier_variance(m_grid, A, B, C, D))

    plt.figure(figsize=(10, 7))
    plt.plot(vol_grid * 100, m_grid * 100, label="Mean-variance frontier (30 assets)")
    plt.scatter([mvp_vol * 100], [mvp_mean * 100], color="black", zorder=5, label="MVP")
    plt.scatter([tan_vol * 100], [tan_mean * 100], color="red", zorder=5, label="Tangency portfolio")

    # Capital allocation line: from (0, rf) through the tangency portfolio,
    # extended past it.
    cal_vol = np.linspace(0, vol_grid.max(), 50)
    cal_mean = rf + (tan_mean - rf) / tan_vol * cal_vol
    plt.plot(cal_vol * 100, cal_mean * 100, linestyle="--", color="darkorange",
             label="Capital allocation line")

    plt.scatter(stock_returns.std() * np.sqrt(TRADING_DAYS) * 100, mu * 100,
                color="gray", alpha=0.5, s=15, label="Individual stocks")

    plt.title("Mean-Variance Frontier -- 30-Stock Universe (2016-2025 inputs)")
    plt.xlabel("Annualized volatility (%)")
    plt.ylabel("Annualized expected return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "part2_frontier.png", dpi=150)
    plt.show()

    # --- Two-asset frontier: effect of varying correlation ---
    a1, a2 = TWO_ASSET_PAIR
    mu1, mu2 = mu[a1], mu[a2]
    sig1 = stock_returns[a1].std() * np.sqrt(TRADING_DAYS)
    sig2 = stock_returns[a2].std() * np.sqrt(TRADING_DAYS)
    w1_grid = np.linspace(0, 1, 200)

    plt.figure(figsize=(10, 7))
    for rho in CORRELATIONS_TO_PLOT:
        port_mean = w1_grid * mu1 + (1 - w1_grid) * mu2
        port_var = (
            (w1_grid * sig1) ** 2
            + ((1 - w1_grid) * sig2) ** 2
            + 2 * w1_grid * (1 - w1_grid) * sig1 * sig2 * rho
        )
        plt.plot(np.sqrt(port_var) * 100, port_mean * 100, label=f"rho = {rho:+.1f}")

    plt.scatter([sig1 * 100], [mu1 * 100], color="black", zorder=5)
    plt.annotate(a1, (sig1 * 100, mu1 * 100), textcoords="offset points", xytext=(6, 4))
    plt.scatter([sig2 * 100], [mu2 * 100], color="black", zorder=5)
    plt.annotate(a2, (sig2 * 100, mu2 * 100), textcoords="offset points", xytext=(6, 4))

    plt.title(f"Two-Asset Frontier ({a1} & {a2}): Effect of Correlation")
    plt.xlabel("Annualized volatility (%)")
    plt.ylabel("Annualized expected return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "part2_two_asset_frontier.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
