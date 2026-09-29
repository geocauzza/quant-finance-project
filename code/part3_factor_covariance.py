"""
part3_factor_covariance.py
Project 3: Estimating Covariance Matrices with Factors.

Uses the same 30-stock, 2016-2025 panel as Projects 1-2, plus daily factor
returns from the Ken French Data Library (Fama/French 5 factors + momentum).

Tasks covered (core; the two "extra/optional" items -- 5 industry factors,
and a momentum-based tactical version -- are intentionally left out, as the
assignment allows them to be submitted later):

1. Build a factor-based covariance matrix for the 30 stocks from a
   multi-factor regression on Mkt-RF, SMB, HML, RMW, CMA, and Mom.
2. Full sample: compare the MVP under the sample covariance matrix vs. the
   factor-based covariance matrix -- max weight, effective number of
   positions (1 / sum(w_i^2)), and realized volatility.
3. Out-of-sample, monthly rebalanced: at each month's first trading day,
   estimate BOTH covariance matrices on an expanding window of data up to
   (and not including) that day, compute each MVP, and hold for one month
   (weights then drift with prices, same 1-day-lag convention as Project 1).
   This is where the sample matrix's larger parameter count (465 free
   parameters vs. the factor model's ~200) should start to cost it.

Outputs:
- output/tables/part3_full_sample_comparison.csv
- output/tables/part3_oos_comparison.csv
- output/tables/part3_oos_concentration_history.csv
- output/figures/part3_effective_n_oos.png
- output/figures/part3_oos_growth.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TICKERS, FACTOR_NAMES, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR
from data_utils import download_prices, download_factors, compute_returns

TRADING_DAYS = 252

# Require at least 2 years of daily history before the first out-of-sample
# rebalance -- with 30 assets you need a reasonably long window before the
# sample covariance matrix (465 free parameters) is estimated on more
# observations than parameters.
BURN_IN_DAYS = 2 * TRADING_DAYS


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------
def sample_covariance(stock_returns):
    """Plain historical covariance matrix, annualized."""
    return stock_returns.cov().values * TRADING_DAYS


def factor_covariance(stock_returns, factor_returns, factor_names=FACTOR_NAMES):
    """
    Multi-factor covariance matrix: Sigma = B * Sigma_F * B' + D, where
      B       = (N x K) factor betas from OLS of each stock on the factors
      Sigma_F = (K x K) factor covariance matrix
      D       = (N x N) diagonal matrix of idiosyncratic (residual) variances

    This is the direct multi-factor generalization of the single-index
    model: instead of one market beta, each stock gets a full vector of
    factor exposures, and everything the factors don't explain is treated
    as diagonal (uncorrelated) idiosyncratic risk.
    """
    common = stock_returns.index.intersection(factor_returns.index)
    Y = stock_returns.loc[common].values                       # T x N
    F = factor_returns.loc[common, factor_names].values         # T x K
    X = np.column_stack([np.ones(len(common)), F])              # T x (K+1)

    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)               # (K+1) x N
    B = coeffs[1:, :].T                                          # N x K

    fitted = X @ coeffs
    resid = Y - fitted
    dof = max(X.shape[0] - X.shape[1], 1)
    resid_var = (resid ** 2).sum(axis=0) / dof                  # N,

    factor_cov = np.cov(F, rowvar=False)                        # K x K (daily)
    sigma_daily = B @ factor_cov @ B.T + np.diag(resid_var)
    return sigma_daily * TRADING_DAYS


# ---------------------------------------------------------------------------
# MVP and concentration metrics
# ---------------------------------------------------------------------------
def mvp_weights(cov):
    """Minimum-variance portfolio weights (unconstrained, can short)."""
    ones = np.ones(cov.shape[0])
    raw = np.linalg.solve(cov, ones)
    return raw / raw.sum()


def effective_n(weights):
    """Effective number of positions: 1 / sum(w_i^2)."""
    return 1.0 / np.sum(np.asarray(weights) ** 2)


def annualize_return(daily_returns, periods=TRADING_DAYS):
    return (1 + daily_returns.mean()) ** periods - 1


def annualize_vol(daily_returns, periods=TRADING_DAYS):
    return daily_returns.std(ddof=1) * np.sqrt(periods)


def full_sample_row(weights, stock_returns, label):
    port_returns = stock_returns @ weights
    w = pd.Series(weights, index=stock_returns.columns)
    return {
        "Estimator": label,
        "Annualized Return": annualize_return(port_returns),
        "Annualized Volatility": annualize_vol(port_returns),
        "Max Weight": w.max(),
        "Max |Weight|": w.abs().max(),
        "Effective N": effective_n(weights),
    }


# ---------------------------------------------------------------------------
# Out-of-sample, monthly-rebalanced backtest
# ---------------------------------------------------------------------------
def run_oos_backtest(stock_returns, factor_returns, estimator_fn, burn_in_days=BURN_IN_DAYS):
    """
    Expanding-window, monthly-rebalanced MVP backtest.

    On the first trading day of each month (once at least burn_in_days of
    history exist), the covariance matrix is estimated using ONLY data
    strictly before that day (no look-ahead), the MVP weights are computed,
    and the portfolio is held -- drifting with prices, no intramonth trading
    -- until the next rebalance. Same 1-day-lag convention as Project 1/2.

    Returns:
      port_returns  -- realized daily return series of the OOS portfolio
      history       -- DataFrame indexed by rebalance date with the max
                        weight, max |weight|, and effective N chosen that month
    """
    dates = stock_returns.index
    n = len(dates)
    tickers = stock_returns.columns

    port_returns = pd.Series(index=dates, dtype=float)
    weights = None
    current_period = None
    history_rows = []

    for i in range(n):
        t = dates[i]
        if i < burn_in_days:
            continue

        period = (t.year, t.month)
        if period != current_period:
            hist_returns = stock_returns.iloc[:i]   # strictly before t: no look-ahead
            cov = estimator_fn(hist_returns, factor_returns)
            w = mvp_weights(cov)
            weights = pd.Series(w, index=tickers)
            current_period = period
            history_rows.append({
                "date": t,
                "max_weight": weights.max(),
                "max_abs_weight": weights.abs().max(),
                "effective_n": effective_n(weights.values),
            })

        day_returns = stock_returns.loc[t]
        port_returns[t] = (weights * day_returns).sum()

        grown = weights * (1 + day_returns)
        weights = grown / grown.sum()

    port_returns = port_returns.dropna()
    history = pd.DataFrame(history_rows).set_index("date")
    return port_returns, history


def main():
    prices = download_prices()          # cached from Project 1
    returns = compute_returns(prices)
    stock_returns = returns[TICKERS]

    factors = download_factors()
    common = stock_returns.index.intersection(factors.index)
    stock_returns = stock_returns.loc[common]
    factors = factors.loc[common]
    print(f"Aligned {len(common)} trading days of stock and factor data "
          f"({common.min().date()} to {common.max().date()}).")

    n_params_sample = len(TICKERS) * (len(TICKERS) + 1) // 2
    n_params_factor = len(TICKERS) * len(FACTOR_NAMES) + len(FACTOR_NAMES) * (len(FACTOR_NAMES) + 1) // 2 + len(TICKERS)
    print(f"Sample covariance free parameters: {n_params_sample}")
    print(f"Factor-model free parameters (betas + factor moments + residual variances): ~{n_params_factor}")

    # -----------------------------------------------------------------
    # Q3: full-sample comparison
    # -----------------------------------------------------------------
    cov_sample_full = sample_covariance(stock_returns)
    cov_factor_full = factor_covariance(stock_returns, factors)

    w_sample_full = mvp_weights(cov_sample_full)
    w_factor_full = mvp_weights(cov_factor_full)

    full_sample = pd.DataFrame([
        full_sample_row(w_sample_full, stock_returns, "Sample Covariance"),
        full_sample_row(w_factor_full, stock_returns, "Factor-Based Covariance"),
    ]).set_index("Estimator")
    print("\nFull-sample MVP comparison:\n", full_sample)
    full_sample.to_csv(OUTPUT_TABLES_DIR / "part3_full_sample_comparison.csv")

    less_concentrated = (
        "factor-based" if full_sample.loc["Factor-Based Covariance", "Effective N"]
        > full_sample.loc["Sample Covariance", "Effective N"] else "sample-covariance"
    )
    print(f"\nLess concentrated full-sample MVP: {less_concentrated}")

    # -----------------------------------------------------------------
    # Q4: out-of-sample, monthly-rebalanced comparison
    # -----------------------------------------------------------------
    print("\nRunning out-of-sample monthly-rebalanced backtest "
          f"(burn-in: {BURN_IN_DAYS} trading days)...")

    oos_sample_returns, oos_sample_history = run_oos_backtest(
        stock_returns, factors, lambda r, f: sample_covariance(r)
    )
    oos_factor_returns, oos_factor_history = run_oos_backtest(
        stock_returns, factors, lambda r, f: factor_covariance(r, f)
    )

    oos_comparison = pd.DataFrame({
        "Sample Covariance": {
            "Realized Annualized Return": annualize_return(oos_sample_returns),
            "Realized Annualized Volatility": annualize_vol(oos_sample_returns),
            "Avg. Max |Weight|": oos_sample_history["max_abs_weight"].mean(),
            "Avg. Effective N": oos_sample_history["effective_n"].mean(),
        },
        "Factor-Based Covariance": {
            "Realized Annualized Return": annualize_return(oos_factor_returns),
            "Realized Annualized Volatility": annualize_vol(oos_factor_returns),
            "Avg. Max |Weight|": oos_factor_history["max_abs_weight"].mean(),
            "Avg. Effective N": oos_factor_history["effective_n"].mean(),
        },
    })
    print("\nOut-of-sample MVP comparison:\n", oos_comparison)
    oos_comparison.to_csv(OUTPUT_TABLES_DIR / "part3_oos_comparison.csv")

    concentration_history = oos_sample_history.join(
        oos_factor_history, lsuffix="_sample", rsuffix="_factor"
    )
    concentration_history.to_csv(OUTPUT_TABLES_DIR / "part3_oos_concentration_history.csv")

    vol_gap = (
        oos_comparison.loc["Realized Annualized Volatility", "Sample Covariance"]
        - oos_comparison.loc["Realized Annualized Volatility", "Factor-Based Covariance"]
    )
    print(f"\nOOS realized-volatility gap (sample - factor): {vol_gap:+.2%}")

    # --- Plot: effective N over time, sample vs. factor ---
    plt.figure(figsize=(10, 6))
    plt.plot(oos_sample_history.index, oos_sample_history["effective_n"],
              label="Sample covariance MVP")
    plt.plot(oos_factor_history.index, oos_factor_history["effective_n"],
              label="Factor-based MVP")
    plt.axhline(len(TICKERS), color="gray", linestyle=":", label=f"Fully diversified ({len(TICKERS)})")
    plt.title("Out-of-Sample MVP Concentration Over Time")
    plt.xlabel("Rebalance date")
    plt.ylabel("Effective number of positions (1 / Σ w_i²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "part3_effective_n_oos.png", dpi=150)
    plt.show()

    # --- Plot: OOS growth of $1, sample vs. factor ---
    cum_sample = (1 + oos_sample_returns).cumprod()
    cum_factor = (1 + oos_factor_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cum_sample, label="Sample covariance MVP (OOS)")
    plt.plot(cum_factor, label="Factor-based MVP (OOS)")
    plt.title("Out-of-Sample MVP Growth of $1, Monthly Rebalanced")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "part3_oos_growth.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
