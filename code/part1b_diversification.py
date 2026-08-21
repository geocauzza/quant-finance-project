"""
part1b_diversification.py
Project 1, Part B: the diversification experiment.

For each portfolio size n = 1..30:
  - Randomly draw n stocks (without replacement) from the 30-stock universe,
    200 times.
  - For each draw, build the REALIZED equal-weighted portfolio (using the
    actual historical daily returns of the chosen stocks, not a simulated
    process) and record that portfolio's return variance over the full
    sample.
  - Average variance across the 200 draws -> the empirical diversification
    curve at that n.

Also computes:
  - The theoretical closed-form curve,
        Var(n) = (1/n) * avg_own_variance + ((n-1)/n) * avg_covariance,
    using this universe's OWN average variance and average pairwise
    covariance (recomputed from the real data, not the textbook's
    illustrative sigma=50%/rho=0.40 example).
  - A risk-reduction schedule: the marginal drop in annualized volatility
    (in percentage points) from adding the 2nd, 6th, 11th, 21st, and 30th
    stock.

Outputs:
- output/tables/part1b_diversification_curve.csv
- output/tables/part1b_risk_reduction_schedule.csv
- output/figures/part1b_diversification_curve.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TICKERS, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR
from data_utils import download_prices, compute_returns

N_TRIALS = 200
RANDOM_SEED = 42   # fixed seed -> reproducible curve across runs


def empirical_diversification_curve(stock_returns, n_trials=N_TRIALS, seed=RANDOM_SEED):
    """
    For each n = 1..len(TICKERS): draw n stocks at random (n_trials times),
    build the equal-weighted portfolio's actual daily return series, and
    record its variance. Returns the average variance/volatility across
    trials, for every n.
    """
    rng = np.random.default_rng(seed)
    tickers = stock_returns.columns.to_list()
    n_assets = len(tickers)

    avg_daily_var = np.zeros(n_assets)

    for n in range(1, n_assets + 1):
        trial_vars = np.empty(n_trials)
        for trial in range(n_trials):
            chosen = rng.choice(tickers, size=n, replace=False)
            port_returns = stock_returns[chosen].mean(axis=1)   # w_i = 1/n
            trial_vars[trial] = port_returns.var(ddof=1)
        avg_daily_var[n - 1] = trial_vars.mean()

    return pd.DataFrame({
        "n": np.arange(1, n_assets + 1),
        "avg_daily_variance": avg_daily_var,
        "avg_annualized_variance": avg_daily_var * 252,
        "avg_annualized_volatility": np.sqrt(avg_daily_var * 252),
    })


def theoretical_diversification_curve(stock_returns):
    """
    Closed-form curve: Var(n) = (1/n)*avg_var + ((n-1)/n)*avg_cov,
    using this universe's own average own-variance and average pairwise
    covariance (both computed from daily returns, then annualized).
    """
    cov_matrix = stock_returns.cov() * 252   # annualize the daily covariance matrix
    n_assets = cov_matrix.shape[0]

    avg_var = np.diag(cov_matrix).mean()
    off_diag_sum = cov_matrix.values.sum() - np.trace(cov_matrix.values)
    avg_cov = off_diag_sum / (n_assets * (n_assets - 1))

    n_range = np.arange(1, n_assets + 1)
    var_theory = avg_var / n_range + avg_cov * (n_range - 1) / n_range

    result = pd.DataFrame({
        "n": n_range,
        "theoretical_annualized_variance": var_theory,
        "theoretical_annualized_volatility": np.sqrt(var_theory),
    })
    return result, avg_var, avg_cov


def _ordinal_suffix(n):
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def risk_reduction_schedule(theory_df, checkpoints=(2, 6, 11, 21, 30)):
    """
    Marginal reduction in annualized volatility (percentage points) from
    adding each checkpoint stock, i.e. sigma(n-1) - sigma(n). Mirrors the
    course's schedule table (2nd, 6th, 11th, 21st...), capped at n=30
    since that's our full universe size.
    """
    vol = theory_df.set_index("n")["theoretical_annualized_volatility"]
    rows = []
    for n in checkpoints:
        if n > vol.index.max() or n < 2:
            continue
        sigma_n = vol.loc[n] * 100
        sigma_prev = vol.loc[n - 1] * 100
        rows.append({
            "Adding the...": f"{n}{_ordinal_suffix(n)} stock",
            "Sigma_p falls to (%)": round(sigma_n, 2),
            "Reduction (pp)": round(sigma_prev - sigma_n, 2),
        })
    return pd.DataFrame(rows)


def main():
    prices = download_prices()   # loads from cache -- already downloaded in Part A
    returns = compute_returns(prices)
    stock_returns = returns[TICKERS]

    empirical = empirical_diversification_curve(stock_returns)
    theoretical, avg_var, avg_cov = theoretical_diversification_curve(stock_returns)

    print(f"Universe average annualized variance (own risk): {avg_var:.4f}")
    print(f"Universe average annualized covariance (the systematic floor): {avg_cov:.4f}")
    print(f"Floor volatility (sqrt of avg covariance): {np.sqrt(avg_cov):.2%}")

    curve = empirical.merge(theoretical, on="n")
    curve.to_csv(OUTPUT_TABLES_DIR / "part1b_diversification_curve.csv", index=False)

    schedule = risk_reduction_schedule(theoretical)
    print("\nRisk-reduction schedule:")
    print(schedule.to_string(index=False))
    schedule.to_csv(OUTPUT_TABLES_DIR / "part1b_risk_reduction_schedule.csv", index=False)

    # --- Plot: empirical vs theoretical diversification curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(curve["n"], curve["avg_annualized_volatility"] * 100,
              marker="o", markersize=3, label="Empirical (200 random draws per n)")
    plt.plot(curve["n"], curve["theoretical_annualized_volatility"] * 100,
              linestyle="--", label="Theoretical closed-form")
    plt.axhline(np.sqrt(avg_cov) * 100, color="gray", linestyle=":",
                label=f"Systematic floor ({np.sqrt(avg_cov):.1%})")
    plt.title("Diversification: Portfolio Volatility vs. Number of Stocks")
    plt.xlabel("Number of assets, n")
    plt.ylabel("Annualized portfolio volatility (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "part1b_diversification_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()