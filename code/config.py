"""
config.py
Shared configuration for the Quant Finance course project (Project 1, Part A).

Holds:
- The 30-stock universe (chosen broadly across GICS sectors)
- The sample date range
- Benchmark and risk-free tickers
- File paths used across scripts

Every other script imports from here, so the universe/dates are defined
exactly once and stay consistent across the whole project.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Universe: 30 S&P 500 names, spread across all 11 GICS sectors
# ---------------------------------------------------------------------------
TICKERS = [
    # Technology
    "AAPL", "MSFT", "CSCO", "ADBE",
    # Health Care
    "JNJ", "PFE", "UNH",
    # Financials
    "JPM", "BAC", "V",
    # Consumer Discretionary
    "AMZN", "HD", "MCD",
    # Communication Services
    "GOOGL", "VZ", "NFLX",
    # Industrials
    "BA", "HON", "UPS",
    # Consumer Staples
    "PG", "KO", "WMT",
    # Energy
    "XOM", "CVX",
    # Utilities
    "NEE", "DUK",
    # Real Estate
    "SPG", "PLD",
    # Materials
    "APD", "FCX",
]

assert len(TICKERS) == 30, f"Expected 30 tickers, got {len(TICKERS)}"

# ---------------------------------------------------------------------------
# Sample window and benchmark / risk-free assets
# ---------------------------------------------------------------------------
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

BENCHMARK_TICKER = "^GSPC"   # S&P 500 index (price return)
RISK_FREE_TICKER = "BIL"     # SPDR 1-3 Month T-Bill ETF, used as a risk-free proxy.
                              # (Switched from ^IRX: Yahoo/yfinance frequently fails
                              # to serve "^"-prefixed index tickers -- a known,
                              # recurring issue, not specific to this project.)

# ---------------------------------------------------------------------------
# File paths (relative to the project root, one level above code/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "output" / "tables"

for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

PRICES_CACHE = DATA_RAW_DIR / "sp500_daily_2016_2025.csv"
SHARES_CACHE = DATA_RAW_DIR / "shares_outstanding.csv"