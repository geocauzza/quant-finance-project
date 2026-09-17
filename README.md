# quant-finance-project

## Structure

- `code/` — shared modules (`config.py`, `data_utils.py`) and one script per assignment part
- `data/raw/` — cached price data, downloaded once and reused
- `data/processed/` — derived files (returns, weights, computed series)
- `output/figures/` — saved plots
- `output/tables/` — saved result tables

## Setup

```powershell
& "C:\pyenvs\quant-finance-project\venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

## Running

```powershell
cd code
python part1a_portfolios.py        # Project 1, Part A: VW vs EW portfolios
python part1b_diversification.py   # Project 1, Part B: diversification experiment
python part2_frontier.py           # Project 2: tangency, MVP, and the frontier
```

`part1a_portfolios.py` downloads and caches the 30-stock + benchmark + risk-free
price panel on first run; every later script reuses that cache.
