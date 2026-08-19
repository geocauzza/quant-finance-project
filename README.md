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
python part1a_portfolios.py
```
