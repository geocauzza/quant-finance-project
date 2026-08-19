import yfinance as yf

tickers = ["AAPL"]  # your 30 stocks
data = yf.download(tickers, start="2016-01-01", end="2025-12-31", auto_adjust=True)
print(data.head())