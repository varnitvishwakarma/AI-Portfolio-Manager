import yfinance as yf
import os

os.makedirs("./data/raw", exist_ok=True)

tickers = ["^GSPC", "^FTSE", "^N225", "EEM", "GC=F", "^TNX"]
start_date = "2010-01-01"
end_date = "2020-12-31"

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

for ticker in tickers:
    df = data[ticker].copy()
    file_path = f"data/raw/{ticker.replace('^','')}.csv"
    df.to_csv(file_path)
    print(f"Saved {ticker} data to {file_path}")

print("✅ Data collection completed and saved in data/raw/")
