import yfinance as yf
import pandas as pd
from pathlib import Path
import time
import random

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG

class YFinanceIngestionEngine:
    def __init__(self):
        self.data_dir = Path(CONFIG['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tickers = CONFIG['tickers']

    def fetch_and_store_history(self, start_date="2018-01-01"):
        print(f"Initiating yfinance data extraction for {len(self.tickers)} equities...")
        
        for i, ticker in enumerate(self.tickers):
            save_path = self.data_dir / f"{ticker}.csv"
            if save_path.exists():
                continue
            print(f"[{i+1}/{len(self.tickers)}] Fetching {ticker}...")
            yf_ticker = ticker.replace('.', '-')
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = yf.download(
                        yf_ticker,
                        start=start_date,
                        auto_adjust=False,
                        progress=False
                    )
                    df = df.dropna(how='all')
                    if df.empty:
                        print(f"  -> {ticker} returned empty data. Possibly delisted/acquired.")
                        break
                    df.index.name = 'Date'
                    df.reset_index(inplace=True)
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.rename(columns={
                        'Adj Close': 'Close',
                        'Open': 'Open',
                        'High': 'High',
                        'Low': 'Low',
                        'Volume': 'Volume'
                    })
                    df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv(save_path, index=False)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  -> Timeout on {ticker}. Retrying in 5 seconds... ({attempt+1}/{max_retries})")
                        time.sleep(5)
                    else:
                        print(f"  -> Error fetching {ticker} after {max_retries} attempts: {e}")
            time.sleep(random.uniform(1.0, 2.5))

if __name__ == "__main__":
    engine = YFinanceIngestionEngine()
    engine.fetch_and_store_history()