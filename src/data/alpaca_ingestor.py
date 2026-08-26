import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Import your existing config to keep the universe synchronized
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG

class AlpacaIngestionEngine:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API keys not found in .env file.")
            
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.data_dir = Path(CONFIG['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tickers = CONFIG['tickers']

    def _batch_tickers(self, batch_size=100):
        """Splits the 413-stock universe into chunks to prevent URL length limits."""
        for i in range(0, len(self.tickers), batch_size):
            yield self.tickers[i:i + batch_size]

    def fetch_and_store_history(self, start_date="2018-01-01"):
        """Fetches daily OHLCV data and saves isolated CSVs per ticker."""
        print(f"Initiating Alpaca data extraction for {len(self.tickers)} equities...")
        
        for batch in self._batch_tickers():
            print(f"Fetching batch: {batch[0]} to {batch[-1]}")
            
            request_params = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(start_date, "%Y-%m-%d") if start_date else None,
                end=datetime.now(),
                feed=DataFeed.IEX
            )
            
            try:
                bars = self.client.get_stock_bars(request_params)
                if bars.df.empty:
                    continue
                
                # Alpaca returns a MultiIndex DataFrame (symbol, timestamp)
                df = bars.df.reset_index()
                
                # Standardize columns to match UniversalDataProcessor expectations
                df = df.rename(columns={
                    'symbol': 'Ticker',
                    'timestamp': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Ensure Date is timezone-naive to match Fama-French data
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                
                # Isolate and save individual CSVs
                for ticker, group in df.groupby('Ticker'):
                    save_path = self.data_dir / f"{ticker}.csv"
                    group[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv(save_path, index=False)
                    
            except Exception as e:
                print(f"Error fetching batch ending in {batch[-1]}: {e}")
                
            # Brief pause to respect rate limits
            time.sleep(1)

if __name__ == "__main__":
    engine = AlpacaIngestionEngine()
    engine.fetch_and_store_history()