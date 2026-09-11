import os
import databento as db
import pandas as pd

client = db.Historical(os.getenv("DATABENTO_API_KEY", ""))

start_date = "2000-01-01"
end_date = "2023-03-30"

print(f"Fetching complete active market instruments for {start_date}...")

data = client.timeseries.get_range(
    dataset="DBEQ.BASIC",
    start=start_date,
    end=end_date,        
    schema="definition",  
)

df = data.to_df()

active_stocks = df[df['security_type'] == 'C']
active_tickers = active_stocks['raw_symbol'].unique()
active_tickers = [ticker for ticker in active_tickers if not ticker.endswith('.TEST')]

print(f"\nSuccess! Found {len(active_tickers)} active common stocks on {start_date}.")
print("Sample Tickers:", active_tickers[:20])