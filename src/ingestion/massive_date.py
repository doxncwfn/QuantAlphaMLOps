import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def get_valid_api_key(cli_arg_key: str = None) -> str:
    api_key = cli_arg_key or os.getenv("MASSIVE_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not api_key or api_key.lower().startswith("your_"):
        raise ValueError("Valid Massive API Key must be supplied via argument or MASSIVE_API_KEY env/.env.")
    return api_key

class MassiveThrottledManager:
    def __init__(self, api_key: str = None):
        self.api_key = get_valid_api_key(api_key)

    def get_active_symbols_on_date(self, target_date: str) -> list[str]:
        active_symbols = []
        current_url = "https://api.massive.com/v3/reference/tickers"
        params = {
            "date": target_date,
            "active": "true",
            "limit": 1000,
            "apiKey": self.api_key
        }
        backoff_delay = 3.0 
        print("Commencing Throttled Data Stream Extraction...")

        while current_url:
            try:
                request_url = current_url
                request_params = params if "cursor" not in current_url else {"apiKey": self.api_key}

                response = requests.get(
                    request_url, 
                    params=request_params,
                    timeout=15
                )

                if response.status_code == 429:
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 30.0) 
                    continue 

                if not response.text or not response.headers.get("Content-Type", "").startswith("application/json"):
                    print(f"Received unparseable non-JSON raw body from server (Status: {response.status_code}).")
                    print(f"Snippet: {response.text[:200]}")
                    break

                response.raise_for_status()
                payload = response.json()
                backoff_delay = 3.0 
                
                results = payload.get("results", [])
                if not results:
                    break

                for item in results:
                    if isinstance(item, dict) and "ticker" in item:
                        active_symbols.append(item["ticker"])

                current_url = payload.get("next_url")
                if current_url:
                    time.sleep(12.0)

            except Exception as err:
                print(f"Failed during page transmission: {err}")
                time.sleep(5.0)
                continue

        return active_symbols

if __name__ == "__main__":
    try:
        api_token = get_valid_api_key()
        manager = MassiveThrottledManager(api_key=api_token)
        target_day = "2003-12-31"

        print(f"Processing point-in-time assets for: {target_day}...")
        historical_tickers = manager.get_active_symbols_on_date(target_day)

        print(f"Download complete. Total Active tickers found: {len(historical_tickers)}")
        if historical_tickers:
            print(f"All Historical Tickers: {historical_tickers}")

    except ValueError as err:
        print(f"Environment Error: {err}")