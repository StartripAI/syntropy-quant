import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime

from dotenv import load_dotenv

class DataFetcher:
    def __init__(self, cache_dir=None):
        load_dotenv()
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
    def fetch(self, symbol, start, end):
        # 1. Check Cache
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{symbol}_{start}_{end}.csv")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        print(f"Downloading {symbol}...", end=" ", flush=True)
        # 2. Try Yahoo Finance
        for _ in range(2):
            try:
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
                if len(df) > 10:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [c.capitalize() for c in df.columns]
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill')
                    print("YF-OK")
                    if self.cache_dir:
                        df.to_csv(cache_path)
                    return df
            except Exception as e:
                time.sleep(2)
        
        # 3. Try Alpaca Fallback
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            api_key = os.environ.get('ALPACA_API_KEY')
            secret_key = os.environ.get('ALPACA_SECRET_KEY')
            
            if api_key and secret_key:
                client = StockHistoricalDataClient(api_key, secret_key)
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=datetime.strptime(start, '%Y-%m-%d'),
                    end=datetime.strptime(end, '%Y-%m-%d')
                )
                bars = client.get_stock_bars(request)
                df = bars.df
                if not df.empty:
                    # Alpaca returns MultiIndex (symbol, timestamp)
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.xs(symbol)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    print("ALPACA-OK")
                    if self.cache_dir:
                        df.to_csv(cache_path)
                    return df
        except Exception as e:
            # print(f"Alpaca-Error: {e}")
            pass

        # 4. Try Tiingo Fallback
        try:
            tiingo_key = os.environ.get('TIINGO_API_KEY')
            if tiingo_key:
                import requests
                url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start}&endDate={end}&token={tiingo_key}"
                headers = {'Content-Type': 'application/json'}
                response = requests.get(url, headers=headers)
                data = response.json()
                if data and isinstance(data, list):
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    print("TIINGO-OK")
                    if self.cache_dir:
                        df.to_csv(cache_path)
                    return df
        except Exception as e:
            pass

        print("Failed")
        return pd.DataFrame()
