import yfinance as yf
import pandas as pd
import time

class DataFetcher:
    def fetch(self, symbol, start, end):
        print(f"Downloading {symbol}...", end=" ", flush=True)
        for _ in range(3):
            try:
                # Yahoo Finance is the only robust free source
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
                if len(df) > 100:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill')
                    print("OK")
                    return df
            except:
                time.sleep(1)
        print("Failed")
        return pd.DataFrame()
