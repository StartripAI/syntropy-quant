"""
Robust Multi-Provider Data Fetcher v5.0
Priority: Tiingo -> Polygon -> Yahoo (fallback)
Includes delay management and error handling.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
from typing import List, Dict, Optional
from requests.exceptions import Timeout, RequestException

class DataFetcherV5:
    """
    Robust Data Fetcher with Tiingo + Polygon API keys.
    Automatic retry, delay, and fallback logic.
    """
    
    def __init__(self, cache_dir="data_cache_v5"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load API keys from environment
        self.tiingo_key = os.environ.get("TIINGO_API_KEY")
        self.polygon_key = os.environ.get("POLYGON_API_KEY")
        
        if self.tiingo_key:
            print("   [Fetcher] ✅ Tiingo API Key loaded")
        if self.polygon_key:
            print("   [Fetcher] ✅ Polygon API Key loaded")
        
        # Rate limiting protection
        self.request_delays = {
            'tiingo': 2.0,  # Wait 2s after each Tiingo request
            'polygon': 1.0,  # Wait 1s after each Polygon request
            'yahoo': 5.0,   # Wait 5s after each Yahoo request
        }
        
        self.last_request_time = {
            'tiingo': 0,
            'polygon': 0,
            'yahoo': 0
        }
    
    def _wait_for_rate_limit(self, provider: str):
        """Wait appropriate time based on provider rate limit."""
        delay = self.request_delays.get(provider, 2.0)
        last_time = self.last_request_time.get(provider, 0)
        
        current_time = time.time()
        elapsed = current_time - last_time
        
        if elapsed < delay:
            actual_wait = delay - elapsed
            if actual_wait > 0:
                print(f"   [RateLimit] Waiting {actual_wait:.1f}s before {provider} request...")
                time.sleep(actual_wait)
    
        self.last_request_time[provider] = time.time()
    
    def _get_cache_path(self, symbol: str, start: str, end: str) -> str:
        return f"{self.cache_dir}/{symbol}_{start}_{end}.parquet"
    
    def _load_from_cache(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        cache_file = self._get_cache_path(symbol, start, end)
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                return df
            except:
                pass
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, start: str, end: str):
        cache_file = self._get_cache_path(symbol, start, end)
        if not df.empty and len(df) > 50:
            try:
                df.to_parquet(cache_file)
            except:
                pass
    
    def _fetch_tiingo(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch from Tiingo API with retry logic."""
        if not self.tiingo_key:
            return pd.DataFrame()
        
        self._wait_for_rate_limit('tiingo')
        
        for attempt in range(3):
            try:
                import pandas_datareader as pdr
                
                df = pdr.get_data_tiingo(
                    symbol,
                    start=start,
                    end=end,
                    api_key=self.tiingo_key
                )
                
                if df is not None and len(df) > 50:
                    # Reset index and standardize
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.reset_index(level=0, drop=True)
                    
                    df = df.rename(columns={
                        'open': 'Open', 'high': 'High',
                        'low': 'Low', 'close': 'Close',
                        'volume': 'Volume'
                    })
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill').fillna(method='bfill')
                    self.last_request_time['tiingo'] = time.time()
                    
                    print(f"   OK (Tiingo, attempt {attempt+1})")
                    return df
                except Exception as e:
                    if 'Rate limit' in str(e) or 'Too Many Requests' in str(e):
                        print(f"   [RateLimit] Tiingo rate limit (attempt {attempt+1})")
                        time.sleep(3 ** attempt)  # Exponential backoff
                        continue
                    else:
                        print(f"   [Error] Tiingo failed: {str(e)[:50]}")
                        time.sleep(1)
                        continue
        
        print(f"   Failed (Tiingo, all attempts)")
        return pd.DataFrame()
    
    def _fetch_polygon(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch from Polygon API with retry logic."""
        if not self.polygon_key:
            return pd.DataFrame()
        
        self._wait_for_rate_limit('polygon')
        
        for attempt in range(2):
            try:
                import requests
                from datetime import datetime
                
                start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
                end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
                
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_ts}/{end_ts}"
                params = {
                    'adjusted': 'true',
                    'sort': 'asc',
                    'apiKey': self.polygon_key
                }
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                if 'results' in data and len(data['results']) > 50:
                    results = data['results']
                    
                    df = pd.DataFrame(results)
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.set_index('date')
                    
                    df = df.rename(columns={
                        'o': 'Open', 'h': 'High',
                        'l': 'Low', 'c': 'Close',
                        'v': 'Volume'
                    })
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    self.last_request_time['polygon'] = time.time()
                    
                    print(f"   OK (Polygon, attempt {attempt+1})")
                    return df
                except RequestException as e:
                    if '403' in str(e) or '429' in str(e):
                        print(f"   [RateLimit] Polygon rate limit (attempt {attempt+1})")
                        time.sleep(5)
                        continue
                    except Timeout:
                        print(f"   [Timeout] Polygon timeout (attempt {attempt+1})")
                        continue
                    except Exception as e:
                        print(f"   [Error] Polygon failed: {str(e)[:50]}")
                        time.sleep(2)
                        continue
        
        print(f"   Failed (Polygon, all attempts)")
        return pd.DataFrame()
    
    def _fetch_yahoo(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch from Yahoo Finance (last resort)."""
        self._wait_for_rate_limit('yahoo')
        
        for attempt in range(2):
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True
                )
                
                if df is not None and len(df) > 50:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill').fillna(method='bfill')
                    self.last_request_time['yahoo'] = time.time()
                    
                    print(f"   OK (Yahoo, attempt {attempt+1})")
                    return df
                except Exception as e:
                    print(f"   [Error] Yahoo failed: {str(e)[:50]}")
                    time.sleep(3)
                    continue
        
        print(f"   Failed (Yahoo, all attempts)")
        return pd.DataFrame()
    
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Main fetch method with multi-provider fallback:
        Tiingo -> Polygon -> Yahoo
        """
        # Check cache first
        cached = self._load_from_cache(symbol, start, end)
        if cached is not None:
            print(f"{symbol} (cached)")
            return cached
        
        print(f"Fetching {symbol}...", end=" ", flush=True)
        
        # Try Tiingo first
        df = self._fetch_tiingo(symbol, start, end)
        if not df.empty:
            self._save_to_cache(df, symbol, start, end)
            return df
        
        # Fallback to Polygon
        df = self._fetch_polygon(symbol, start, end)
        if not df.empty:
            self._save_to_cache(df, symbol, start, end)
            return df
        
        # Last resort: Yahoo
        df = self._fetch_yahoo(symbol, start, end)
        if not df.empty:
            self._save_to_cache(df, symbol, start, end)
            return df
        
        print("Failed (all providers)")
        return pd.DataFrame()
    
    def fetch_multiple(self, symbols: List[str], start: str, end: str, parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple symbols with optional parallel execution.
        """
        if not parallel:
            results = {}
            for i, symbol in enumerate(symbols):
                print(f"[{i+1}/{len(symbols)}] Fetching {symbol}...")
                results[symbol] = self.fetch(symbol, start, end)
                time.sleep(0.5)  # Small delay between requests
            return results
        else:
            # Parallel execution
            from concurrent.futures import ThreadPoolExecutor
            import os
            
            os.environ['TIINGO_API_KEY'] = self.tiingo_key or ''
            os.environ['POLYGON_API_KEY'] = self.polygon_key or ''
            
            def fetch_with_env(sym):
                return self.fetch(sym, start, end)
            
            results = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(fetch_with_env, sym): sym
                    for sym in symbols
                }
                
                for future in futures:
                    sym = future.result()
                    results[sym] = sym
                    print(f"  [DONE] {sym}")
            
            return results


# Backward compatibility
DataFetcher = DataFetcherV5

