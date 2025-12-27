"""
Data Fetcher Module

Fetches historical market data for backtesting.
"""

import os
import time
import io
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class AssetCategory(Enum):
    """Asset categories for classification"""
    INDEX = "index"
    TECH = "tech"
    PHARMA = "pharma"
    CONSUMER = "consumer"
    FINANCIALS = "financials"


@dataclass
class AssetInfo:
    """Information about a tradeable asset"""
    symbol: str
    name: str
    category: AssetCategory


# Asset universe definition
ASSET_UNIVERSE = {
    # Major Indices (non-overlapping broad coverage)
    'IWM': AssetInfo('IWM', 'IWM', AssetCategory.INDEX),
    'VTI': AssetInfo('VTI', 'VTI', AssetCategory.INDEX),
    'RSP': AssetInfo('RSP', 'RSP', AssetCategory.INDEX),

    # Tech Giants
    'AAPL': AssetInfo('AAPL', 'AAPL', AssetCategory.TECH),
    'MSFT': AssetInfo('MSFT', 'MSFT', AssetCategory.TECH),
    'GOOGL': AssetInfo('GOOGL', 'GOOGL', AssetCategory.TECH),
    'AMZN': AssetInfo('AMZN', 'AMZN', AssetCategory.TECH),
    'NVDA': AssetInfo('NVDA', 'NVDA', AssetCategory.TECH),
    'META': AssetInfo('META', 'META', AssetCategory.TECH),
    'TSLA': AssetInfo('TSLA', 'TSLA', AssetCategory.TECH),

    # Pharma/Biotech
    'LLY': AssetInfo('LLY', 'LLY', AssetCategory.PHARMA),
    'UNH': AssetInfo('UNH', 'UNH', AssetCategory.PHARMA),
    'JNJ': AssetInfo('JNJ', 'JNJ', AssetCategory.PHARMA),
    'PFE': AssetInfo('PFE', 'PFE', AssetCategory.PHARMA),
    'ABBV': AssetInfo('ABBV', 'ABBV', AssetCategory.PHARMA),
    'MRK': AssetInfo('MRK', 'MRK', AssetCategory.PHARMA),

    # Consumer Staples
    'WMT': AssetInfo('WMT', 'WMT', AssetCategory.CONSUMER),
    'PG': AssetInfo('PG', 'PG', AssetCategory.CONSUMER),
    'KO': AssetInfo('KO', 'KO', AssetCategory.CONSUMER),
    'PEP': AssetInfo('PEP', 'PEP', AssetCategory.CONSUMER),
    'COST': AssetInfo('COST', 'COST', AssetCategory.CONSUMER),
    'MCD': AssetInfo('MCD', 'MCD', AssetCategory.CONSUMER),
}


class DataFetcher:
    """
    Fetches and caches market data from Yahoo Finance.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        provider_priority: Optional[List[str]] = None,
        adjust_prices: bool = True,
        yahoo_pause: float = 1.0
    ):
        self.cache_dir = cache_dir
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.adjust_prices = adjust_prices
        self.yahoo_pause = yahoo_pause
        self.providers = self._build_providers(provider_priority)
        self.cache_tag = "-".join(self.providers).replace("/", "_")[:64]

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _build_providers(self, provider_priority: Optional[List[str]]) -> List[str]:
        if provider_priority:
            return [p.strip().lower() for p in provider_priority if p.strip()]

        env_providers = os.getenv("SYNQUANT_PROVIDERS", "").strip()
        if env_providers:
            return [p.strip().lower() for p in env_providers.split(",") if p.strip()]

        providers = []
        if os.getenv("TIINGO_API_KEY"):
            providers.append("tiingo")
        if os.getenv("POLYGON_API_KEY"):
            providers.append("polygon")
        providers.extend(["yahoo", "stooq"])
        return providers

    def _normalize_columns(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """Normalize column names to expected OHLCV fields."""
        if isinstance(df.columns, pd.MultiIndex):
            if symbol and symbol in df.columns.get_level_values(0):
                df = df[symbol].copy()
            elif symbol and symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, level=-1, axis=1).copy()
            else:
                df = df.copy()
                df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

        cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                parts = [str(p) for p in c if p not in (None, '')]
                field = None
                for p in parts:
                    pl = p.lower()
                    if pl in ('open', 'high', 'low', 'close', 'adj close', 'volume'):
                        field = p
                        break
                col = field if field else (parts[-1] if parts else '')
            else:
                col = c
            col = str(col).lower()
            cols.append(col)

        df.columns = cols

        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']

        return df

    def _fetch_stooq(
        self,
        symbol: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """Fallback data source using Stooq CSV."""
        stooq_symbol = f"{symbol.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"

        try:
            df = pd.read_csv(url)
            if df.empty or 'Date' not in df.columns:
                return pd.DataFrame()

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns=str.lower)
            df = df.set_index('date').sort_index()
            df = df.loc[start:end]

            if df.empty:
                return pd.DataFrame()

            df['symbol'] = symbol
            return self._normalize_columns(df, symbol)
        except Exception as e:
            print(f"Error fetching {symbol} from stooq: {e}")
            return pd.DataFrame()

    def _fetch_yahoo(
        self,
        symbol: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=self.adjust_prices,
                threads=False
            )
            if df.empty:
                return pd.DataFrame()
            df = self._normalize_columns(df, symbol)
            return df
        except Exception as e:
            print(f"Error fetching {symbol} from yahoo: {e}")
            return pd.DataFrame()

    def _fetch_tiingo(
        self,
        symbol: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        api_key = os.getenv("TIINGO_API_KEY")
        if not api_key:
            return pd.DataFrame()
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        try:
            resp = requests.get(
                url,
                params={"startDate": start, "endDate": end, "format": "csv", "token": api_key},
                timeout=20
            )
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if df.empty:
                return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            if self.adjust_prices and {'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume'}.issubset(df.columns):
                df = df.rename(columns={
                    'adjOpen': 'open',
                    'adjHigh': 'high',
                    'adjLow': 'low',
                    'adjClose': 'close',
                    'adjVolume': 'volume'
                })
            else:
                df = df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
            return self._normalize_columns(df, symbol)
        except Exception as e:
            print(f"Error fetching {symbol} from tiingo: {e}")
            return pd.DataFrame()

    def _fetch_polygon(
        self,
        symbol: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            return pd.DataFrame()
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
        try:
            resp = requests.get(
                url,
                params={"adjusted": "true" if self.adjust_prices else "false", "sort": "asc", "apiKey": api_key},
                timeout=20
            )
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                return pd.DataFrame()
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('date').sort_index()
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            return self._normalize_columns(df, symbol)
        except Exception as e:
            print(f"Error fetching {symbol} from polygon: {e}")
            return pd.DataFrame()

    def _ensure_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ['open', 'high', 'low', 'close', 'volume']
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
        volume = df['volume']
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
            df = df.copy()
            df['volume'] = volume
        if volume.isna().all():
            df['volume'] = 1.0
        df = df[required].copy()
        df = df.sort_index()
        return df

    def _cache_path(self, symbol: str, start: str, end: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe_symbol = symbol.replace("^", "IDX")
        return os.path.join(self.cache_dir, f"{safe_symbol}_{start}_{end}_{self.cache_tag}.csv")

    def _read_cache(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol, start, end)
        if not path or not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    def _write_cache(self, symbol: str, start: str, end: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol, start, end)
        if not path:
            return
        df.to_csv(path)

    def fetch(
        self,
        symbol: str,
        start: str = '2020-01-01',
        end: str = '2024-12-31'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start}_{end}"

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        cached = self._read_cache(symbol, start, end)
        if cached is not None and not cached.empty:
            self.data_cache[cache_key] = cached
            return cached

        for provider in self.providers:
            if provider == "yahoo":
                df = self._fetch_yahoo(symbol, start, end)
                if self.yahoo_pause > 0:
                    time.sleep(self.yahoo_pause)
            elif provider == "stooq":
                df = self._fetch_stooq(symbol, start, end)
            elif provider == "tiingo":
                df = self._fetch_tiingo(symbol, start, end)
            elif provider == "polygon":
                df = self._fetch_polygon(symbol, start, end)
            else:
                continue

            if df.empty:
                continue

            df = self._ensure_ohlcv(df)
            if df.empty:
                continue

            df['symbol'] = symbol
            self._write_cache(symbol, start, end, df)
            self.data_cache[cache_key] = df
            return df

        print(f"Error fetching {symbol}: no data from providers {self.providers}")
        return pd.DataFrame()

    def fetch_multiple(
        self,
        symbols: List[str],
        start: str = '2020-01-01',
        end: str = '2024-12-31'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        """
        results = {}
        for symbol in symbols:
            df = self.fetch(symbol, start, end)
            if not df.empty:
                results[symbol] = df
        return results

    def fetch_by_category(
        self,
        category: AssetCategory,
        start: str = '2020-01-01',
        end: str = '2024-12-31'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all assets in a category.
        """
        symbols = [
            info.symbol for info in ASSET_UNIVERSE.values()
            if info.category == category
        ]
        return self.fetch_multiple(symbols, start, end)

    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Get asset information"""
        return ASSET_UNIVERSE.get(symbol)

    def list_symbols_by_category(self, category: AssetCategory) -> List[str]:
        """List all symbols in a category"""
        return [
            info.symbol for info in ASSET_UNIVERSE.values()
            if info.category == category
        ]


def generate_synthetic_data(
    n_days: int = 1000,
    volatility: float = 0.02,
    trend: float = 0.0005,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.
    """
    np.random.seed(seed)

    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')

    # Generate returns
    returns = np.random.normal(trend, volatility, n_days)

    # Add regime changes
    regimes = np.sin(np.linspace(0, 4 * np.pi, n_days)) * 0.5
    returns = returns * (1 + regimes * 0.5)

    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate volume
    base_volume = 1e6
    volume = base_volume * (1 + 0.5 * np.abs(returns) / volatility)
    volume = volume * np.random.uniform(0.8, 1.2, n_days)

    # Create OHLC
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_price = prices * (1 + np.random.normal(0, 0.005, n_days))

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)

    return df
