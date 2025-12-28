"""
Data Fetcher Module v4.0

Robust data fetching with multiple provider fallback.
Prioritizes Yahoo Finance for reliability.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


class AssetCategory(Enum):
    """Asset categories for universe organization"""
    INDEX = "index"
    TECH = "tech"
    PHARMA = "pharma"
    CONSUMER = "consumer"
    FINANCE = "finance"
    ENERGY = "energy"


@dataclass
class AssetInfo:
    """Asset metadata"""
    symbol: str
    name: str
    category: AssetCategory


# Full asset universe
ASSET_UNIVERSE: Dict[str, AssetInfo] = {
    # Indices
    "SPY": AssetInfo("SPY", "S&P 500 ETF", AssetCategory.INDEX),
    "QQQ": AssetInfo("QQQ", "Nasdaq 100 ETF", AssetCategory.INDEX),
    "IWM": AssetInfo("IWM", "Russell 2000 ETF", AssetCategory.INDEX),
    "DIA": AssetInfo("DIA", "Dow Jones ETF", AssetCategory.INDEX),

    # Tech
    "AAPL": AssetInfo("AAPL", "Apple", AssetCategory.TECH),
    "MSFT": AssetInfo("MSFT", "Microsoft", AssetCategory.TECH),
    "GOOGL": AssetInfo("GOOGL", "Alphabet", AssetCategory.TECH),
    "AMZN": AssetInfo("AMZN", "Amazon", AssetCategory.TECH),
    "NVDA": AssetInfo("NVDA", "NVIDIA", AssetCategory.TECH),
    "META": AssetInfo("META", "Meta Platforms", AssetCategory.TECH),
    "TSLA": AssetInfo("TSLA", "Tesla", AssetCategory.TECH),

    # Pharma
    "LLY": AssetInfo("LLY", "Eli Lilly", AssetCategory.PHARMA),
    "UNH": AssetInfo("UNH", "UnitedHealth", AssetCategory.PHARMA),
    "JNJ": AssetInfo("JNJ", "Johnson & Johnson", AssetCategory.PHARMA),
    "PFE": AssetInfo("PFE", "Pfizer", AssetCategory.PHARMA),
    "ABBV": AssetInfo("ABBV", "AbbVie", AssetCategory.PHARMA),
    "MRK": AssetInfo("MRK", "Merck", AssetCategory.PHARMA),

    # Consumer
    "WMT": AssetInfo("WMT", "Walmart", AssetCategory.CONSUMER),
    "PG": AssetInfo("PG", "Procter & Gamble", AssetCategory.CONSUMER),
    "KO": AssetInfo("KO", "Coca-Cola", AssetCategory.CONSUMER),
    "PEP": AssetInfo("PEP", "PepsiCo", AssetCategory.CONSUMER),
    "COST": AssetInfo("COST", "Costco", AssetCategory.CONSUMER),
    "MCD": AssetInfo("MCD", "McDonald's", AssetCategory.CONSUMER),
}


class DataFetcher:
    """
    Robust Data Fetcher v4.0

    Features:
    - Multiple provider fallback (Yahoo -> Tiingo -> Stooq)
    - Local caching for faster repeated access
    - Automatic price adjustment
    - NaN handling and data validation
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        provider_priority: Optional[List[str]] = None,
        adjust_prices: bool = True
    ):
        """
        Initialize data fetcher.

        Args:
            cache_dir: Directory for caching data (optional)
            provider_priority: List of providers in priority order
            adjust_prices: Whether to adjust prices for splits/dividends
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.provider_priority = provider_priority or ["yahoo", "tiingo", "stooq"]
        self.adjust_prices = adjust_prices
        self.tiingo_key = os.environ.get("TIINGO_API_KEY")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, start: str, end: str) -> Optional[Path]:
        """Get cache file path for a symbol"""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{symbol}_{start}_{end}.parquet"

    def _load_from_cache(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        cache_path = self._get_cache_path(symbol, start, end)
        if cache_path and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass
        return None

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, start: str, end: str):
        """Save data to cache"""
        cache_path = self._get_cache_path(symbol, start, end)
        if cache_path and not df.empty:
            try:
                df.to_parquet(cache_path)
            except Exception:
                pass

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase"""
        df = df.copy()

        # Handle MultiIndex columns (from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Map to standard names
        column_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close',
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume', 'adjClose': 'adj_close'
        }

        df = df.rename(columns=column_map)

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    df[col] = df['close'] if 'close' in df.columns else 0

        return df[required]

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data"""
        if df.empty:
            return df

        df = df.copy()

        # Forward fill NaN values
        df = df.ffill()

        # Remove rows with zero/negative prices
        mask = (df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0)
        df = df[mask]

        # Ensure high >= low
        df.loc[df['high'] < df['low'], 'high'] = df.loc[df['high'] < df['low'], 'low']

        return df

    def _fetch_yahoo(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch from Yahoo Finance"""
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=self.adjust_prices
            )
            if len(df) > 10:
                return self._standardize_columns(df)
        except Exception:
            pass
        return pd.DataFrame()

    def _fetch_tiingo(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch from Tiingo"""
        if not self.tiingo_key:
            return pd.DataFrame()

        try:
            import pandas_datareader as pdr
            df = pdr.get_data_tiingo(
                symbol,
                start=start,
                end=end,
                api_key=self.tiingo_key
            )
            if len(df) > 10:
                df = df.reset_index(level=0, drop=True)
                return self._standardize_columns(df)
        except Exception:
            pass
        return pd.DataFrame()

    def _fetch_stooq(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch from Stooq"""
        try:
            import pandas_datareader as pdr
            df = pdr.get_data_stooq(symbol, start=start, end=end)
            if len(df) > 10:
                df = df.sort_index()
                return self._standardize_columns(df)
        except Exception:
            pass
        return pd.DataFrame()

    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch market data for a symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns [open, high, low, close, volume]
        """
        print(f"Fetching {symbol}...", end=" ", flush=True)

        # Try cache first
        cached = self._load_from_cache(symbol, start_date, end_date)
        if cached is not None and len(cached) > 50:
            print("(cached)")
            return cached

        # Try each provider in priority order
        providers = {
            "yahoo": self._fetch_yahoo,
            "tiingo": self._fetch_tiingo,
            "stooq": self._fetch_stooq
        }

        for provider in self.provider_priority:
            if provider in providers:
                df = providers[provider](symbol, start_date, end_date)
                if not df.empty and len(df) > 50:
                    df = self._validate_data(df)
                    self._save_to_cache(df, symbol, start_date, end_date)
                    print(f"Success ({provider})")
                    return df

        print("Failed")
        return pd.DataFrame()

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        results = {}
        for symbol in symbols:
            df = self.fetch(symbol, start_date, end_date)
            if not df.empty:
                results[symbol] = df
        return results

    def get_universe(self, category: Optional[AssetCategory] = None) -> List[str]:
        """
        Get list of symbols in universe.

        Args:
            category: Filter by category (optional)

        Returns:
            List of ticker symbols
        """
        if category is None:
            return list(ASSET_UNIVERSE.keys())
        return [
            info.symbol
            for info in ASSET_UNIVERSE.values()
            if info.category == category
        ]
