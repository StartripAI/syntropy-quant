import numpy as np
import pandas as pd
import torch
from typing import Tuple


class FeatureBuilder:
    """
    Robust Feature Engineering (v4.0).
    Mathematical Safety: Epsilon stabilization for all divisions.
    """

    def __init__(self, short_window: int = 5, mid_window: int = 20, long_window: int = 60):
        """
        Initialize with configurable windows for multi-scale analysis.

        Args:
            short_window: Short-term lookback (default 5)
            mid_window: Medium-term lookback (default 20)
            long_window: Long-term lookback (default 60)
        """
        self.short_window = short_window
        self.mid_window = mid_window
        self.long_window = long_window
        self.warmup = max(short_window, mid_window, long_window)

    def build(self, df: pd.DataFrame) -> torch.Tensor:
        if len(df) < 30: return torch.empty(0)
        
        df = df.copy()
        epsilon = 1e-8
        
        # 1. Log Returns (Scale Invariant)
        close = df['Close'].values
        log_ret = np.diff(np.log(close + epsilon), prepend=close[0])
        
        # 2. Volatility (Local Energy)
        vol = pd.Series(log_ret).rolling(20).std().fillna(0).values
        
        # 3. Order Flow / Momentum
        # Avoid High==Low singularity (CRITICAL FIX)
        hl_range = df['High'].values - df['Low'].values
        hl_range = np.maximum(hl_range, epsilon) 
        
        # Close Location Value
        clv = ((close - df['Low'].values) - (df['High'].values - close)) / hl_range
        flow = clv * np.log(df['Volume'].values + 1.0)
        
        # Normalize Flow (Z-Score)
        flow_mean = pd.Series(flow).rolling(20).mean().fillna(0).values
        flow_std = pd.Series(flow).rolling(20).std().replace(0, 1).values
        momentum = (flow - flow_mean) / (flow_std + epsilon)
        momentum = np.clip(momentum, -5, 5) # Clip outliers
        
        # 4. Potential Well (Mean Reversion)
        ma20 = pd.Series(close).rolling(20).mean().fillna(close[0]).values
        position = (close - ma20) / (ma20 + epsilon) * 100
        
        # Stack: [LogRet, Vol, Momentum, Position]
        features = np.stack([log_ret, vol, momentum, position], axis=1)
        
        # Drop initial NaN window
        return torch.tensor(features[20:], dtype=torch.float32)

    def build_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build features for backtest engine (numpy array output).
        Extended feature set for regime detection and momentum analysis.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]

        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        # Normalize column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if len(df) < self.warmup + 10:
            return np.zeros((len(df), 12))

        epsilon = 1e-8
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_ = df['open'].values if 'open' in df.columns else close

        n = len(close)
        features = np.zeros((n, 12))

        # 1. Log Returns
        log_ret = np.zeros(n)
        log_ret[1:] = np.diff(np.log(close + epsilon))
        features[:, 0] = log_ret

        # 2. Volatility (short, mid, long)
        for i in range(self.short_window, n):
            features[i, 1] = np.std(log_ret[max(0, i-self.short_window):i]) * np.sqrt(252)
        for i in range(self.mid_window, n):
            features[i, 2] = np.std(log_ret[max(0, i-self.mid_window):i]) * np.sqrt(252)

        # 3. RSI (Relative Strength Index)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().fillna(0).values
        avg_loss = pd.Series(loss).rolling(14).mean().fillna(epsilon).values
        rs = avg_gain / (avg_loss + epsilon)
        features[:, 3] = (100 - 100 / (1 + rs)) / 100 - 0.5  # Centered RSI

        # 4. MACD
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9).mean().values
        features[:, 4] = macd_line / (close + epsilon)  # Normalized MACD
        features[:, 5] = signal_line / (close + epsilon)
        features[:, 6] = (macd_line - signal_line) / (close + epsilon)  # MACD histogram

        # 5. Bollinger Band Position
        ma20 = pd.Series(close).rolling(20).mean().fillna(close[0]).values
        std20 = pd.Series(close).rolling(20).std().fillna(1).values
        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20
        bb_width = upper_band - lower_band
        features[:, 7] = (close - lower_band) / (bb_width + epsilon) - 0.5  # BB position

        # 6. Volume Profile
        hl_range = np.maximum(high - low, epsilon)
        clv = ((close - low) - (high - close)) / hl_range
        vol_flow = clv * np.log(volume + 1)
        vol_mean = pd.Series(vol_flow).rolling(20).mean().fillna(0).values
        vol_std = pd.Series(vol_flow).rolling(20).std().replace(0, 1).values
        features[:, 8] = np.clip((vol_flow - vol_mean) / (vol_std + epsilon), -5, 5)

        # 7. Order Flow Imbalance
        features[:, 9] = clv  # Raw CLV as order flow proxy

        # 8. Trend Strength (ADX proxy)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)),
                                                np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().fillna(epsilon).values
        dm_plus = np.maximum(high - np.roll(high, 1), 0)
        dm_minus = np.maximum(np.roll(low, 1) - low, 0)
        di_plus = pd.Series(dm_plus / (atr + epsilon)).rolling(14).mean().fillna(0).values
        di_minus = pd.Series(dm_minus / (atr + epsilon)).rolling(14).mean().fillna(0).values
        features[:, 10] = (di_plus - di_minus) / (di_plus + di_minus + epsilon)  # DX normalized

        # 9. Price Position (mean reversion signal)
        features[:, 11] = (close - ma20) / (ma20 + epsilon)

        # Handle any remaining NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)

        return features

    def get_dt_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute time delta series (for volume-clock integration).

        Returns array of time deltas normalized by average.
        """
        if not hasattr(df.index, 'to_pydatetime'):
            # Numeric index - assume uniform spacing
            return np.ones(len(df))

        try:
            times = pd.to_datetime(df.index)
            deltas = times.diff().dt.total_seconds().fillna(86400).values
            # Normalize by median
            median_dt = np.median(deltas[deltas > 0])
            if median_dt > 0:
                return deltas / median_dt
            return np.ones(len(df))
        except Exception:
            return np.ones(len(df))
