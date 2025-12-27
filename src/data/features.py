"""
Feature Engineering Module

Builds features for the physics-based kernel.
Maps market data to phase space coordinates.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class FeatureBuilder:
    """
    Builds input features for the Syntropy Quant Kernel.

    Features are designed to map to:
    - Position (q): Mispricing, mean reversion signals
    - Momentum (p): Trend, order flow signals
    """

    def __init__(
        self,
        lookback_short: int = 5,
        lookback_medium: int = 20,
        lookback_long: int = 60
    ):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long

    def compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute log returns"""
        return np.diff(np.log(prices), prepend=np.log(prices[0]))

    def compute_volatility(
        self,
        returns: np.ndarray,
        window: int
    ) -> np.ndarray:
        """Compute rolling volatility"""
        vol = np.zeros_like(returns)
        for i in range(window, len(returns)):
            vol[i] = np.std(returns[i-window:i]) * np.sqrt(252)
        vol[:window] = vol[window] if len(vol) > window else 0.2
        return vol

    def compute_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Compute RSI normalized to [-1, 1]"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))

        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        for i in range(window, len(prices)):
            avg_gain[i] = np.mean(gains[i-window:i])
            avg_loss[i] = np.mean(losses[i-window:i])

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)

        # Normalize to [-1, 1]
        return (rsi - 50) / 50

    def compute_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MACD and signal line"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, signal)
        return macd, macd_signal

    def _ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute exponential moving average"""
        alpha = 2 / (window + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def compute_bollinger_position(
        self,
        prices: np.ndarray,
        window: int = 20,
        num_std: float = 2.0
    ) -> np.ndarray:
        """
        Compute position within Bollinger Bands.
        Returns value in [-1, 1] where:
        - -1 = at lower band
        - 0 = at middle (SMA)
        - 1 = at upper band
        """
        position = np.zeros_like(prices)

        for i in range(window, len(prices)):
            window_prices = prices[i-window:i]
            sma = np.mean(window_prices)
            std = np.std(window_prices)

            if std > 0:
                z_score = (prices[i] - sma) / (num_std * std)
                position[i] = np.clip(z_score, -1, 1)

        return position

    def compute_volume_clock(self, volume: np.ndarray) -> np.ndarray:
        """
        Compute volume clock (relativistic time).
        Higher volume = faster time evolution.
        """
        mean_vol = np.mean(volume)
        return volume / (mean_vol + 1e-10)

    def compute_order_flow(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """
        Estimate order flow from OHLC data.
        Positive = buying pressure, Negative = selling pressure.
        """
        # Money flow multiplier
        range_hl = high - low
        clv = np.zeros_like(close)
        numerator = (close - low) - (high - close)
        np.divide(numerator, range_hl, out=clv, where=range_hl != 0)

        # Money flow volume
        mfv = clv * volume

        return mfv / (np.abs(mfv).mean() + 1e-10)

    def compute_trend_strength(
        self,
        prices: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """
        Compute trend strength using linear regression slope.
        """
        strength = np.zeros_like(prices)

        for i in range(window, len(prices)):
            y = prices[i-window:i]
            x = np.arange(window)
            slope = np.polyfit(x, y, 1)[0]
            strength[i] = slope / (np.std(y) + 1e-10)

        return strength

    def build_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build full feature set from OHLCV data.

        Returns:
            Array of shape [n_samples, 12]
        """
        prices = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Compute all features
        returns = self.compute_returns(prices)
        vol_short = self.compute_volatility(returns, self.lookback_short)
        vol_medium = self.compute_volatility(returns, self.lookback_medium)

        rsi = self.compute_rsi(prices)
        macd, macd_signal = self.compute_macd(prices)
        bb_pos = self.compute_bollinger_position(prices)

        volume_clock = self.compute_volume_clock(volume)
        order_flow = self.compute_order_flow(high, low, prices, volume)
        trend = self.compute_trend_strength(prices)

        # Combine into feature matrix
        features = np.column_stack([
            returns,                      # 0: Raw return
            returns * 10,                 # 1: Scaled return (momentum proxy)
            vol_short,                    # 2: Short-term volatility
            vol_medium,                   # 3: Medium-term volatility
            rsi,                          # 4: RSI (mean reversion)
            macd / (np.std(macd) + 1e-8), # 5: MACD normalized
            (macd - macd_signal) / (np.std(macd) + 1e-8),  # 6: MACD histogram
            bb_pos,                       # 7: Bollinger position
            volume_clock,                 # 8: Volume clock
            order_flow,                   # 9: Order flow
            trend,                        # 10: Trend strength
            np.gradient(vol_medium)       # 11: Volatility change
        ])

        # Handle NaNs
        features = np.nan_to_num(features, 0)

        return features

    def get_dt_series(self, df: pd.DataFrame) -> np.ndarray:
        """Get time step series based on volume clock"""
        return self.compute_volume_clock(df['volume'].values)
