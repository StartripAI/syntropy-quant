
import numpy as np
import pandas as pd
import torch

class FeatureBuilderV5:
    """
    v5.2 Features: Relativistic Physics Labels.
    Ensures balanced training classes even in low-volatility regimes.
    """
    def build(self, df: pd.DataFrame) -> tuple:
        if len(df) < 60: return torch.empty(0), torch.empty(0)
        
        df = df.copy()
        eps = 1e-8
        
        # --- 1. Standard Features ---
        close = df['Close'].values
        ret = np.diff(np.log(close + eps), prepend=close[0])
        
        # Volatility (Energy)
        vol_10 = pd.Series(ret).rolling(10).std().fillna(0).values
        
        # Order Flow (Momentum)
        hl_range = np.maximum(df['High'].values - df['Low'].values, eps)
        clv = ((close - df['Low'].values) - (df['High'].values - close)) / hl_range
        flow = clv * np.log(df['Volume'].values + 1.0)
        flow_z = (flow - pd.Series(flow).rolling(20).mean()) / (pd.Series(flow).rolling(20).std() + eps)
        flow_z = flow_z.fillna(0).values
        
        # --- 2. Physics Parameters ---
        
        # A. Hurst Exponent (Fractal Dimension)
        def rolling_hurst(series, window=20):
            h_vals = np.zeros(len(series)) + 0.5
            for i in range(window, len(series)):
                chunk = series[i-window:i]
                r = np.max(chunk) - np.min(chunk) + eps
                s = np.std(chunk) + eps
                if s < 1e-9: h_vals[i] = 0.5
                else: h = np.log(r/s) / np.log(window)
                    h_vals[i] = h
            return np.clip(h_vals, 0, 1)
        
        hurst = rolling_hurst(close)
        
        # B. Trend Strength (Directional Energy)
        ema_short = pd.Series(close).ewm(span=10).mean().values
        ema_long = pd.Series(close).ewm(span=50).mean().values
        trend = np.abs(ema_short - ema_long) / (ema_long + eps)
        
        # --- 3. Relativistic Physics Regime Labeling (CRITICAL FIX v5.2) ---
        # Instead of hard threshold, use dynamic quantile.
        # Identify the top 30% most "Chaotic" points in this specific history.
        chaos_score = hurst + (trend * 10)
        threshold = np.percentile(chaos_score, 70)
        
        regime_label = np.zeros_like(hurst)
        regime_label[chaos_score > threshold] = 1.0
        
        # Relative Potential
        ma50 = pd.Series(close).rolling(50).mean().fillna(close[0]).values
        pot = (close - ma50) / (ma50 + eps)

        # Stack Features (Dim=6)
        feat = np.stack([ret, vol_10, flow_z, hurst, pot, trend], axis=1)
        feat = np.clip(feat, -5.0, 5.0)
        
        return (torch.tensor(feat[50:], dtype=torch.float32), 
                torch.tensor(regime_label[50:], dtype=torch.float32))
