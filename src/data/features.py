import numpy as np
import pandas as pd
import torch
from .features_v5 import FeatureBuilderV5

class FeatureBuilder:
    """
    Bridge for Feature Engineering
    Supports both legacy v4 and new v5 feature sets.
    """
    def __init__(self):
        self.v5_builder = FeatureBuilderV5()

    def build(self, df: pd.DataFrame) -> torch.Tensor:
        """v4 Features - 4 dimensions"""
        if len(df) < 20: return torch.empty(0)
        
        # Normalize columns
        df_copy = df.copy()
        df_copy.columns = [c.lower() for c in df_copy.columns]
        
        close = df_copy['close'].values
        ret = np.diff(np.log(close + 1e-8), prepend=close[0])
        vol = pd.Series(ret).rolling(10).std().fillna(0).values
        # Volume delta
        volume = df_copy['volume'].values
        vol_delta = np.diff(np.log(volume + 1.0), prepend=np.log(volume[0] + 1.0))
        # Distance from MA
        ma = pd.Series(close).rolling(20).mean().fillna(close[0]).values
        dist = (close - ma) / (ma + 1e-8)
        
        feat = np.stack([ret, vol, vol_delta, dist], axis=1)
        return torch.tensor(feat, dtype=torch.float32)

    def build_features(self, df: pd.DataFrame) -> torch.Tensor:
        """v5 Features - 12+ dimensions"""
        # FeatureBuilderV5.build returns (features, labels)
        f_tuple = self.v5_builder.build(df)
        if isinstance(f_tuple, tuple):
            f_base = f_tuple[0]
        else:
            f_base = f_tuple
            
        if len(f_base) == 0: return torch.empty(0)
        
        # Expand features to reach the expected 14-dim (or close to it)
        # f_base is likely 6-dim [ret, vol_10, flow_z, hurst, pot, trend]
        f_base_np = f_base.numpy()
        f_diff = np.diff(f_base_np, axis=0, prepend=f_base_np[:1])
        f_combined = np.concatenate([f_base_np, f_diff], axis=1)
        
        # If still not 14, pad with zeros
        if f_combined.shape[1] < 14:
            padding = np.zeros((f_combined.shape[0], 14 - f_combined.shape[1]))
            f_combined = np.concatenate([f_combined, padding], axis=1)
        elif f_combined.shape[1] > 14:
            f_combined = f_combined[:, :14]
            
        return torch.tensor(f_combined, dtype=torch.float32)
