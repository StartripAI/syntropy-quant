"""
Syntropy Quant v5.0 - The Principle of Least Action
Advanced Portfolio Configuration with Hierarchical Risk Parity (HRP)

Core Innovation:
- HRP weighting (removes dependence on mean-variance instability)
- Regime-aware signal thresholds
- Least-action path optimization
"""

import numpy as np
import scipy.cluster.hierarchy as sch
from typing import List, Dict

# Blacklist - consistently negative performers or extreme singularities
BLACKLIST = ['UNH', 'TGT', 'PFE', 'BMY', 'MRK', 'INTC']

# Optimized Universe
PORTFOLIO = {
    'core': ['QQQ', 'VOO', 'VTI', 'SPY', 'IWM', 'DIA'],
    'tech': ['GOOGL', 'AMZN', 'META', 'MSFT', 'NVDA', 'AAPL', 'TSLA'],
    'semis': ['AMAT', 'LRCX', 'AVGO', 'QCOM', 'AMD', 'ASML'],
    'finance': ['JPM', 'GS', 'BRK-B', 'V', 'MA'],
    'consumer': ['WMT', 'COST', 'PEP', 'KO'],
}

# Trading parameters (Regime-Aware)
TRADING_CONFIG = {
    'default_threshold': 0.04,   # Default signal threshold
    'crisis_threshold': 0.08,    # Higher bar during market stress
    'bubble_threshold': 0.06,    # Caution during bubbles
    'max_leverage': 1.2,
    'short_ratio': 0.4,
}

# Risk parameters
RISK_CONFIG = {
    'max_drawdown_limit': 0.20,
    'position_limit': 0.08,      # Max 8% in single position (Diversification)
    'sector_limit': 0.25,        # Max 25% in single sector
    'vol_target': 0.12,          # Target 12% annualized volatility
}

def get_all_symbols():
    symbols = []
    for bucket in PORTFOLIO.values():
        symbols.extend(bucket)
    return [s for s in symbols if s not in BLACKLIST]

def compute_hrp_weights(cov: np.ndarray, symbols: List[str]) -> Dict[str, float]:
    """
    Hierarchical Risk Parity (HRP) Implementation.
    Robust to estimation errors in covariance matrix.
    """
    if cov.shape[0] <= 1:
        return {symbols[0]: 1.0} if symbols else {}
        
    # 1. Clustering
    vols = np.sqrt(np.diag(cov))
    vols[vols < 1e-8] = 1e-8 # Prevent division by zero
    corr = cov / np.outer(vols, vols)
    corr = np.clip(corr, -1, 1) # Ensure within range
    corr[np.isnan(corr)] = 0
    np.fill_diagonal(corr, 1)
    
    dist = np.sqrt(np.maximum(0, 0.5 * (1 - corr)))
    dist[np.isnan(dist)] = 0
    
    # Check for finite values
    if not np.all(np.isfinite(dist)):
        # Fallback if distance matrix is still bad
        return {s: 1.0/len(symbols) for s in symbols}

    link = sch.linkage(dist, 'single')
    sort_idx = sch.leaves_list(link)
    sorted_symbols = [symbols[i] for i in sort_idx]
    
    # 2. Recursive Bisection
    weights = np.ones(len(symbols))
    
    def get_cluster_var(c_cov):
        # Inverse variance weighting within cluster
        inv_diag = 1.0 / np.diag(c_cov)
        w = inv_diag / inv_diag.sum()
        return np.dot(w, np.dot(c_cov, w))
        
    queue = [sort_idx.tolist()]
    while len(queue) > 0:
        cluster = queue.pop(0)
        if len(cluster) <= 1: continue
        
        # Split cluster
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]
        
        # Compute variances
        var_l = get_cluster_var(cov[np.ix_(left, left)])
        var_r = get_cluster_var(cov[np.ix_(right, right)])
        
        # Compute allocation factor (alpha)
        alpha = 1 - var_l / (var_l + var_r)
        
        # Update weights
        for i in left: weights[i] *= alpha
        for i in right: weights[i] *= (1 - alpha)
        
        queue.append(left)
        queue.append(right)
        
    return {symbols[i]: float(weights[i]) for i in range(len(symbols))}

def get_final_weights(cov: np.ndarray = None, symbols: List[str] = None):
    """
    Get weights. If cov is provided, use HRP. 
    Otherwise fallback to equal weight / bucketed weight.
    """
    if symbols is None:
        symbols = get_all_symbols()
        
    if cov is not None:
        return compute_hrp_weights(cov, symbols)
        
    # Default: Sector-balanced equal weighting
    weights = {}
    num_buckets = len(PORTFOLIO)
    for bucket_name, bucket in PORTFOLIO.items():
        active_in_bucket = [s for s in bucket if s not in BLACKLIST]
        if not active_in_bucket: continue
        
        b_weight = 1.0 / num_buckets
        for s in active_in_bucket:
            weights[s] = b_weight / len(active_in_bucket)
            
    return weights

if __name__ == '__main__':
    print("Syntropy Quant v5.0 - Default Weights (Equal-Sector Balance):")
    print("-" * 55)
    weights = get_final_weights()
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"{sym:8} {w*100:6.2f}%")
    print("-" * 55)
    print(f"Total:   {sum(weights.values())*100:6.2f}%")
