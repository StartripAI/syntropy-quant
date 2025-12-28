#!/usr/bin/env python3
"""
Syntropy Quant - Version Comparison Backtest
v4.0 (Physics Kernel) vs v5.0 (Gauge Field Kernel)
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel
from src.core.gauge import GaugeFieldKernel, GaugeConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('Comparison')

def run_backtest_v4(symbol, df, model_path):
    builder = FeatureBuilder()
    model = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        logger.error(f"v4.0 Model load failed: {e}")
        return None
        
    model.eval()
    
    # v4.0 features (4-dim: LogRet, Vol, Momentum, Position)
    feat = builder.build(df)
    if len(feat) == 0: return None
    
    # Process
    df_c = df.copy()
    df_c.columns = [c.lower() for c in df_c.columns]
    closes = df_c['close'].values[20:]
    returns = (closes[1:] - closes[:-1]) / closes[:-1]
    
    with torch.no_grad():
        logits, _ = model(feat)
        probs = torch.softmax(logits, dim=1).numpy()
    
    # Signal: Long - Short
    signal = probs[:, 2] - probs[:, 0]
    signal = signal[:-1]
    
    # Strategy
    pos = np.zeros_like(signal)
    pos[signal > 0.03] = 1.0
    pos[signal < -0.03] = -0.5
    
    strat_ret = pos * returns[:len(pos)]
    return strat_ret

def run_backtest_v5(symbol, df, model_path):
    builder = FeatureBuilder()
    # v5.0 uses GaugeFieldKernel with 14-dim input
    model = GaugeFieldKernel(input_dim=14)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        logger.error(f"v5.0 Model load failed: {e}")
        return None
        
    model.eval()
    
    # v5.0 features (14-dim)
    feat = builder.build_features(df)
    if feat.shape[0] == 0: return None
    
    df_c = df.copy()
    df_c.columns = [c.lower() for c in df_c.columns]
    closes = df_c['close'].values[20:]
    returns = (closes[1:] - closes[:-1]) / closes[:-1]
    
    strat_ret = []
    
    # We use process_step for v5 to get gauge signals
    for i in range(20, len(feat)-1):
        res = model.process_step(feat[i:i+1])
        signal = res.signal
        
        pos = 0
        if signal > 0.04: # v5 threshold
            pos = 1.0
        elif signal < -0.04:
            pos = -0.4 # More conservative on shorts in v5
            
        # Realized return for this step is next day's close
        # In this simple backtest, i matches returns[i-20]
        strat_ret.append(pos * returns[i-20])
        
    return np.array(strat_ret)

def analyze_returns(returns):
    if returns is None or len(returns) == 0:
        return "N/A"
    
    cum = np.prod(1 + returns) - 1
    ann = np.mean(returns) * 252
    vol = np.std(returns) * np.sqrt(252) + 1e-6
    sharpe = ann / vol
    
    eq = np.cumprod(1 + returns)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    mdd = np.min(dd) if len(dd) > 0 else 0
    
    return {
        'Return': f"{cum*100:.1f}%",
        'Sharpe': f"{sharpe:.2f}",
        'MaxDD': f"{mdd*100:.1f}%"
    }

def main():
    fetcher = DataFetcher()
    universe = ["AAPL", "NVDA", "QQQ", "YANG"]
    
    # Start/End
    start = "2024-01-01"
    end = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
    print("="*80)
    print(f"Syntropy Quant Comparison: v4.0 vs v5.0 ({start} to {end})")
    print("="*80)
    
    results = []
    
    for symbol in universe:
        logger.info(f"Processing {symbol}...")
        df = fetcher.fetch(symbol, start, end)
        if df.empty:
            logger.warning(f"No data for {symbol}")
            continue
            
        ret_v4 = run_backtest_v4(symbol, df, "models/gauge_kernel_v2.pt")
        ret_v5 = run_backtest_v5(symbol, df, "models/gauge_kernel_v5.pt")
        
        stats_v4 = analyze_returns(ret_v4)
        stats_v5 = analyze_returns(ret_v5)
        
        results.append({
            'Symbol': symbol,
            'v4': stats_v4,
            'v5': stats_v5
        })
        
    # Print Table
    print("\n" + "-"*80)
    print(f"{'Symbol':<8} | {'Version':<8} | {'Return':<10} | {'Sharpe':<8} | {'MaxDD':<8}")
    print("-"*80)
    
    for r in results:
        sym = r['Symbol']
        v4 = r['v4']
        v5 = r['v5']
        
        if v4 != "N/A":
            print(f"{sym:<8} | {'v4.0':<8} | {v4['Return']:<10} | {v4['Sharpe']:<8} | {v4['MaxDD']:<8}")
        if v5 != "N/A":
            print(f"{'':<8} | {'v5.0':<8} | {v5['Return']:<10} | {v5['Sharpe']:<8} | {v5['MaxDD']:<8}")
        print("-" * 80)

if __name__ == "__main__":
    main()
