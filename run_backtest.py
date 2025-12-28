import argparse
import torch
import pandas as pd
import numpy as np
from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel

def run(args):
    print(">>> Physics Backtest (Negative Damping Enabled)...")
    model = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
    try:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    except:
        print("Model not found.")
        return
    model.eval()
    
    fetcher = DataFetcher()
    builder = FeatureBuilder()
    
    # Full Universe
    universe = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "LLY", "UNH", "WMT", "COST", "SPY", "QQQ"]
    
    print(f"{'Ticker':<6} | {'Return':<8} | {'Sharpe':<6} | {'MaxDD':<6}")
    print("-" * 35)
    
    results = []
    for sym in universe:
        df = fetcher.fetch(sym, "2023-01-01", "2025-12-31")
        if df.empty: continue
        
        feat = builder.build(df)
        if len(feat) == 0: continue
        
        # Returns calculation
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        closes = df['close'].values[20:]
        returns = (closes[1:] - closes[:-1]) / closes[:-1]
        
        with torch.no_grad():
            logits, _ = model(feat)
            probs = torch.softmax(logits, dim=1).numpy()
        
        # Strategy: Long - Short Prob
        signal = probs[:, 2] - probs[:, 0]
        signal = signal[:-1]
        
        pos = np.zeros_like(signal)
        pos[signal > args.thresh] = 1.0
        pos[signal < -args.thresh] = -0.5 # Conservative Short
        
        strat_ret = pos * returns[:len(pos)]
        
        cum = np.prod(1 + strat_ret) - 1
        ann = np.mean(strat_ret) * 252
        vol = np.std(strat_ret) * np.sqrt(252) + 1e-6
        sharpe = ann / vol
        
        # MaxDD
        eq = np.cumprod(1 + strat_ret)
        dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
        mdd = np.min(dd) if len(dd) > 0 else 0
        
        print(f"{sym:<6} | {cum*100:>7.1f}% | {sharpe:>6.2f} | {mdd*100:>6.1f}%")
        results.append(cum)
        
    print("-" * 35)
    print(f"Avg Return: {np.mean(results)*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/gauge_kernel.pt')
    parser.add_argument('--providers', type=str, default='')
    parser.add_argument('--min-trade-threshold', dest='thresh', type=float, default=0.15)
    parser.add_argument('--kernel', type=str, default='')
    parser.add_argument('--multi-scale', action='store_true')
    args = parser.parse_args()
    run(args)
