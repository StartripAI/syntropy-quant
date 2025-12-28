import torch
import pandas as pd
import numpy as np
from src.core.hybrid.kernel_v5 import SyntropyQuantKernelV5
from src.data.features_v5 import FeatureBuilderV5
from src.data.fetcher import DataFetcher

def run():
    print("\n===========================================================")
    print(" SYNTROPY QUANT v5.0 | HYBRID REGIME BACKTEST")
    print("===========================================================")
    
    model = SyntropyQuantKernelV5(input_dim=7)
    try:
        model.load_state_dict(torch.load("models/syntropy_v5.pt", map_location='cpu'))
    except:
        print("Model not found. Train first.")
        return
    model.eval()
    
    fetcher = DataFetcher()
    builder = FeatureBuilderV5()
    
    universe = {
        "INDEX": ["SPY", "QQQ", "IWM"],
        "TECH": ["NVDA", "MSFT", "META", "TSLA"],
        "PHARMA": ["LLY", "UNH"],
        "CONS": ["WMT", "KO", "MCD"]
    }
    
    print(f"{'Ticker':<8} | {'Ret%':<8} | {'Sharpe':<6} | {'MaxDD':<6} | {'Regime (Chaos%)':<15}")
    print("-" * 65)
    
    total_ret = []
    
    for cat, tickers in universe.items():
        for sym in tickers:
            df = fetcher.fetch(sym, "2023-01-01", "2025-12-31")
            if df.empty: continue
            feat = builder.build(df)
            if len(feat) == 0: continue
            
            closes = df['Close'].values[50:]
            returns = (closes[1:] - closes[:-1]) / closes[:-1]
            
            with torch.no_grad():
                out = model(feat)
                logits = out['logits']
                regimes = out['regime'].numpy().flatten()
            
            probs = torch.softmax(logits, dim=1).numpy()
            sig = probs[:, 2] - probs[:, 0]
            sig = sig[:-1]
            regimes = regimes[:-1]
            
            # Adaptive Strategy based on Physics Regime
            pos = np.zeros_like(sig)
            
            for i in range(len(sig)):
                r = regimes[i] # 0=Symplectic, 1=Gauge
                s = sig[i]
                
                # In Gauge Field (Chaos/Trend), we lower threshold to capture momentum
                # In Symplectic (Stable), we increase threshold to filter noise
                thresh = 0.20 * (1.0 - r * 0.6) 
                
                if s > thresh: pos[i] = 1.0
                elif s < -thresh: pos[i] = -1.0
            
            strat_ret = pos * returns[:len(pos)]
            cum = np.prod(1 + strat_ret) - 1
            ann = np.mean(strat_ret) * 252
            vol = np.std(strat_ret) * np.sqrt(252) + 1e-6
            sharpe = ann / vol
            
            eq = np.cumprod(1 + strat_ret)
            dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
            mdd = np.min(dd)
            
            avg_reg = np.mean(regimes) * 100
            reg_label = "GAUGE" if avg_reg > 50 else "SYMPL"
            
            print(f"{sym:<8} | {cum*100:>7.1f}% | {sharpe:>6.2f} | {mdd*100:>6.1f}% | {reg_label:<5} ({avg_reg:.0f}%)")
            total_ret.append(cum)

    print("-" * 65)
    print(f"Avg Portfolio Return: {np.mean(total_ret)*100:.1f}%")

if __name__ == "__main__":
    run()
