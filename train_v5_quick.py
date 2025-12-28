#!/usr/bin/env python3
"""
Syntropy Quant v5.0 - Quick Training Initialization
Generates a valid state-dict for the GaugeFieldKernel.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.core.gauge import GaugeFieldKernel, GaugeConfig
from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder

def initialize_v5_model(save_path="models/gauge_kernel_v5.pt"):
    print(f"Syntropy v5.0 - Intelligence Bootstrap")
    
    # 1. Setup Architecture
    config = GaugeConfig(input_dim=14)
    model = GaugeFieldKernel(input_dim=14, config=config)
    
    # 2. Setup Data
    fetcher = DataFetcher()
    builder = FeatureBuilder()
    symbols = ['QQQ', 'NVDA', 'AAPL', 'YANG', 'SPY']
    
    data_list, target_list = [], []
    for s in symbols:
        print(f"Fetching training data for {s}...")
        df = fetcher.fetch(s, "2023-01-01", "2024-12-01")
        if df.empty: continue
        feat = builder.build_features(df)
        
        # Labels: return sign
        closes = df['close'].values if 'close' in df.columns else df['Close'].values
        ret = np.diff(closes) / closes[:-1]
        labels = np.ones(len(ret))
        labels[ret > 0.002] = 2
        labels[ret < -0.002] = 0
        
        # Align
        warmup = 60
        X = feat[warmup:-1]
        Y = labels[warmup:]
        
        data_list.append(torch.tensor(X, dtype=torch.float32))
        target_list.append(torch.tensor(Y, dtype=torch.long))
    
    if not data_list:
        print("No data found. Aborting.")
        return

    X_train = torch.cat(data_list)
    Y_train = torch.cat(target_list)
    
    # 3. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(X_train)} samples for 50 epochs...")
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits, free_energy, _ = model(X_train)
        loss = criterion(logits, Y_train) + free_energy.mean() * 0.05
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
        
    # 4. Save
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"SUCCESS: Model saved to {save_path}")

if __name__ == "__main__":
    initialize_v5_model()
