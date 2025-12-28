#!/usr/bin/env python3
"""
UNH 单独分析 - 为什么表现差？
"""
import torch
import pandas as pd
import numpy as np
from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel
import matplotlib.pyplot as plt

def analyze_unh():
    print("=" * 60)
    print("UNH 单独分析")
    print("=" * 60)
    
    # 加载模型
    model = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
    try:
        model.load_state_dict(torch.load('models/gauge_kernel.pt', map_location='cpu'))
        print("✅ 模型加载成功")
    except:
        print("❌ 模型加载失败")
        return
    
    model.eval()
    
    # 获取数据
    fetcher = DataFetcher()
    builder = FeatureBuilder()
    
    print("\n1. 获取 UNH 数据...")
    df = fetcher.fetch("UNH", "2023-01-01", "2025-12-31")
    if df.empty:
        print("❌ 无法获取数据")
        return
    
    print(f"   数据点: {len(df)}")
    df.columns = [c.lower() for c in df.columns]
    
    # 构建特征
    feat = builder.build(df)
    if len(feat) == 0:
        print("❌ 无法构建特征")
        return
    
    print(f"   特征维度: {feat.shape}")
    
    # 计算收益
    closes = df['close'].values[20:]
    returns = (closes[1:] - closes[:-1]) / closes[:-1]
    
    # 模型预测
    print("\n2. 模型预测分析...")
    with torch.no_grad():
        logits, gamma = model(feat)
        probs = torch.softmax(logits, dim=1).numpy()
    
    # 信号分析
    signal = probs[:, 2] - probs[:, 0]  # Long - Short
    signal = signal[:-1]
    
    # 不同阈值下的表现
    print("\n3. 不同阈值下的表现:")
    print("-" * 60)
    print(f"{'Threshold':<12} | {'Return':<10} | {'Sharpe':<8} | {'MaxDD':<8} | {'Trades':<8}")
    print("-" * 60)
    
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
    for thresh in thresholds:
        pos = np.zeros_like(signal)
        pos[signal > thresh] = 1.0
        pos[signal < -thresh] = -0.5
        
        strat_ret = pos * returns[:len(pos)]
        cum = np.prod(1 + strat_ret) - 1
        ann = np.mean(strat_ret) * 252
        vol = np.std(strat_ret) * np.sqrt(252) + 1e-6
        sharpe = ann / vol
        
        eq = np.cumprod(1 + strat_ret)
        dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
        mdd = np.min(dd) if len(dd) > 0 else 0
        
        trades = np.sum(np.abs(np.diff(pos)) > 0)
        
        print(f"{thresh:<12.2f} | {cum*100:>9.1f}% | {sharpe:>7.2f} | {mdd*100:>7.1f}% | {trades:>7}")
    
    # 信号分布分析
    print("\n4. 信号分布:")
    print(f"   平均信号: {signal.mean():.3f}")
    print(f"   信号标准差: {signal.std():.3f}")
    print(f"   信号范围: [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"   正信号比例: {(signal > 0).mean()*100:.1f}%")
    print(f"   强信号 (>0.2): {(np.abs(signal) > 0.2).sum()} / {len(signal)}")
    
    # Gamma (阻尼) 分析
    print("\n5. 物理参数分析:")
    gamma_vals = gamma.mean(dim=1).numpy()[:-1]
    print(f"   平均 Gamma (阻尼): {gamma_vals.mean():.3f}")
    print(f"   Gamma 范围: [{gamma_vals.min():.3f}, {gamma_vals.max():.3f}]")
    print(f"   负阻尼比例: {(gamma_vals < 0).mean()*100:.1f}% (能量注入)")
    print(f"   正阻尼比例: {(gamma_vals > 0).mean()*100:.1f}% (能量耗散)")
    
    # 市场表现对比
    print("\n6. 市场表现:")
    buy_hold = (closes[-1] / closes[0] - 1) * 100
    print(f"   Buy & Hold: {buy_hold:.1f}%")
    best_strat = np.prod(1 + (np.sign(signal) * (np.abs(signal) > 0.1)) * returns[:len(signal)]) - 1
    print(f"   策略最佳 (thresh=0.1): {best_strat*100:.1f}%")
    
    # 建议
    print("\n7. 建议:")
    if signal.std() < 0.1:
        print("   ⚠️  信号波动太小，模型对UNH不够敏感")
    if gamma_vals.mean() > 0.5:
        print("   ⚠️  阻尼过高，模型认为市场过于有效（均值回归）")
    if buy_hold < 0:
        print("   ⚠️  Buy & Hold 本身负收益，市场环境不利")
    print("   💡 建议: 考虑排除UNH或使用更低的threshold")

if __name__ == "__main__":
    analyze_unh()

