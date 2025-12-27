#!/usr/bin/env python3
"""
Train Gauge Field Kernel (offline or online).
"""

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.gauge import GaugeFieldKernel, GaugeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train gauge kernel")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--exclude", default="", help="Comma-separated symbols to exclude")
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--online", action="store_true", help="Online training mode")
    parser.add_argument("--save", default="models/gauge_kernel.pt")
    parser.add_argument("--neutral-band", type=float, default=0.0035)
    parser.add_argument("--forward-window", type=int, default=7)
    parser.add_argument("--providers", default="", help="Comma-separated provider priority")
    parser.add_argument("--no-adjust", action="store_true", help="Disable price adjustment")
    parser.add_argument("--energy-weight", type=float, default=0.0075)
    parser.add_argument("--no-balance-classes", dest="balance_classes", action="store_false")
    parser.set_defaults(balance_classes=True)
    return parser.parse_args()


def build_dataset(
    features: np.ndarray,
    prices: np.ndarray,
    neutral_band: float,
    forward_window: int
) -> TensorDataset:
    log_prices = np.log(prices)
    fwd = np.log(prices[forward_window:] / prices[:-forward_window])
    X = features[:-forward_window]

    labels = np.zeros_like(fwd, dtype=np.int64)
    labels[fwd > neutral_band] = 2
    labels[fwd < -neutral_band] = 0
    labels[(fwd >= -neutral_band) & (fwd <= neutral_band)] = 1

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


def train_offline(model, loader, optimizer, class_weights, energy_weight):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits, free_energy, _ = model(batch_x)
        loss = F.cross_entropy(logits, batch_y, weight=class_weights)
        loss = loss + energy_weight * free_energy.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train_online(model, dataset, optimizer, class_weights, energy_weight):
    model.train()
    total_loss = 0.0
    for x, y in dataset:
        optimizer.zero_grad()
        logits, free_energy, _ = model(x.unsqueeze(0))
        loss = F.cross_entropy(logits, y.unsqueeze(0), weight=class_weights)
        loss = loss + energy_weight * free_energy.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataset), 1)


def main():
    args = parse_args()
    exclude = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}
    if args.symbols:
        symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from src.data.fetcher import ASSET_UNIVERSE
        symbols = sorted({info.symbol for info in ASSET_UNIVERSE.values()})

    if exclude:
        symbols = [s for s in symbols if s not in exclude]

    provider_list = [p.strip() for p in args.providers.split(",") if p.strip()]
    fetcher = DataFetcher(
        cache_dir="data_cache",
        provider_priority=provider_list if provider_list else None,
        adjust_prices=not args.no_adjust
    )
    feature_builder = FeatureBuilder()

    datasets = []
    for symbol in symbols:
        df = fetcher.fetch(symbol, args.start, args.end)
        if df.empty:
            print(f"Skipping {symbol} (no data)")
            continue

        features = feature_builder.build_features(df)
        prices = df["close"].values
        if len(prices) <= args.forward_window:
            print(f"Skipping {symbol} (insufficient history)")
            continue
        datasets.append(build_dataset(features, prices, args.neutral_band, args.forward_window))

    if not datasets:
        print("No data available for training.")
        return

    all_x = torch.cat([d.tensors[0] for d in datasets], dim=0)
    all_y = torch.cat([d.tensors[1] for d in datasets], dim=0)

    feature_mean = all_x.mean(dim=0)
    feature_std = all_x.std(dim=0)
    feature_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)
    all_x = (all_x - feature_mean) / feature_std

    dataset = TensorDataset(all_x, all_y)

    config = GaugeConfig(input_dim=all_x.shape[1])
    model = GaugeFieldKernel(input_dim=all_x.shape[1], config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_weights = None
    if args.balance_classes:
        counts = torch.bincount(all_y, minlength=3).float()
        weights = counts.sum() / (counts + 1e-8)
        class_weights = weights / weights.mean()

    if args.online:
        for epoch in range(args.epochs):
            loss = train_online(model, dataset, optimizer, class_weights, args.energy_weight)
            print(f"Epoch {epoch+1}/{args.epochs} - loss: {loss:.4f}")
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for epoch in range(args.epochs):
            loss = train_offline(model, loader, optimizer, class_weights, args.energy_weight)
            print(f"Epoch {epoch+1}/{args.epochs} - loss: {loss:.4f}")

    # Calibrate signal bias on training set to avoid directional drift
    model.eval()
    bias_sum = 0.0
    bias_count = 0
    calib_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in calib_loader:
            logits, _, _ = model(batch_x)
            logit_delta = (logits[:, 2] - logits[:, 0]).mean().item()
            bias_sum += logit_delta * batch_x.size(0)
            bias_count += batch_x.size(0)
    if bias_count > 0:
        config.signal_bias = bias_sum / bias_count

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "forward_window": args.forward_window,
            "neutral_band": args.neutral_band,
        },
        args.save
    )
    print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
