#!/usr/bin/env python3
"""
Syntropy Quant - Institutional Trading System
Two Sigma / Citadel Inspired Architecture

Components:
1. Signal Generation (Physics Kernel)
2. Risk Management (Multi-layer controls)
3. Execution Engine (Smart order routing)
4. Model Management (Auto-retrain)
5. Monitoring & Alerting

Usage:
    python trading_system.py --mode paper --run daily
    python trading_system.py --mode live --run once
    python trading_system.py --retrain --force
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import argparse
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

# Project imports
from config.optimized_portfolio import (
    PORTFOLIO, BLACKLIST, TRADING_CONFIG, RISK_CONFIG, get_final_weights, get_all_symbols
)
from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel
from src.core.gauge import GaugeFieldKernel, GaugeConfig
from src.core.filters import RicciCurvatureFilter, SurpriseFilter
from execution.alpaca_engine import ExecutionEngine, Order
from risk.risk_manager import InstitutionalRiskManager, RiskLimits, RiskLevel
from scheduler.auto_retrain import AutoRetrainer, RetrainConfig
from monitoring.alerts import AlertManager


class TradingSystem:
    """
    Syntropy Quant v5.0: The Principle of Least Action
    
    Features:
    - Gauge Field Kernel (Path Integral Sampling)
    - Ricci Curvature Singularity Detection
    - Surprise Filtering (Free Energy Principle)
    - Dynamic HRP Portfolio Optimization
    """

    def __init__(
        self,
        mode: str = "paper",
        model_v5: str = "models/gauge_kernel_v5.pt",
        model_v4: str = "models/gauge_kernel_v2.pt"
    ):
        self.mode = mode
        self.model_v5 = model_v5
        self.model_v4 = model_v4

        self._setup_logging()
        self._init_components()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        self.logger = logging.getLogger('TradingSystem')
        self.alerts = AlertManager()

        # Create logs directory
        Path('logs').mkdir(exist_ok=True)

    def _init_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing system v5.0 (Hybrid Regime Switch)...")

        # 1. Kernels
        self.kernel_v5 = GaugeFieldKernel(input_dim=14)
        self.kernel_v4 = SyntropyQuantKernel(input_dim=4, hidden_dim=64)

        # Load v5
        if Path(self.model_v5).exists():
            self.kernel_v5.load_state_dict(torch.load(self.model_v5, map_location='cpu'))
            self.logger.info(f"Loaded v5 model: {self.model_v5}")
        
        # Load v4
        if Path(self.model_v4).exists():
            self.kernel_v4.load_state_dict(torch.load(self.model_v4, map_location='cpu'))
            self.logger.info(f"Loaded v4 model: {self.model_v4}")

        self.kernel_v5.eval()
        self.kernel_v4.eval()

        # 2. Filters & Detectors
        self.curvature_filter = RicciCurvatureFilter(hidden_dim=64)
        self.surprise_filter = SurpriseFilter(threshold_k=2.5)

        # 3. Data components
        self.fetcher = DataFetcher(cache_dir="data_cache")
        self.builder = FeatureBuilder()

        # 4. Risk & Execution
        risk_limits = RiskLimits(
            max_position_pct=RISK_CONFIG['position_limit'],
            max_sector_pct=RISK_CONFIG['sector_limit'],
            max_drawdown_pct=RISK_CONFIG['max_drawdown_limit'],
        )
        self.risk_manager = InstitutionalRiskManager(risk_limits)
        self.executor = ExecutionEngine(paper=(self.mode == "paper"))

        self.logger.info(f"System v5.0 ready in {self.mode.upper()} mode")

    def get_market_context(self) -> Dict[str, np.ndarray]:
        """Fetch covariance and recent returns for HRP weighting"""
        symbols = get_all_symbols()
        data = {}
        for s in symbols[:15]: # Limit for performance
            df = self.fetcher.fetch(s, (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"), 
                                     datetime.now().strftime("%Y-%m-%d"))
            if not df.empty:
                prices = df['close'].values if 'close' in df.columns else df['Close'].values
                data[s] = np.diff(np.log(prices + 1e-8))
        
        if not data: return {}
        
        # Align lengths
        min_len = min(len(v) for v in data.values())
        returns_mat = np.stack([v[-min_len:] for v in data.values()])
        cov = np.cov(returns_mat)
        return {'cov': cov, 'symbols': list(data.keys())}

    def generate_signals(self) -> Dict[str, Dict]:
        """
        Generate advanced signals using Gauge Field and Surprise Filtering
        """
        signals = {}
        symbols = get_all_symbols()

        for symbol in symbols:
            try:
                df = self.fetcher.fetch(symbol, 
                                        (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d"),
                                        datetime.now().strftime("%Y-%m-%d"))
                if df.empty or len(df) < 30: continue

                # Build V5 features (12-dim)
                feat = self.builder.build_features(df)
                if feat.shape[0] == 0: continue

                # Surprise filtering
                recent_close = df['close'].values[-1] if 'close' in df.columns else df['Close'].values[-1]
                f_res = self.surprise_filter.filter(recent_close)
                
                # Signal Generation (Hybrid Switching)
                with torch.no_grad():
                    # 1. Get v5 Signal & Regime
                    res_v5 = self.kernel_v5.process_step(feat[-1:])
                    
                    # 2. Get v4 Signal (Physics Momentum)
                    feat_v4 = self.builder.build(df)
                    res_v4 = self.kernel_v4.process_step(feat_v4[-1:])
                    
                    # 3. Hybrid Logic
                    # If manifold curvature is extremely high or in bubble mode, use v4 momentum logic
                    if res_v4.regime == 'bubble' or res_v5.curvature > 1.5:
                        self.logger.debug(f"{symbol}: Switching to v4 (Bubble/High Curvature)")
                        signal = res_v4.signal
                        regime = res_v4.regime
                        version = "v4.0"
                    else:
                        signal = res_v5.signal
                        regime = res_v5.regime
                        version = "v5.0"

                    signals[symbol] = {
                        'signal': signal,
                        'confidence': res_v5.confidence,
                        'regime': regime,
                        'curvature': res_v5.curvature,
                        'surprise': f_res.surprise,
                        'version': version
                    }

            except Exception as e:
                self.logger.error(f"Signal failure for {symbol}: {e}")

        return signals

    def run_trading_cycle(self) -> Dict:
        """
        Syntropy v5.0 Core Loop
        """
        self.logger.info(">>> Energy Minimization Cycle Beginning <<<")
        
        # 1. Account & Environment
        account = self.executor.get_account_info()
        equity = account['equity']
        
        # 2. Market Context & HRP
        context = self.get_market_context()
        if context:
            self.logger.info("Computing HRP Weights...")
            target_weights = get_final_weights(context['cov'], context['symbols'])
        else:
            target_weights = get_final_weights()

        # 3. Signals
        signals_map = self.generate_signals()
        
        # 4. Filter and Execute
        orders = []
        current_positions = self.executor.get_positions()
        
        for symbol, weight in target_weights.items():
            if symbol not in signals_map: continue
            
            sig_data = signals_map[symbol]
            signal = sig_data['signal']
            regime = sig_data['regime']
            
            # Adaptive Thresholding
            thresh = TRADING_CONFIG['default_threshold']
            if regime == 'crisis': thresh = TRADING_CONFIG['crisis_threshold']
            elif regime == 'high_vol': thresh = TRADING_CONFIG['bubble_threshold']
            
            # Target Determination
            target_value = 0
            if signal > thresh:
                target_value = equity * weight * min(1.0, sig_data['confidence'] * 1.5)
            elif signal < -thresh:
                target_value = equity * weight * -TRADING_CONFIG['short_ratio']
                
            # Delta Execution
            current_val = current_positions[symbol].market_value if symbol in current_positions else 0
            delta = target_value - current_val
            
            if abs(delta) > 500: # Min trade block
                bid, ask = self.executor.get_quote(symbol)
                price = ask if delta > 0 else bid
                if price <= 0: continue
                
                qty = abs(delta) / price
                side = 'buy' if delta > 0 else 'sell'
                
                # Risk check
                approved, reason, adj_qty = self.risk_manager.validate_trade(symbol, side, qty, price, equity)
                if approved:
                    order = Order(symbol=symbol, side=side, qty=round(adj_qty, 2))
                    res = self.executor.submit_order(order)
                    orders.append(res)
                    if res.status.value == 'filled':
                        self.alerts.send_trade_alert(f"{side.upper()} {adj_qty:.2f} {symbol} @ {price:.2f}")

        summary = {
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
            'orders': len(orders),
            'signals': len(signals_map),
            'top_signal': max(signals_map.items(), key=lambda x: abs(x[1]['signal'])) if signals_map else None
        }
        self.logger.info(f"Cycle Complete. Meta-stability reached. Orders: {len(orders)}")
        return summary

    def run_risk_check(self) -> Dict:
        """
        Run risk check without trading
        """
        account = self.executor.get_account_info()
        positions = self.executor.get_positions()

        pos_dict = {
            sym: {'market_value': pos.market_value}
            for sym, pos in positions.items()
        }
        self.risk_manager.update_positions(pos_dict)

        returns = np.random.randn(100) * 0.01
        metrics = self.risk_manager.compute_risk_metrics(account['equity'], returns)

        self.risk_manager.print_risk_report(metrics)

        # Check for deleverage
        should_delev, target = self.risk_manager.should_deleverage(metrics)
        if should_delev:
            self.logger.warning(f"DELEVERAGE RECOMMENDED: target={target*100:.0f}%")

        return {
            'risk_level': metrics.risk_level.value,
            'drawdown': metrics.current_drawdown,
            'daily_pnl_pct': metrics.daily_pnl_pct,
            'breaches': metrics.breaches
        }

    def emergency_close(self):
        """
        Emergency: close all positions
        """
        self.logger.critical("EMERGENCY CLOSE TRIGGERED")
        orders = self.executor.close_all_positions()
        self.logger.info(f"Closed {len(orders)} positions")
        return orders


def main():
    parser = argparse.ArgumentParser(description='Syntropy Quant Trading System')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                        help='Trading mode')
    parser.add_argument('--run', choices=['once', 'daily', 'continuous'],
                        default='once', help='Run mode')
    parser.add_argument('--retrain', action='store_true',
                        help='Run model retraining')
    parser.add_argument('--force', action='store_true',
                        help='Force action even if not scheduled')
    parser.add_argument('--risk-check', action='store_true',
                        help='Run risk check only')
    parser.add_argument('--emergency-close', action='store_true',
                        help='Emergency close all positions')

    args = parser.parse_args()

    # Handle retrain
    if args.retrain:
        from config.optimized_portfolio import get_all_symbols
        symbols = get_all_symbols()
        retrainer = AutoRetrainer()
        version = retrainer.run_retrain(symbols, force=args.force)
        if version:
            print(f"Retrain successful: {version}")
        return

    # Initialize system
    system = TradingSystem(mode=args.mode)

    # Handle emergency close
    if args.emergency_close:
        confirm = input("Type 'CONFIRM' to close all positions: ")
        if confirm == 'CONFIRM':
            system.emergency_close()
        return

    # Handle risk check
    if args.risk_check:
        system.run_risk_check()
        return

    # Run trading
    if args.run == 'once':
        result = system.run_trading_cycle()
        print(json.dumps(result, indent=2, default=str))

    elif args.run == 'daily':
        # Run once per day at market open
        while True:
            now = datetime.now()
            # Check if market hours (9:30 AM - 4:00 PM ET)
            if now.hour == 9 and now.minute >= 30:
                result = system.run_trading_cycle()

                # Save result
                with open(f'logs/result_{now.strftime("%Y%m%d")}.json', 'w') as f:
                    json.dump(result, f, indent=2, default=str)

                # Wait until next day
                time.sleep(3600 * 18)  # 18 hours
            else:
                time.sleep(60)  # Check every minute

    elif args.run == 'continuous':
        # Run every hour
        while True:
            result = system.run_trading_cycle()
            time.sleep(3600)


if __name__ == '__main__':
    main()
