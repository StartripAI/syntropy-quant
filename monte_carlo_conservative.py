#!/usr/bin/env python3
"""
CONSERVATIVE Monte Carlo Simulation
Fixed version with:
- No daily compounding (reset capital each day)
- Strict position limits
- Conservative signal scaling
- Daily stop-loss
"""
import argparse
import torch
import numpy as np
from typing import Dict
import multiprocessing as mp
import time

from src.core.gauge import GaugeFieldKernel, GaugeConfig


class ConservativeSimulator:
    """
    Conservative Monte Carlo simulator with realistic constraints.
    """
    
    def __init__(
        self,
        model_path: str,
        initial_capital: float = 100000,
        trading_days: int = 20,
        trades_per_day: int = 10,
    ):
        self.initial_capital = initial_capital
        self.trading_days = trading_days
        self.trades_per_day = trades_per_day
        
        # Load GaugeFieldKernel with CONSERVATIVE config
        self.config = GaugeConfig(
            input_dim=14,
            signal_scale=1.0,  # Much smaller (default 4.5)
            signal_bias=0.0,
            free_energy_scale=0.08,
            curvature_scale=0.1,
            regime_thresholds=(0.7, 1.4),
        )
        self.model = GaugeFieldKernel(input_dim=14, config=self.config)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ GaugeFieldKernel loaded (CONSERVATIVE mode)")
            print(f"   Signal scale: {self.config.signal_scale} (reduced from 4.5)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        self.model.eval()
    
    def generate_features(self, price: float, day: int) -> np.ndarray:
        """Generate 14-dim features."""
        daily_vol = 0.02
        log_return = np.random.normal(0.001, daily_vol)
        momentum = (price - 100) / 100
        volatility = daily_vol * (1 + np.random.normal(0, 0.2))
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        volume_change = np.random.normal(0, 0.3)
        volume_ma_ratio = np.random.uniform(0.8, 1.2)
        order_flow = np.random.normal(0, 0.5)
        vwap_deviation = np.random.normal(0, 0.01)
        rsi = np.random.uniform(30, 70) / 100
        macd = np.random.normal(0, 0.5)
        bollinger_position = np.random.uniform(-1, 1)
        atr = daily_vol * np.random.uniform(0.8, 1.2)
        market_regime = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        time_of_day = np.sin(2 * np.pi * day / 5)
        
        return np.array([
            log_return, momentum, volatility, trend,
            volume_change, volume_ma_ratio, order_flow, vwap_deviation,
            rsi, macd, bollinger_position, atr,
            market_regime, time_of_day,
        ])
    
    def simulate_day(self, symbol: str, day: int, price: float) -> float:
        """
        Simulate a single trading day.
        Returns daily P&L.
        
        KEY FIX: No compounding - each day is independent.
        """
        capital = self.initial_capital
        position = 0.0
        daily_pnl = 0.0
        
        # Generate features
        features = self.generate_features(price, day)
        
        # Model prediction
        with torch.no_grad():
            output = self.model.process_step(features, dt=1.0)
            signal = output.signal
            confidence = output.confidence
            curvature = output.curvature
            regime = output.regime
        
        # Risk-based position sizing
        risk_multiplier = 1.0
        if regime == 'crisis':
            risk_multiplier = 0.1  # Very conservative in crisis
        elif regime == 'high_vol':
            risk_multiplier = 0.3
        elif curvature > 0.5:
            risk_multiplier = 0.5
        
        # Max position: 5% of capital (very conservative)
        max_position_value = capital * 0.05
        position_size = abs(signal) * confidence * risk_multiplier
        position_size = min(position_size, 0.05)  # Max 5% position
        
        # Only trade on strong signals
        trade_threshold = 0.1  # Higher threshold
        
        if abs(signal) > trade_threshold:
            # Execute ONE trade per day
            side = 'buy' if signal > 0 else 'sell'
            trade_value = capital * position_size
            trade_value = min(trade_value, max_position_value)
            
            # Simulate execution
            execution_delay_ms = np.random.uniform(150, 700)
            price_impact = (execution_delay_ms / 1000) * 0.02 * price * np.random.choice([-1, 1])
            execution_price = price * (1 + price_impact)
            
            # Slippage + Commission
            slippage = execution_price * 0.0005  # 5 bps
            commission = trade_value * 0.0001  # 1 bps
            
            if side == 'buy':
                shares = trade_value / (execution_price + slippage)
                position = shares
                entry_price = execution_price + slippage
            else:  # sell
                shares = trade_value / (execution_price - slippage)
                position = -shares
                entry_price = execution_price - slippage
            
            # Close at end of day (no overnight risk)
            if position > 0:
                exit_price = price * (1 + np.random.normal(0.0005, 0.01))  # Day-end price move
                pnl = position * (exit_price - entry_price) - commission
                daily_pnl = pnl
            else:
                exit_price = price * (1 + np.random.normal(-0.0005, 0.01))
                pnl = abs(position) * (entry_price - exit_price) - commission
                daily_pnl = pnl
        else:
            # No trade
            daily_pnl = 0.0
        
        # Daily stop-loss: limit loss to 1% of capital
        # Also cap daily profit to 5% to prevent extreme outliers
        daily_pnl = max(daily_pnl, -self.initial_capital * 0.01)
        daily_pnl = min(daily_pnl, self.initial_capital * 0.05)  # Max 5% daily gain
        
        return daily_pnl
    
    def simulate_session(self, symbol: str, seed: int = None) -> Dict:
        """
        Simulate a 20-day session.
        KEY FIX: Each day is independent, no compounding.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        daily_pnls = []
        
        for day in range(self.trading_days):
            # Simulate price movement
            if day == 0:
                price = 100.0
            else:
                price_change = np.random.normal(0.0005, 0.02)  # 0.05% drift, 2% vol
                price = price * (1 + price_change)
            
            # Simulate this day independently
            daily_pnl = self.simulate_day(symbol, day, price)
            daily_pnls.append(daily_pnl)
        
        # Total P&L = sum of daily P&L (not compounding)
        total_pnl = sum(daily_pnls)
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Sharpe ratio
        if len(daily_pnls) > 1:
            mean_return = np.mean(daily_pnls) / self.initial_capital
            std_return = np.std(daily_pnls) / self.initial_capital
            sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        equity_curve = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / (self.initial_capital + running_max + 1e-8)
        max_dd = drawdown.min()
        
        # Win rate
        win_count = sum(1 for pnl in daily_pnls if pnl > 0)
        win_rate = win_count / len(daily_pnls) if len(daily_pnls) > 0 else 0
        
        # Trade count (days with trades)
        trade_days = sum(1 for pnl in daily_pnls if abs(pnl) > 1)  # Assume trades have >$1 impact
        
        return {
            'final_pnl': total_pnl,
            'final_pnl_pct': total_pnl_pct,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_trades': trade_days,
            'win_rate': win_rate,
            'daily_pnls': daily_pnls,
        }
    
    def run_monte_carlo(self, symbol: str, n_simulations: int, n_workers: int = None) -> Dict:
        """Run Monte Carlo simulation."""
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)
        
        print(f"🚀 CONSERVATIVE Monte Carlo Simulation")
        print(f"   Symbol: {symbol}")
        print(f"   Simulations: {n_simulations:,}")
        print(f"   Trading Days: {self.trading_days}")
        print(f"   Initial Capital: ${self.initial_capital:,.0f}")
        print(f"   Signal Scale: {self.config.signal_scale} (conservative)")
        print(f"   Max Position: 5%")
        print(f"   Daily Stop-Loss: 1%")
        print(f"   Workers: {n_workers}")
        print()
        
        start_time = time.time()
        
        # Use functools.partial to avoid lambda pickling issue
        from functools import partial
        sim_func = partial(self.simulate_session, symbol)
        
        with mp.Pool(n_workers) as pool:
            results = pool.map(sim_func, range(n_simulations))
        
        elapsed = time.time() - start_time
        
        pnls = [r['final_pnl'] for r in results]
        pnl_pcts = [r['final_pnl_pct'] for r in results]
        sharpes = [r['sharpe'] for r in results]
        max_dds = [r['max_dd'] for r in results]
        trades = [r['total_trades'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        
        stats = {
            'n_simulations': n_simulations,
            'elapsed_time': elapsed,
            'mean_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'std_pnl': np.std(pnls),
            'min_pnl': np.min(pnls),
            'max_pnl': np.max(pnls),
            'mean_pnl_pct': np.mean(pnl_pcts),
            'median_pnl_pct': np.median(pnl_pcts),
            'mean_sharpe': np.mean(sharpes),
            'mean_max_dd': np.mean(max_dds),
            'worst_max_dd': np.min(max_dds),
            'mean_trades': np.mean(trades),
            'mean_win_rate': np.mean(win_rates),
            'prob_profit': np.mean([p > 0 for p in pnls]),
            'prob_loss': np.mean([p < 0 for p in pnls]),
            'pnl_5th': np.percentile(pnls, 5),
            'pnl_25th': np.percentile(pnls, 25),
            'pnl_50th': np.percentile(pnls, 50),
            'pnl_75th': np.percentile(pnls, 75),
            'pnl_95th': np.percentile(pnls, 95),
        }
        
        return stats


def print_results(stats: Dict, symbol: str):
    """Print formatted results"""
    print("=" * 80)
    print(f"  CONSERVATIVE MONTE CARLO RESULTS - {symbol}")
    print("=" * 80)
    print()
    
    print(f"📊 Simulation:")
    print(f"   Simulations: {stats['n_simulations']:,}")
    print(f"   Elapsed: {stats['elapsed_time']:.1f}s")
    print()
    
    print(f"💰 Expected P&L (20 days):")
    print(f"   Mean:   ${stats['mean_pnl']:>10,.2f} ({stats['mean_pnl_pct']:>6.2f}%)")
    print(f"   Median: ${stats['median_pnl']:>10,.2f}")
    print(f"   Std:    ${stats['std_pnl']:>10,.2f}")
    print(f"   Min:    ${stats['min_pnl']:>10,.2f}")
    print(f"   Max:    ${stats['max_pnl']:>10,.2f}")
    print()
    
    print(f"📈 Distribution:")
    print(f"   5th:  ${stats['pnl_5th']:>10,.2f}")
    print(f"   25th: ${stats['pnl_25th']:>10,.2f}")
    print(f"   50th: ${stats['pnl_50th']:>10,.2f}")
    print(f"   75th: ${stats['pnl_75th']:>10,.2f}")
    print(f"   95th: ${stats['pnl_95th']:>10,.2f}")
    print()
    
    print(f"🎲 Probability:")
    print(f"   Profit: {stats['prob_profit']*100:>5.1f}%")
    print(f"   Loss:   {stats['prob_loss']*100:>5.1f}%")
    print()
    
    print(f"📊 Metrics:")
    print(f"   Mean Sharpe: {stats['mean_sharpe']:>5.2f}")
    print(f"   Mean MaxDD: {stats['mean_max_dd']*100:>5.2f}%")
    print(f"   Worst MaxDD: {stats['worst_max_dd']*100:>5.2f}%")
    print(f"   Mean Trades: {stats['mean_trades']:>5.1f}")
    print(f"   Win Rate: {stats['mean_win_rate']*100:>5.1f}%")
    print()
    
    print("=" * 80)
    print()
    
    # Summary
    print(f"🎯 Summary (20 days):")
    print(f"   Expected P&L: ${stats['mean_pnl']:,.2f}")
    print(f"   Annualized: ${stats['mean_pnl'] * 12.5:,.2f} (~{stats['mean_pnl'] * 12.5 / 100000 * 100:.1f}%)")
    
    if stats['mean_pnl'] > 2000:
        print(f"   ✅ Strong positive edge - Strategy is viable")
    elif stats['mean_pnl'] > 0:
        print(f"   ✅ Positive edge - Strategy works")
    elif stats['mean_pnl'] > -1000:
        print(f"   ⚠️  Small negative edge - Needs tuning")
    else:
        print(f"   ❌ Negative edge - Strategy needs revision")
    
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='NVDA')
    parser.add_argument('--simulations', type=int, default=100000)
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--days', type=int, default=20)
    parser.add_argument('--workers', type=int, default=None)
    
    args = parser.parse_args()
    
    sim = ConservativeSimulator(
        model_path='models/gauge_kernel.pt',
        initial_capital=args.capital,
        trading_days=args.days,
    )
    
    stats = sim.run_monte_carlo(args.symbol, args.simulations, args.workers)
    print_results(stats, args.symbol)


if __name__ == "__main__":
    main()

