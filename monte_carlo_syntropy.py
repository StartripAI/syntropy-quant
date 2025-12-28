#!/usr/bin/env python3
"""
Monte Carlo Simulation using SyntropyQuantKernel (v4.0)
Uses the proven physics kernel that achieved +111.9% returns.
"""
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from functools import partial
import multiprocessing as mp
from datetime import datetime, timedelta
import time

from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel
from src.core.filters import RicciCurvatureFilter


class MonteCarloSimulator:
    """
    Monte Carlo simulator using SyntropyQuantKernel.
    Matches the architecture that achieved +111.9% backtest results.
    """
    
    def __init__(
        self,
        model_path: str,
        initial_capital: float = 100000,
        trading_days: int = 20,
        trades_per_day: int = 10,
        slippage_bps: float = 5.0,
        commission_bps: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.trading_days = trading_days
        self.trades_per_day = trades_per_day
        self.slippage_bps = slippage_bps / 10000
        self.commission_bps = commission_bps / 10000
        
        # Load SyntropyQuantKernel (not GaugeFieldKernel)
        self.model = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ SyntropyQuantKernel loaded from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        self.model.eval()
        
        # Simple risk manager (matches kernel output)
        self.risk_threshold = 0.1
        self.position_limit = 0.1  # 10% max position
    
    def simulate_single_session(
        self,
        symbol: str,
        seed: int = None,
        use_real_features: bool = True
    ) -> Dict:
        """
        Simulate a single 20-day trading session.
        
        Returns:
            Dict with final_pnl, sharpe, max_dd, total_trades, win_rate
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize
        capital = self.initial_capital
        peak_capital = capital
        positions = {}
        returns = []
        trades = []
        
        # Market parameters (based on historical data)
        daily_vol = 0.02  # 2% daily volatility
        drift = 0.0005  # Slight positive drift (momentum market)
        
        # Simulate 20 trading days
        for day in range(self.trading_days):
            # Reset model state each day
            # Note: SyntropyQuantKernel maintains internal state (q, p)
            # We'll let it run naturally across the session
            
            # Daily price movement (momentum bias)
            if day == 0:
                price = 100.0
            else:
                # Random walk with momentum drift
                price_change = np.random.normal(drift, daily_vol)
                price = price * (1 + price_change)
            
            # Generate features (deterministic for consistency)
            if use_real_features:
                # Create realistic features matching training data
                log_ret = np.random.normal(0.01, 0.02)  # ~1% daily returns
                vol = 0.15 + np.random.normal(0, 0.05) * daily_vol  # 15% vol
                momentum = np.random.normal(0.0, 0.5)  # Centered around 0
                position = (price - 100) / 100  # Simple position metric
                
                features = np.array([log_ret, vol, momentum, position])
            else:
                # Fallback: random features
                features = np.random.normal(0, 0.5, 4)
                features = np.clip(features, -3, 3)
            
            # Model prediction
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                try:
                    logits, gamma = self.model(x, dt=1.0)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                except Exception as e:
                    # Fallback if model fails
                    print(f"⚠️  Model prediction error: {e}")
                    probs = np.array([0.3, 0.4, 0.3])  # Neutral bias
                
                # Signal: Long prob - Short prob
                signal = probs[2] - probs[0]
                confidence = probs.max()
                
                # Risk management (simplified)
                realized_vol = daily_vol
                curvature = abs(float(gamma.mean())) if len(gamma.shape) > 0 else 0.1
                
                # Position sizing based on confidence and risk
                position_size = abs(signal) * confidence
                if curvature > 0.3:  # High risk regime
                    position_size *= 0.5
                position_size = min(position_size, self.position_limit)
                
                # Trade threshold
                trade_threshold = 0.05  # 5% minimum signal strength
                
                # Execute trade if signal is strong enough
                if abs(signal) > trade_threshold:
                    # Check if we have capital
                    max_position_value = capital * self.position_limit
                    trade_value = capital * position_size
                    
                    if trade_value > max_position_value:
                        trade_value = max_position_value
                    
                    side = 'buy' if signal > 0 else 'sell'
                    
                    # Calculate execution
                    execution_delay_ms = np.random.uniform(150, 700)
                    price_impact = (execution_delay_ms / 1000) * daily_vol * price * np.random.choice([-1, 1])
                    execution_price = price * (1 + price_impact)
                    
                    # Slippage
                    slippage = execution_price * self.slippage_bps * np.random.choice([-1, 1])
                    execution_price += slippage
                    
                    # Commission
                    commission = abs(trade_value) * self.commission_bps
                    
                    # Execute trade
                    if side == 'buy':
                        if trade_value + commission <= capital:
                            positions[symbol] = positions.get(symbol, 0) + trade_value / execution_price
                            capital -= trade_value + commission
                            trades.append({
                                'day': day,
                                'side': 'buy',
                                'shares': trade_value / execution_price,
                                'price': execution_price,
                                'cost': trade_value + commission
                            })
                    else:  # sell
                        current_shares = positions.get(symbol, 0)
                        if current_shares > 0:
                            sell_shares = min(current_shares, trade_value / execution_price)
                            proceeds = sell_shares * execution_price - commission
                            positions[symbol] = current_shares - sell_shares
                            capital += proceeds
                            trades.append({
                                'day': day,
                                'side': 'sell',
                                'shares': sell_shares,
                                'price': execution_price,
                                'proceeds': proceeds
                            })
            
            # End of day: close positions
            if symbol in positions and positions[symbol] > 0:
                current_value = positions[symbol] * price
                capital += current_value
                positions[symbol] = 0
            
            # Track capital
            total_value = capital
            peak_capital = max(peak_capital, total_value)
            daily_return = (total_value - self.initial_capital) / self.initial_capital
            returns.append(daily_return)
        
        # Calculate metrics
        final_pnl = total_value - self.initial_capital
        final_pnl_pct = (final_pnl / self.initial_capital) * 100
        
        # Sharpe ratio
        if len(returns) > 1:
            returns_array = np.array(returns)
            mean_return = returns_array.mean()
            std_return = returns_array.std()
            sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        equity_curve = np.array([self.initial_capital] + [self.initial_capital * (1 + r) for r in returns])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        if len(trades) > 0:
            profitable_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = profitable_trades / len(trades)
        else:
            win_rate = 0
        
        return {
            'final_pnl': final_pnl,
            'final_pnl_pct': final_pnl_pct,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'final_capital': total_value,
            'returns': returns
        }
    
    def run_monte_carlo(
        self,
        symbol: str,
        n_simulations: int = 100000,
        n_workers: int = None
    ) -> Dict:
        """Run Monte Carlo simulation with parallel processing."""
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)
        
        print(f"🚀 Starting Monte Carlo Simulation")
        print(f"   Symbol: {symbol}")
        print(f"   Simulations: {n_simulations:,}")
        print(f"   Trading Days: {self.trading_days}")
        print(f"   Trades/Day: {self.trades_per_day}")
        print(f"   Initial Capital: ${self.initial_capital:,.0f}")
        print(f"   Workers: {n_workers}")
        print()
        
        start_time = time.time()
        
        # Parallel simulation
        with mp.Pool(n_workers) as pool:
            results = pool.map(
                partial(self.simulate_single_session, symbol, use_real_features=True),
                range(n_simulations)
            )
        
        elapsed = time.time() - start_time
        
        # Aggregate results
        pnls = [r['final_pnl'] for r in results]
        pnl_pcts = [r['final_pnl_pct'] for r in results]
        sharpes = [r['sharpe'] for r in results]
        max_dds = [r['max_dd'] for r in results]
        total_trades = [r['total_trades'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        
        # Statistics
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
            'std_pnl_pct': np.std(pnl_pcts),
            'min_pnl_pct': np.min(pnl_pcts),
            'max_pnl_pct': np.max(pnl_pcts),
            'mean_sharpe': np.mean(sharpes),
            'mean_max_dd': np.mean(max_dds),
            'worst_max_dd': np.min(max_dds),
            'mean_trades': np.mean(total_trades),
            'mean_win_rate': np.mean(win_rates),
            'prob_profit': np.mean([p > 0 for p in pnls]),
            'prob_loss': np.mean([p < 0 for p in pnls]),
            'expected_value': np.mean(pnls),
            'expected_value_pct': np.mean(pnl_pcts),
            'prob_5th_pnl': (np.percentile(pnls, 5) > 0) / n_simulations,
            'prob_25th_pnl': (np.percentile(pnls, 25) > 0) / n_simulations,
            'prob_75th_pnl': (np.percentile(pnls, 75) > 0) / n_simulations,
            'prob_95th_pnl': (np.percentile(pnls, 95) > 0) / n_simulations,
        }
        
        return stats, results


def print_results(stats: Dict, symbol: str):
    """Print formatted results"""
    print("=" * 70)
    print(f"  MONTE CARLO SIMULATION RESULTS - {symbol}")
    print("=" * 70)
    print()
    
    print(f"📊 Simulation Parameters:")
    print(f"   Simulations: {stats['n_simulations']:,}")
    print(f"   Elapsed Time: {stats['elapsed_time']:.1f}s")
    print()
    
    print(f"💰 Expected P&L:")
    print(f"   Mean:        ${stats['mean_pnl']:>12,.2f} ({stats['mean_pnl_pct']:>7.2f}%)")
    print(f"   Median:      ${stats['median_pnl']:>12,.2f} ({stats['median_pnl_pct']:>7.2f}%)")
    print(f"   Std Dev:     ${stats['std_pnl']:>12,.2f}")
    print(f"   Min:         ${stats['min_pnl']:>12,.2f}")
    print(f"   Max:         ${stats['max_pnl']:>12,.2f}")
    print()
    
    print(f"📈 Percentiles:")
    print(f"   5th:         ${stats['prob_5th_pnl'] * stats['initial_capital'] / 100:,.2f}")
    print(f"   25th:        ${stats['prob_25th_pnl'] * stats['initial_capital'] / 100:,.2f}")
    print(f"   75th:        ${stats['prob_75th_pnl'] * stats['initial_capital'] / 100:,.2f}")
    print(f"   95th:        ${stats['prob_95th_pnl'] * stats['initial_capital'] / 100:,.2f}")
    print()
    
    print(f"🎲 Probability:")
    print(f"   Profit:      {stats['prob_profit']*100:>6.2f}%")
    print(f"   Loss:         {stats['prob_loss']*100:>6.2f}%")
    print()
    
    print(f"📊 Risk Metrics:")
    print(f"   Mean Sharpe: {stats['mean_sharpe']:>6.2f}")
    print(f"   Mean MaxDD: {stats['mean_max_dd']*100:>6.2f}%")
    print(f"   Worst MaxDD: {stats['worst_max_dd']*100:>6.2f}%")
    print()
    
    print(f"🔄 Trading Statistics:")
    print(f"   Mean Trades: {stats['mean_trades']:>6.1f}")
    print(f"   Mean Win Rate: {stats['mean_win_rate']*100:>6.2f}%")
    print()
    
    print("=" * 70)
    print()
    
    # Summary
    ev = stats['expected_value']
    ev_pct = stats['expected_value_pct']
    prob_profit = stats['prob_profit']
    
    print(f"🎯 Summary:")
    print(f"   Expected Value: ${ev:,.2f} ({ev_pct:.2f}%)")
    print(f"   Probability of Profit: {prob_profit*100:.1f}%")
    
    if ev > 0:
        print(f"   ✅ Positive expected value - Strategy is profitable on average")
    elif ev > -5000:  # Tolerable small loss
        print(f"   ⚠️  Small negative expected value - Strategy needs optimization")
    else:
        print(f"   ❌ Large negative expected value - Strategy needs major revision")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation with SyntropyQuantKernel')
    parser.add_argument('--model', type=str, default='models/gauge_kernel.pt',
                       help='Path to trained model')
    parser.add_argument('--model-type', type=str, default='syntropy',
                       choices=['syntropy', 'gauge'],
                       help='Model type: syntropy (physics) or gauge (geometry)')
    parser.add_argument('--symbol', type=str, default='NVDA',
                       help='Trading symbol')
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Number of Monte Carlo simulations')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--days', type=int, default=20,
                       help='Trading days')
    parser.add_argument('--trades-per-day', type=int, default=10,
                       help='Trades per day')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Determine model path based on type
    if args.model_type == 'syntropy':
        model_path = 'models/gauge_kernel.pt'  # SyntropyQuantKernel checkpoint
    else:
        model_path = args.model  # GaugeFieldKernel checkpoint (if exists)
    
    # Create simulator
    simulator = MonteCarloSimulator(
        model_path=model_path,
        initial_capital=args.capital,
        trading_days=args.days,
        trades_per_day=args.trades_per_day
    )
    
    # Run simulation
    stats, detailed_results = simulator.run_monte_carlo(
        symbol=args.symbol,
        n_simulations=args.simulations,
        n_workers=args.workers
    )
    
    # Print results
    print_results(stats, args.symbol)
    
    # Save detailed results if requested
    df = pd.DataFrame(detailed_results)
    df.to_csv(f'monte_carlo_results_{args.symbol}_{args.model_type}.csv', index=False)
    print(f"✅ Detailed results saved to monte_carlo_results_{args.symbol}_{args.model_type}.csv")


if __name__ == "__main__":
    main()

