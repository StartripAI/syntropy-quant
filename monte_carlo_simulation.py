#!/usr/bin/env python3
"""
Monte Carlo Simulation for 20-Day Intraday Mid-Frequency Trading
Syntropy Quant v5.0

Simulates 100,000 trading sessions to estimate expected returns and risk.
"""
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial
import time

from src.data.fetcher import DataFetcher
from src.data.features import FeatureBuilder
from src.core.kernel import SyntropyQuantKernel, RiskManager
from src.core.filters import RicciCurvatureFilter


class MonteCarloSimulator:
    """
    Monte Carlo simulator for intraday mid-frequency trading.
    
    Simulates:
    - Market volatility
    - Execution delays (150-700ms)
    - Slippage
    - Signal noise
    - Position sizing
    """
    
    def __init__(
        self,
        model_path: str,
        initial_capital: float = 100000,
        trading_days: int = 20,
        trades_per_day: int = 10,
        slippage_bps: float = 5.0,  # 5 basis points
        commission_bps: float = 1.0,  # 1 basis point
    ):
        self.initial_capital = initial_capital
        self.trading_days = trading_days
        self.trades_per_day = trades_per_day
        self.slippage_bps = slippage_bps / 10000
        self.commission_bps = commission_bps / 10000
        
        # Load model (try both kernel types)
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if it's a state_dict or full checkpoint
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                self.model = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✅ SyntropyQuantKernel loaded from {model_path}")
            elif isinstance(checkpoint, dict) and any(k.startswith('engine.') or k.startswith('policy') for k in checkpoint.keys()):
                # GaugeFieldKernel format - use it directly
                from src.core.gauge import GaugeFieldKernel, GaugeConfig
                config = GaugeConfig(input_dim=12)
                self.model = GaugeFieldKernel(input_dim=12, config=config)
                self.model.load_state_dict(checkpoint, strict=False)
                self.use_gauge_kernel = True
                print(f"✅ GaugeFieldKernel loaded from {model_path}")
            else:
                # SyntropyQuantKernel format
                self.model = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
                self.model.load_state_dict(checkpoint, strict=False)
                self.use_gauge_kernel = False
                print(f"✅ SyntropyQuantKernel loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Model loading issue: {e}")
            print("   Creating new model for simulation...")
            self.model = SyntropyQuantKernel(input_dim=4, hidden_dim=64)
            self.use_gauge_kernel = False
        
        if not hasattr(self, 'use_gauge_kernel'):
            self.use_gauge_kernel = isinstance(self.model, type) and 'Gauge' in str(type(self.model))
        
        self.model.eval()
        self.risk_manager = RiskManager(
            max_position=0.1,  # 10% max position
            max_drawdown=0.15,
            vol_target=0.15
        )
        
    def simulate_single_session(
        self,
        symbol: str,
        seed: int = None
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
        positions = {}  # symbol -> shares
        returns = []
        trades = []
        
        # Market parameters (estimated from historical data)
        daily_vol = 0.02  # 2% daily volatility
        drift = 0.0005  # Slight positive drift
        
        # Simulate 20 trading days
        for day in range(self.trading_days):
            # Reset model state each day
            self.model.q = None
            self.model.p = None
            
            # Daily price movement (random walk with drift)
            if day == 0:
                price = 100.0  # Starting price
            else:
                # Random walk with drift
                price_change = np.random.normal(drift, daily_vol)
                price = price * (1 + price_change)
            
            # Generate trading signals (trades_per_day times per day)
            for trade_idx in range(self.trades_per_day):
                # Simulate feature vector (normalized)
                # [log_ret, vol, momentum, position]
                features = np.random.normal(0, 0.5, 4)
                features = np.clip(features, -3, 3)
                
                # Model prediction
                with torch.no_grad():
                    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    try:
                        if self.use_gauge_kernel:
                            # GaugeFieldKernel: needs 12 features, pad if needed
                            if features.shape[0] < 12:
                                x_padded = torch.zeros(1, 12)
                                x_padded[0, :features.shape[0]] = x[0]
                                x = x_padded
                            logits, free_energy, confidence = self.model(x)
                            probs = torch.softmax(logits, dim=1).numpy()[0]
                            gamma = torch.tensor([[free_energy.mean().item() * 0.1]])  # Scale for damping
                        else:
                            # SyntropyQuantKernel interface
                            logits, gamma = self.model(x, dt=1.0)
                            probs = torch.softmax(logits, dim=1).numpy()[0]
                    except Exception as e:
                        # Fallback: use process_step if available
                        if hasattr(self.model, 'process_step'):
                            output = self.model.process_step(features, dt=1.0)
                            probs = np.array([
                                max(0, 1 - output.signal - 0.1),  # Short
                                0.1,  # Neutral
                                max(0, output.signal + 0.1)  # Long
                            ])
                            probs = probs / probs.sum()
                            gamma = torch.tensor([[output.damping]])
                        else:
                            # Default: random probabilities
                            probs = np.array([0.33, 0.34, 0.33])
                            gamma = torch.tensor([[0.1]])
                
                # Signal: Long prob - Short prob
                signal = probs[2] - probs[0]  # Long - Short
                confidence = probs.max()
                
                # Risk management
                realized_vol = daily_vol
                curvature = float(gamma.abs().mean())
                
                position_signal = self.risk_manager.compute_position_size(
                    signal=signal,
                    curvature=curvature,
                    realized_vol=realized_vol
                )
                
                # Execute trade if signal is strong enough
                threshold = 0.1
                if abs(position_signal.risk_adjusted_signal) > threshold:
                    # Calculate position size
                    position_value = capital * abs(position_signal.risk_adjusted_signal)
                    shares = position_value / price
                    
                    # Round to whole shares
                    shares = int(shares)
                    if shares == 0:
                        continue
                    
                    # Execution delay (150-700ms) -> price impact
                    execution_delay_ms = np.random.uniform(150, 700)
                    price_impact = (execution_delay_ms / 1000) * daily_vol * np.random.choice([-1, 1])
                    execution_price = price * (1 + price_impact)
                    
                    # Slippage
                    slippage = execution_price * self.slippage_bps * np.random.choice([-1, 1])
                    execution_price += slippage
                    
                    # Commission
                    commission = position_value * self.commission_bps
                    
                    # Update position
                    side = 'buy' if position_signal.risk_adjusted_signal > 0 else 'sell'
                    
                    if side == 'buy':
                        cost = shares * execution_price + commission
                        if cost <= capital:
                            positions[symbol] = positions.get(symbol, 0) + shares
                            capital -= cost
                            trades.append({
                                'day': day,
                                'side': 'buy',
                                'shares': shares,
                                'price': execution_price,
                                'cost': cost
                            })
                    else:  # sell
                        current_shares = positions.get(symbol, 0)
                        if current_shares >= shares:
                            proceeds = shares * execution_price - commission
                            positions[symbol] = current_shares - shares
                            capital += proceeds
                            trades.append({
                                'day': day,
                                'side': 'sell',
                                'shares': shares,
                                'price': execution_price,
                                'proceeds': proceeds
                            })
            
            # End of day: mark to market
            if symbol in positions and positions[symbol] > 0:
                current_value = positions[symbol] * price
                capital += current_value
                positions[symbol] = 0  # Close position at end of day
            
            # Update capital tracking
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
            # Simple win rate: count profitable trades
            profitable_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = profitable_trades / len(trades) if len(trades) > 0 else 0
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
        """
        Run Monte Carlo simulation with parallel processing.
        
        Args:
            symbol: Trading symbol
            n_simulations: Number of simulations
            n_workers: Number of parallel workers (default: CPU count)
        
        Returns:
            Dict with statistics
        """
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
                partial(self.simulate_single_session, symbol),
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
            
            # PnL statistics
            'mean_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'std_pnl': np.std(pnls),
            'min_pnl': np.min(pnls),
            'max_pnl': np.max(pnls),
            
            # PnL percentage
            'mean_pnl_pct': np.mean(pnl_pcts),
            'median_pnl_pct': np.median(pnl_pcts),
            'std_pnl_pct': np.std(pnl_pcts),
            
            # Risk metrics
            'mean_sharpe': np.mean(sharpes),
            'mean_max_dd': np.mean(max_dds),
            'worst_max_dd': np.min(max_dds),
            
            # Trading statistics
            'mean_trades': np.mean(total_trades),
            'mean_win_rate': np.mean(win_rates),
            
            # Percentiles
            'p5_pnl': np.percentile(pnls, 5),
            'p25_pnl': np.percentile(pnls, 25),
            'p75_pnl': np.percentile(pnls, 75),
            'p95_pnl': np.percentile(pnls, 95),
            
            # Probability of profit
            'prob_profit': np.mean([p > 0 for p in pnls]),
            'prob_loss': np.mean([p < 0 for p in pnls]),
            
            # Expected value
            'expected_value': np.mean(pnls),
            'expected_value_pct': np.mean(pnl_pcts),
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
    print(f"   5th:         ${stats['p5_pnl']:>12,.2f}")
    print(f"   25th:        ${stats['p25_pnl']:>12,.2f}")
    print(f"   75th:        ${stats['p75_pnl']:>12,.2f}")
    print(f"   95th:        ${stats['p95_pnl']:>12,.2f}")
    print()
    
    print(f"🎲 Probability:")
    print(f"   Profit:      {stats['prob_profit']*100:>6.2f}%")
    print(f"   Loss:         {stats['prob_loss']*100:>6.2f}%")
    print()
    
    print(f"📊 Risk Metrics:")
    print(f"   Mean Sharpe: {stats['mean_sharpe']:>6.2f}")
    print(f"   Mean MaxDD:  {stats['mean_max_dd']*100:>6.2f}%")
    print(f"   Worst MaxDD: {stats['worst_max_dd']*100:>6.2f}%")
    print()
    
    print(f"🔄 Trading Statistics:")
    print(f"   Mean Trades: {stats['mean_trades']:>6.1f}")
    print(f"   Win Rate:    {stats['mean_win_rate']*100:>6.2f}%")
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
    else:
        print(f"   ⚠️  Negative expected value - Strategy needs optimization")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation for 20-Day Trading')
    parser.add_argument('--model', type=str, default='models/gauge_kernel.pt',
                       help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='NVDA',
                       help='Trading symbol')
    parser.add_argument('--simulations', type=int, default=100000,
                       help='Number of Monte Carlo simulations')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--days', type=int, default=20,
                       help='Trading days')
    parser.add_argument('--trades-per-day', type=int, default=10,
                       help='Trades per day')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file for detailed results')
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = MonteCarloSimulator(
        model_path=args.model,
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
    if args.output:
        df = pd.DataFrame(detailed_results)
        df.to_csv(args.output, index=False)
        print(f"✅ Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()

