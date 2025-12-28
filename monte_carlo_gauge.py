#!/usr/bin/env python3
"""
Monte Carlo Simulation using GaugeFieldKernel (14-dim input)
Corrected version with proper feature dimensions.
"""
import argparse
import torch
import numpy as np
from typing import Dict, Tuple
from functools import partial
import multiprocessing as mp
import time

from src.core.gauge import GaugeFieldKernel, GaugeKernelOutput, GaugeConfig


class MonteCarloSimulatorGauge:
    """
    Monte Carlo simulator using GaugeFieldKernel.
    Correctly uses 14-dimensional features as expected by the model.
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
        
        # Load GaugeFieldKernel (14-dim input)
        self.config = GaugeConfig(input_dim=14)
        self.model = GaugeFieldKernel(input_dim=14, config=self.config)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✅ GaugeFieldKernel loaded from {model_path}")
            print(f"   Input dimension: 14")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        self.model.eval()
    
    def generate_14_dim_features(
        self,
        price: float,
        day: int,
        daily_vol: float = 0.02
    ) -> np.ndarray:
        """
        Generate 14-dimensional features matching the expected input format.
        
        Features include:
        1-4: Price-based (log returns, momentum, etc.)
        5-8: Volume-based
        9-12: Technical indicators
        13-14: Market state
        """
        # Price-based features
        log_return = np.random.normal(0.001, daily_vol)  # 0.1% drift, 2% vol
        momentum = (price - 100) / 100  # Position relative to 100
        volatility = daily_vol * (1 + np.random.normal(0, 0.2))  # Slight variation
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Trend signal
        
        # Volume-based features
        volume_change = np.random.normal(0, 0.3)  # Volume anomaly
        volume_ma_ratio = np.random.uniform(0.8, 1.2)  # Volume vs MA
        order_flow = np.random.normal(0, 0.5)  # Order flow imbalance
        vwap_deviation = np.random.normal(0, 0.01)  # Price vs VWAP
        
        # Technical indicators
        rsi = np.random.uniform(30, 70) / 100  # Normalized RSI [-0.5, 0.5]
        macd = np.random.normal(0, 0.5)  # MACD signal
        bollinger_position = np.random.uniform(-1, 1)  # BB position
        atr = daily_vol * np.random.uniform(0.8, 1.2)  # ATR
        
        # Market state
        market_regime = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # Normal, Bull, Bear
        time_of_day = np.sin(2 * np.pi * day / 5)  # 5-day cycle
        
        features = np.array([
            log_return,
            momentum,
            volatility,
            trend,
            volume_change,
            volume_ma_ratio,
            order_flow,
            vwap_deviation,
            rsi,
            macd,
            bollinger_position,
            atr,
            market_regime,
            time_of_day,
        ])
        
        return features
    
    def simulate_single_session(
        self,
        symbol: str,
        seed: int = None,
        use_realistic_features: bool = True
    ) -> Dict:
        """
        Simulate a single 20-day trading session using GaugeFieldKernel.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize
        capital = self.initial_capital
        peak_capital = capital
        current_position = 0.0
        trades = []
        daily_returns = []
        
        # Market parameters
        daily_vol = 0.02  # 2% daily volatility
        drift = 0.001  # 0.1% daily drift (slight upward bias)
        
        # Simulate 20 trading days
        for day in range(self.trading_days):
            # Daily price movement
            if day == 0:
                price = 100.0
            else:
                price_change = np.random.normal(drift, daily_vol)
                price = price * (1 + price_change)
            
            # Generate 14-dim features
            if use_realistic_features:
                features = self.generate_14_dim_features(price, day, daily_vol)
            else:
                # Fallback: random features
                features = np.random.normal(0, 0.5, 14)
                features = np.clip(features, -3, 3)
            
            # Model prediction using process_step
            try:
                with torch.no_grad():
                    output = self.model.process_step(features, dt=1.0)
                    
                    signal = output.signal
                    confidence = output.confidence
                    curvature = output.curvature
                    regime = output.regime
                    energy = output.energy
            except Exception as e:
                # Fallback if model fails
                signal = np.random.uniform(-0.1, 0.1)
                confidence = 0.5
                curvature = 0.1
                regime = 'normal'
                energy = 0.0
            
            # Risk-based position sizing
            # Reduce position size in high-risk regimes
            risk_multiplier = 1.0
            if regime == 'crisis':
                risk_multiplier = 0.2
            elif regime == 'high_vol':
                risk_multiplier = 0.5
            
            # Scale signal by confidence and risk
            position_size = abs(signal) * confidence * risk_multiplier
            position_size = min(position_size, 0.15)  # Max 15% position
            
            # Trade threshold (only trade on strong signals)
            trade_threshold = 0.05
            
            # Execute trade if signal is strong enough
            if abs(signal) > trade_threshold:
                side = 'buy' if signal > 0 else 'sell'
                
                # Calculate position
                trade_value = capital * position_size
                
                # Simulate execution delay and slippage
                execution_delay_ms = np.random.uniform(150, 700)
                price_impact = (execution_delay_ms / 1000) * daily_vol * price * np.random.choice([-1, 1])
                execution_price = price * (1 + price_impact)
                
                # Add slippage
                slippage = execution_price * self.slippage_bps * np.random.choice([-1, 1])
                execution_price += slippage
                
                # Commission
                commission = abs(trade_value) * self.commission_bps
                
                # Execute trade
                if side == 'buy':
                    if trade_value + commission <= capital:
                        new_position = trade_value / execution_price
                        if current_position > 0:  # Close existing position
                            pnl = current_position * execution_price
                            capital += pnl
                        current_position = new_position
                        capital -= trade_value + commission
                        trades.append({
                            'day': day,
                            'side': 'buy',
                            'shares': new_position,
                            'price': execution_price,
                            'cost': trade_value + commission,
                            'signal': signal,
                            'confidence': confidence,
                        })
                else:  # sell
                    if current_position > 0:
                        # Close position
                        proceeds = current_position * execution_price - commission
                        capital += proceeds
                        trades.append({
                            'day': day,
                            'side': 'sell',
                            'shares': current_position,
                            'price': execution_price,
                            'proceeds': proceeds,
                            'signal': signal,
                            'confidence': confidence,
                        })
                        current_position = 0.0
            
            # End of day: calculate total value
            total_value = capital
            if current_position > 0:
                total_value += current_position * price
            
            peak_capital = max(peak_capital, total_value)
            daily_return = (total_value - self.initial_capital) / self.initial_capital
            daily_returns.append(daily_return)
        
        # Close any remaining position
        if current_position > 0:
            capital += current_position * price
            current_position = 0.0
        
        total_value = capital
        
        # Calculate metrics
        final_pnl = total_value - self.initial_capital
        final_pnl_pct = (final_pnl / self.initial_capital) * 100
        
        # Sharpe ratio
        if len(daily_returns) > 1:
            returns_array = np.array(daily_returns)
            mean_return = returns_array.mean()
            std_return = returns_array.std()
            sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        equity_curve = np.array([self.initial_capital] + [self.initial_capital * (1 + r) for r in daily_returns])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        if len(trades) > 0:
            win_count = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = win_count / len(trades) if len(trades) > 0 else 0
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
            'returns': daily_returns,
        }
    
    def run_monte_carlo(
        self,
        symbol: str,
        n_simulations: int = 100000,
        n_workers: int = None
    ) -> Tuple[Dict, list]:
        """Run Monte Carlo simulation with parallel processing."""
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)
        
        print(f"🚀 Starting Monte Carlo Simulation (GaugeFieldKernel)")
        print(f"   Symbol: {symbol}")
        print(f"   Simulations: {n_simulations:,}")
        print(f"   Trading Days: {self.trading_days}")
        print(f"   Initial Capital: ${self.initial_capital:,.0f}")
        print(f"   Workers: {n_workers}")
        print()
        
        start_time = time.time()
        
        # Parallel simulation
        with mp.Pool(n_workers) as pool:
            results = pool.map(
                partial(self.simulate_single_session, symbol, use_realistic_features=True),
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
            'initial_capital': self.initial_capital,
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
        }
        
        # Percentiles
        stats['pnl_5th'] = np.percentile(pnls, 5)
        stats['pnl_25th'] = np.percentile(pnls, 25)
        stats['pnl_50th'] = np.percentile(pnls, 50)
        stats['pnl_75th'] = np.percentile(pnls, 75)
        stats['pnl_95th'] = np.percentile(pnls, 95)
        stats['pnl_99th'] = np.percentile(pnls, 99)
        
        return stats, results


def print_results(stats: Dict, symbol: str):
    """Print formatted results"""
    print("=" * 80)
    print(f"  MONTE CARLO SIMULATION RESULTS (GaugeFieldKernel) - {symbol}")
    print("=" * 80)
    print()
    
    print(f"📊 Simulation Parameters:")
    print(f"   Simulations: {stats['n_simulations']:,}")
    print(f"   Elapsed Time: {stats['elapsed_time']:.1f}s")
    print(f"   Initial Capital: ${stats['initial_capital']:,.0f}")
    print()
    
    print(f"💰 Expected P&L:")
    print(f"   Mean:        ${stats['mean_pnl']:>12,.2f} ({stats['mean_pnl_pct']:>7.2f}%)")
    print(f"   Median:      ${stats['median_pnl']:>12,.2f} ({stats['median_pnl_pct']:>7.2f}%)")
    print(f"   Std Dev:     ${stats['std_pnl']:>12,.2f}")
    print(f"   Min:         ${stats['min_pnl']:>12,.2f}")
    print(f"   Max:         ${stats['max_pnl']:>12,.2f}")
    print()
    
    print(f"📈 P&L Distribution:")
    print(f"   5th:         ${stats['pnl_5th']:>12,.2f}")
    print(f"   25th:        ${stats['pnl_25th']:>12,.2f}")
    print(f"   50th:        ${stats['pnl_50th']:>12,.2f}")
    print(f"   75th:        ${stats['pnl_75th']:>12,.2f}")
    print(f"   95th:        ${stats['pnl_95th']:>12,.2f}")
    print(f"   99th:        ${stats['pnl_99th']:>12,.2f}")
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
    
    print("=" * 80)
    print()
    
    # Summary
    ev = stats['expected_value']
    ev_pct = stats['expected_value_pct']
    prob_profit = stats['prob_profit']
    
    print(f"🎯 Summary:")
    print(f"   Expected Value: ${ev:,.2f} ({ev_pct:.2f}%)")
    print(f"   Probability of Profit: {prob_profit*100:.1f}%")
    
    if ev > 5000:  # >5% return
        print(f"   ✅ Strong positive expected value - Strategy is highly profitable")
    elif ev > 0:
        print(f"   ✅ Positive expected value - Strategy is profitable on average")
    elif ev > -5000:  # Tolerable small loss
        print(f"   ⚠️  Small negative expected value - Strategy needs optimization")
    else:
        print(f"   ❌ Large negative expected value - Strategy needs major revision")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation with GaugeFieldKernel')
    parser.add_argument('--model', type=str, default='models/gauge_kernel.pt',
                       help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='NVDA',
                       help='Trading symbol')
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Number of Monte Carlo simulations')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--days', type=int, default=20,
                       help='Trading days')
    parser.add_argument('--trades-per-day', type=int, default=10,
                       help='Trades per day (not directly used, for reference)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = MonteCarloSimulatorGauge(
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
    
    # Save detailed results
    import pandas as pd
    df = pd.DataFrame(detailed_results)
    df.to_csv(f'monte_carlo_gauge_results_{args.symbol}.csv', index=False)
    print(f"✅ Detailed results saved to monte_carlo_gauge_results_{args.symbol}.csv")


if __name__ == "__main__":
    main()

