"""
Syntropy Quant - Institutional Risk Management System
Inspired by Two Sigma / Citadel risk frameworks

Multi-layer risk controls:
1. Position-level limits
2. Portfolio-level limits
3. Drawdown controls
4. Volatility-based sizing
5. Correlation monitoring
6. Sector/factor exposure limits
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging


class RiskLevel(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Position limits
    max_position_pct: float = 0.10          # Max 10% in single position
    max_position_value: float = 50000       # Max $50k per position

    # Portfolio limits
    max_sector_pct: float = 0.30            # Max 30% in single sector
    max_beta: float = 1.5                   # Max portfolio beta
    max_leverage: float = 2.0               # Max leverage ratio

    # Drawdown limits
    max_daily_loss_pct: float = 0.02        # -2% daily stop
    max_weekly_loss_pct: float = 0.05       # -5% weekly stop
    max_drawdown_pct: float = 0.15          # -15% max drawdown

    # Volatility limits
    max_position_vol: float = 0.50          # Max 50% annualized vol per position
    target_portfolio_vol: float = 0.15      # Target 15% portfolio vol

    # Correlation limits
    max_correlation: float = 0.80           # Max correlation between positions


@dataclass
class RiskMetrics:
    """Current risk state"""
    timestamp: datetime = field(default_factory=datetime.now)
    risk_level: RiskLevel = RiskLevel.NORMAL

    # Portfolio metrics
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    current_drawdown: float = 0.0
    peak_value: float = 0.0

    # Risk metrics
    portfolio_vol: float = 0.0
    portfolio_beta: float = 0.0
    leverage: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0

    # Position concentration
    max_position_weight: float = 0.0
    max_sector_weight: float = 0.0
    num_positions: int = 0

    # Breaches
    breaches: List[str] = field(default_factory=list)


class InstitutionalRiskManager:
    """
    Two Sigma / Citadel-style risk management system

    Features:
    - Real-time risk monitoring
    - Multi-layer limit checks
    - Automatic position scaling
    - Drawdown-based deleveraging
    - VaR/CVaR calculations
    - Correlation matrix monitoring
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

        # State tracking
        self.peak_value = 0.0
        self.daily_start_value = 0.0
        self.weekly_start_value = 0.0
        self.last_reset_date = None
        self.last_week_reset = None

        # Historical data
        self.value_history: List[Tuple[datetime, float]] = []
        self.pnl_history: List[float] = []

        # Position data
        self.positions: Dict[str, Dict] = {}
        self.sector_map: Dict[str, str] = {}

        self._setup_logging()
        self._init_sector_map()

    def _setup_logging(self):
        self.logger = logging.getLogger('RiskManager')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [RISK] %(levelname)s: %(message)s'
            ))
            self.logger.addHandler(handler)

    def _init_sector_map(self):
        """Initialize sector mappings"""
        self.sector_map = {
            # Tech
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech',
            'META': 'tech', 'NVDA': 'tech', 'TSLA': 'tech', 'AMD': 'tech',

            # Semis
            'AVGO': 'semis', 'QCOM': 'semis', 'AMAT': 'semis', 'LRCX': 'semis',
            'MU': 'semis', 'INTC': 'semis',

            # Finance
            'JPM': 'finance', 'BAC': 'finance', 'GS': 'finance', 'MS': 'finance',
            'V': 'finance', 'MA': 'finance', 'BRK-B': 'finance',

            # Consumer
            'WMT': 'consumer', 'COST': 'consumer', 'PG': 'consumer', 'KO': 'consumer',
            'MCD': 'consumer', 'HD': 'consumer',

            # Healthcare
            'UNH': 'healthcare', 'LLY': 'healthcare', 'JNJ': 'healthcare',
            'PFE': 'healthcare', 'ABBV': 'healthcare',

            # ETFs
            'SPY': 'index', 'QQQ': 'index', 'IWM': 'index', 'DIA': 'index',
            'VTI': 'index', 'VOO': 'index', 'VT': 'index',
        }

    def update_positions(self, positions: Dict[str, Dict]):
        """Update current positions"""
        self.positions = positions

    def _reset_daily_if_needed(self, current_value: float):
        """Reset daily tracking at market open"""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_start_value = current_value
            self.last_reset_date = today

        # Weekly reset on Monday
        weekday = datetime.now().weekday()
        week_num = datetime.now().isocalendar()[1]
        if self.last_week_reset != week_num and weekday == 0:
            self.weekly_start_value = current_value
            self.last_week_reset = week_num

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 20:
            return 0.0
        return -np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) < 20:
            return 0.0
        var = self.calculate_var(returns, confidence)
        return -np.mean(returns[returns <= -var])

    def calculate_portfolio_vol(self, returns: np.ndarray) -> float:
        """Calculate annualized portfolio volatility"""
        if len(returns) < 20:
            return 0.15  # Default
        return np.std(returns) * np.sqrt(252)

    def compute_risk_metrics(self, portfolio_value: float, returns: Optional[np.ndarray] = None) -> RiskMetrics:
        """
        Compute comprehensive risk metrics
        """
        self._reset_daily_if_needed(portfolio_value)

        # Update peak
        self.peak_value = max(self.peak_value, portfolio_value)

        # PnL calculations
        daily_pnl = portfolio_value - self.daily_start_value
        daily_pnl_pct = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0

        weekly_pnl_pct = 0.0
        if self.weekly_start_value > 0:
            weekly_pnl_pct = (portfolio_value - self.weekly_start_value) / self.weekly_start_value

        # Drawdown
        current_drawdown = (portfolio_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0

        # Store history
        self.value_history.append((datetime.now(), portfolio_value))
        if len(self.value_history) > 252:  # Keep 1 year
            self.value_history.pop(0)

        # Portfolio metrics
        portfolio_vol = self.calculate_portfolio_vol(returns) if returns is not None else 0.15
        var_95 = self.calculate_var(returns) if returns is not None else 0.0
        var_99 = self.calculate_var(returns, 0.99) if returns is not None else 0.0

        # Position concentration
        total_value = sum(p.get('market_value', 0) for p in self.positions.values())
        max_position_weight = 0.0
        if total_value > 0:
            max_position_weight = max(
                abs(p.get('market_value', 0)) / total_value
                for p in self.positions.values()
            ) if self.positions else 0.0

        # Sector concentration
        sector_values = {}
        for symbol, pos in self.positions.items():
            sector = self.sector_map.get(symbol, 'other')
            sector_values[sector] = sector_values.get(sector, 0) + abs(pos.get('market_value', 0))

        max_sector_weight = max(sector_values.values()) / total_value if total_value > 0 and sector_values else 0.0

        # Check breaches
        breaches = []

        if daily_pnl_pct < -self.limits.max_daily_loss_pct:
            breaches.append(f"DAILY_LOSS: {daily_pnl_pct*100:.1f}%")

        if weekly_pnl_pct < -self.limits.max_weekly_loss_pct:
            breaches.append(f"WEEKLY_LOSS: {weekly_pnl_pct*100:.1f}%")

        if current_drawdown < -self.limits.max_drawdown_pct:
            breaches.append(f"MAX_DRAWDOWN: {current_drawdown*100:.1f}%")

        if max_position_weight > self.limits.max_position_pct:
            breaches.append(f"POSITION_CONCENTRATION: {max_position_weight*100:.1f}%")

        if max_sector_weight > self.limits.max_sector_pct:
            breaches.append(f"SECTOR_CONCENTRATION: {max_sector_weight*100:.1f}%")

        # Determine risk level
        if len(breaches) >= 3 or current_drawdown < -self.limits.max_drawdown_pct:
            risk_level = RiskLevel.CRITICAL
        elif len(breaches) >= 2 or current_drawdown < -self.limits.max_drawdown_pct * 0.7:
            risk_level = RiskLevel.HIGH
        elif len(breaches) >= 1 or current_drawdown < -self.limits.max_drawdown_pct * 0.5:
            risk_level = RiskLevel.ELEVATED
        else:
            risk_level = RiskLevel.NORMAL

        return RiskMetrics(
            timestamp=datetime.now(),
            risk_level=risk_level,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            current_drawdown=current_drawdown,
            peak_value=self.peak_value,
            portfolio_vol=portfolio_vol,
            var_95=var_95,
            var_99=var_99,
            max_position_weight=max_position_weight,
            max_sector_weight=max_sector_weight,
            num_positions=len(self.positions),
            breaches=breaches,
        )

    def get_position_scalar(self, metrics: RiskMetrics) -> float:
        """
        Calculate position scaling factor based on risk level

        Returns multiplier between 0 and 1 to scale positions down
        """
        if metrics.risk_level == RiskLevel.CRITICAL:
            self.logger.warning("CRITICAL RISK - Scaling to 25%")
            return 0.25

        if metrics.risk_level == RiskLevel.HIGH:
            self.logger.warning("HIGH RISK - Scaling to 50%")
            return 0.50

        if metrics.risk_level == RiskLevel.ELEVATED:
            self.logger.info("ELEVATED RISK - Scaling to 75%")
            return 0.75

        return 1.0

    def get_volatility_adjusted_size(
        self,
        base_size: float,
        asset_vol: float,
        target_vol: Optional[float] = None
    ) -> float:
        """
        Vol-adjusted position sizing (Two Sigma style)

        Higher vol = smaller position
        """
        target_vol = target_vol or self.limits.target_portfolio_vol

        if asset_vol <= 0:
            return base_size

        vol_scalar = target_vol / asset_vol
        vol_scalar = np.clip(vol_scalar, 0.2, 2.0)  # Cap scaling

        return base_size * vol_scalar

    def validate_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        portfolio_value: float
    ) -> Tuple[bool, str, float]:
        """
        Pre-trade risk validation

        Returns: (approved, reason, adjusted_qty)
        """
        trade_value = qty * price

        # Check 1: Max order value
        if trade_value > self.limits.max_position_value:
            adjusted_qty = self.limits.max_position_value / price
            return True, f"Order capped to ${self.limits.max_position_value:,.0f}", adjusted_qty

        # Check 2: Position concentration
        current_position = self.positions.get(symbol, {}).get('market_value', 0)
        new_position = current_position + (trade_value if side == 'buy' else -trade_value)

        if abs(new_position) / portfolio_value > self.limits.max_position_pct:
            max_allowed = portfolio_value * self.limits.max_position_pct
            adjusted_value = max_allowed - abs(current_position)
            if adjusted_value <= 0:
                return False, "Position limit reached", 0

            adjusted_qty = adjusted_value / price
            return True, f"Qty reduced for position limit", adjusted_qty

        # Check 3: Sector concentration
        sector = self.sector_map.get(symbol, 'other')
        sector_value = sum(
            p.get('market_value', 0)
            for s, p in self.positions.items()
            if self.sector_map.get(s, 'other') == sector
        )
        new_sector_value = sector_value + (trade_value if side == 'buy' else 0)

        if new_sector_value / portfolio_value > self.limits.max_sector_pct:
            return False, f"Sector {sector} limit reached", 0

        return True, "OK", qty

    def should_deleverage(self, metrics: RiskMetrics) -> Tuple[bool, float]:
        """
        Check if portfolio should be deleveraged

        Returns: (should_deleverage, target_exposure_pct)
        """
        if metrics.risk_level == RiskLevel.CRITICAL:
            return True, 0.25

        if metrics.risk_level == RiskLevel.HIGH:
            return True, 0.50

        if metrics.current_drawdown < -self.limits.max_drawdown_pct * 0.8:
            return True, 0.50

        return False, 1.0

    def print_risk_report(self, metrics: RiskMetrics):
        """Print formatted risk report"""
        print("\n" + "=" * 60)
        print("RISK REPORT")
        print("=" * 60)
        print(f"Timestamp:        {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Risk Level:       {metrics.risk_level.value.upper()}")
        print("-" * 60)
        print(f"Portfolio Value:  ${metrics.portfolio_value:,.2f}")
        print(f"Peak Value:       ${metrics.peak_value:,.2f}")
        print(f"Daily PnL:        ${metrics.daily_pnl:,.2f} ({metrics.daily_pnl_pct*100:+.2f}%)")
        print(f"Weekly PnL:       {metrics.weekly_pnl_pct*100:+.2f}%")
        print(f"Drawdown:         {metrics.current_drawdown*100:.2f}%")
        print("-" * 60)
        print(f"Portfolio Vol:    {metrics.portfolio_vol*100:.1f}%")
        print(f"VaR (95%):        {metrics.var_95*100:.2f}%")
        print(f"VaR (99%):        {metrics.var_99*100:.2f}%")
        print("-" * 60)
        print(f"Max Position:     {metrics.max_position_weight*100:.1f}%")
        print(f"Max Sector:       {metrics.max_sector_weight*100:.1f}%")
        print(f"Num Positions:    {metrics.num_positions}")
        print("-" * 60)

        if metrics.breaches:
            print("BREACHES:")
            for breach in metrics.breaches:
                print(f"  ⚠️  {breach}")
        else:
            print("No limit breaches")

        print("=" * 60)


if __name__ == '__main__':
    # Test risk manager
    rm = InstitutionalRiskManager()

    # Simulate positions
    rm.update_positions({
        'NVDA': {'market_value': 15000},
        'GOOGL': {'market_value': 12000},
        'SPY': {'market_value': 10000},
        'QQQ': {'market_value': 8000},
    })

    # Compute metrics
    returns = np.random.randn(100) * 0.02
    metrics = rm.compute_risk_metrics(100000, returns)

    rm.print_risk_report(metrics)

    # Test trade validation
    approved, reason, adj_qty = rm.validate_trade('AAPL', 'buy', 500, 180, 100000)
    print(f"\nTrade validation: {approved} - {reason} (qty: {adj_qty})")
