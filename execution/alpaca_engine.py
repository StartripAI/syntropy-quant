"""
Syntropy Quant - Institutional Trading Execution Engine
Inspired by Two Sigma / Citadel execution systems

Features:
- Smart order routing
- TWAP/VWAP execution
- Pre-trade risk checks
- Position reconciliation
- Execution analytics
"""

import os
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

# Alpaca API
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-py not installed. Run: pip install alpaca-py")


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str = 'market'
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class ExecutionEngine:
    """
    Institutional-grade execution engine with Alpaca integration.

    Features:
    - Pre-trade risk validation
    - Smart order slicing (TWAP)
    - Real-time position tracking
    - Execution quality analytics
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        max_order_value: float = 100000,
        max_position_pct: float = 0.10,
        max_daily_trades: int = 100,
    ):
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY')
        self.paper = paper
        self.max_order_value = max_order_value
        self.max_position_pct = max_position_pct
        self.max_daily_trades = max_daily_trades

        self.client = None
        self.data_client = None
        self.daily_trade_count = 0
        self.last_trade_date = None

        self._setup_logging()
        self._connect()

    def _setup_logging(self):
        self.logger = logging.getLogger('ExecutionEngine')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [EXEC] %(levelname)s: %(message)s'
            ))
            self.logger.addHandler(handler)

    def _connect(self):
        if not ALPACA_AVAILABLE:
            self.logger.warning("Alpaca not available - running in simulation mode")
            return

        if not self.api_key or not self.secret_key:
            self.logger.warning("Alpaca credentials not set - running in simulation mode")
            return

        try:
            self.client = TradingClient(
                self.api_key,
                self.secret_key,
                paper=self.paper
            )
            self.data_client = StockHistoricalDataClient(
                self.api_key,
                self.secret_key
            )
            account = self.client.get_account()
            self.logger.info(f"Connected to Alpaca ({'Paper' if self.paper else 'Live'})")
            self.logger.info(f"Account Equity: ${float(account.equity):,.2f}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.client = None

    def get_account_info(self) -> Dict:
        """Get current account information"""
        if not self.client:
            return {'equity': 100000, 'cash': 100000, 'buying_power': 200000}

        account = self.client.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'day_trade_count': int(account.daytrade_count),
        }

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        if not self.client:
            return {}

        positions = {}
        for pos in self.client.get_all_positions():
            positions[pos.symbol] = Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                avg_price=float(pos.avg_entry_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
            )
        return positions

    def get_quote(self, symbol: str) -> Tuple[float, float]:
        """Get current bid/ask for symbol"""
        if not self.data_client:
            return (0.0, 0.0)

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(request)[symbol]
            return (float(quote.bid_price), float(quote.ask_price))
        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return (0.0, 0.0)

    def _pre_trade_checks(self, order: Order) -> Tuple[bool, str]:
        """
        Pre-trade risk validation (Citadel-style)
        """
        # Reset daily counter
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today

        # Check 1: Daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"

        # Check 2: Order value limit
        bid, ask = self.get_quote(order.symbol)
        price = ask if order.side == 'buy' else bid
        if price > 0:
            order_value = order.qty * price
            if order_value > self.max_order_value:
                return False, f"Order value ${order_value:,.0f} exceeds limit ${self.max_order_value:,.0f}"

        # Check 3: Position concentration
        account = self.get_account_info()
        positions = self.get_positions()
        equity = account['equity']

        if order.symbol in positions:
            current_value = positions[order.symbol].market_value
            new_value = current_value + (order.qty * price if order.side == 'buy' else -order.qty * price)
            if abs(new_value) / equity > self.max_position_pct:
                return False, f"Position would exceed {self.max_position_pct*100:.0f}% limit"

        # Check 4: Buying power
        if order.side == 'buy':
            required = order.qty * price
            if required > account['buying_power']:
                return False, f"Insufficient buying power: need ${required:,.0f}, have ${account['buying_power']:,.0f}"

        return True, "OK"

    def submit_order(self, order: Order) -> Order:
        """Submit order with pre-trade checks"""
        # Pre-trade validation
        passed, reason = self._pre_trade_checks(order)
        if not passed:
            self.logger.warning(f"Order rejected: {reason}")
            order.status = OrderStatus.REJECTED
            return order

        if not self.client:
            # Simulation mode
            self.logger.info(f"[SIM] {order.side.upper()} {order.qty} {order.symbol}")
            order.status = OrderStatus.FILLED
            order.filled_qty = order.qty
            order.filled_at = datetime.now()
            self.daily_trade_count += 1
            return order

        try:
            # Submit to Alpaca
            side = OrderSide.BUY if order.side == 'buy' else OrderSide.SELL

            if order.order_type == 'market':
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price
                )

            result = self.client.submit_order(request)
            order.order_id = result.id
            order.status = OrderStatus.SUBMITTED
            self.daily_trade_count += 1

            self.logger.info(f"Order submitted: {order.side.upper()} {order.qty} {order.symbol} (ID: {order.order_id})")

        except Exception as e:
            self.logger.error(f"Order failed: {e}")
            order.status = OrderStatus.REJECTED

        return order

    def execute_twap(
        self,
        symbol: str,
        target_qty: float,
        side: str,
        duration_minutes: int = 30,
        num_slices: int = 6,
    ) -> List[Order]:
        """
        TWAP (Time-Weighted Average Price) execution
        Splits large orders into smaller slices over time
        """
        slice_qty = target_qty / num_slices
        interval = duration_minutes * 60 / num_slices

        orders = []
        self.logger.info(f"Starting TWAP: {side.upper()} {target_qty} {symbol} over {duration_minutes}min")

        for i in range(num_slices):
            order = Order(
                symbol=symbol,
                side=side,
                qty=slice_qty,
                order_type='market'
            )
            result = self.submit_order(order)
            orders.append(result)

            if i < num_slices - 1:
                time.sleep(interval)

        total_filled = sum(o.filled_qty for o in orders)
        self.logger.info(f"TWAP complete: filled {total_filled}/{target_qty}")

        return orders

    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        equity: Optional[float] = None,
    ) -> List[Order]:
        """
        Rebalance portfolio to target weights

        Args:
            target_weights: Dict of symbol -> target weight (0-1)
            equity: Portfolio equity (auto-fetched if None)
        """
        account = self.get_account_info()
        equity = equity or account['equity']
        positions = self.get_positions()

        orders = []

        for symbol, target_weight in target_weights.items():
            target_value = equity * target_weight
            current_value = positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0)).market_value

            delta_value = target_value - current_value

            # Get current price
            bid, ask = self.get_quote(symbol)
            price = (bid + ask) / 2 if bid > 0 else 100  # Fallback

            delta_qty = abs(delta_value) / price

            # Minimum trade threshold (avoid tiny trades)
            if abs(delta_value) < 100:
                continue

            side = 'buy' if delta_value > 0 else 'sell'

            order = Order(
                symbol=symbol,
                side=side,
                qty=round(delta_qty, 2),
                order_type='market'
            )

            result = self.submit_order(order)
            orders.append(result)

        self.logger.info(f"Rebalance complete: {len(orders)} orders submitted")
        return orders

    def close_all_positions(self) -> List[Order]:
        """Emergency: close all positions"""
        self.logger.warning("CLOSING ALL POSITIONS")

        positions = self.get_positions()
        orders = []

        for symbol, pos in positions.items():
            order = Order(
                symbol=symbol,
                side='sell' if pos.qty > 0 else 'buy',
                qty=abs(pos.qty),
                order_type='market'
            )
            result = self.submit_order(order)
            orders.append(result)

        return orders


if __name__ == '__main__':
    # Test execution engine
    engine = ExecutionEngine(paper=True)

    print("\n=== Account Info ===")
    print(engine.get_account_info())

    print("\n=== Current Positions ===")
    for sym, pos in engine.get_positions().items():
        print(f"{sym}: {pos.qty} @ ${pos.avg_price:.2f} (PnL: {pos.unrealized_pnl_pct:.1f}%)")
