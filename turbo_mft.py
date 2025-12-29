import asyncio
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.core.gauge import GaugeFieldKernel
from src.data.realtime_buffer import RealTimeFeatureBuffer
from config.optimized_portfolio import get_all_symbols

class SyntropyTurboMFT:
    """
    Syntropy Turbo - Medium Frequency Trading System (Minute-level)
    Inspired by Citadel/Two Sigma low-latency execution pipelines.
    """
    def __init__(self, mode="paper"):
        load_dotenv()
        self.api_key = os.environ.get('ALPACA_API_KEY')
        self.secret_key = os.environ.get('ALPACA_SECRET_KEY')
        
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=(mode == "paper"))
        self.stream = StockDataStream(self.api_key, self.secret_key)
        
        # Physics Engine
        self.kernel = GaugeFieldKernel(input_dim=14)
        # Load weights
        import torch
        self.kernel.load_state_dict(torch.load("models/gauge_kernel_v5.pt", map_location='cpu'))
        self.kernel.eval()
        
        # Memory Buffers for each symbol
        self.symbols = get_all_symbols()[:100] # Track up to 100 for HFT
        self.buffers = {s: RealTimeFeatureBuffer(window_size=60) for s in self.symbols}
        
        self.logger = logging.getLogger('TurboMFT')
        logging.basicConfig(level=logging.INFO)

    async def handle_trade(self, data):
        symbol = data.symbol
        # Trade data: price, size
        # We synthesize a fast bar from trade stream
        bar = [data.price, data.price, data.price, data.price, data.size]
        
        buffer = self.buffers[symbol]
        buffer.push(bar)
        
        feat = buffer.get_features()
        if feat is not None:
            res = self.kernel.process_step(feat)
            action = abs(res.signal)
            
            # Real-time dashboard output
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol:<5} | Price: {data.price:>7.2f} | Action: {action:.4f} | Curve: {res.curvature:.4f}", flush=True)

            if action > 0.02: 
                await self.execute_trade(symbol, "buy" if res.signal > 0 else "sell", res.confidence)

    async def execute_trade(self, symbol, side, confidence):
        try:
            # Max $5000 per trade for high frequency
            base_qty = 10 
            qty = max(1, int(base_qty * (confidence**2))) 
            
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            self.trading_client.submit_order(request)
            self.logger.info(f"🚀 TURBO EXEC: {side.upper()} {qty} {symbol} (Conf: {confidence:.2f})")
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")

    def run(self):
        self.logger.info(f">>> SYNTROPY TURBO TICK-STREAM STARTING: Monitoring {len(self.symbols)} assets <<<")
        # Subscribe to Real-time Trades (Much faster than Bars)
        self.stream.subscribe_trades(self.handle_trade, *self.symbols)
        self.stream.run()

if __name__ == "__main__":
    turbo = SyntropyTurboMFT(mode="paper")
    turbo.run()
