#!/usr/bin/env python3
"""
Syntropy Quant - Alpaca API Setup and Testing

Usage:
    python config/setup_alpaca.py --setup      # Configure API keys
    python config/setup_alpaca.py --test       # Test paper trading
    python config/setup_alpaca.py --status     # Check account status
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_credentials():
    """Interactive setup for Alpaca API credentials"""
    print("=" * 60)
    print("Alpaca API Setup")
    print("=" * 60)
    print()
    print("Get your API keys from: https://app.alpaca.markets/")
    print("For paper trading, use the Paper Trading keys.")
    print()

    # Get credentials
    api_key = input("Enter ALPACA_API_KEY: ").strip()
    secret_key = input("Enter ALPACA_SECRET_KEY: ").strip()

    if not api_key or not secret_key:
        print("Error: Both keys are required")
        return False

    # Create .env file
    env_path = Path(__file__).parent.parent / '.env'

    with open(env_path, 'w') as f:
        f.write(f"# Alpaca API Credentials\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"ALPACA_API_KEY={api_key}\n")
        f.write(f"ALPACA_SECRET_KEY={secret_key}\n")
        f.write(f"\n# Alerting (optional)\n")
        f.write(f"TELEGRAM_BOT_TOKEN=\n")
        f.write(f"TELEGRAM_CHAT_ID=\n")
        f.write(f"DISCORD_WEBHOOK_URL=\n")

    print()
    print(f"Credentials saved to: {env_path}")
    print()

    # Also export to current shell
    os.environ['ALPACA_API_KEY'] = api_key
    os.environ['ALPACA_SECRET_KEY'] = secret_key

    return True


def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'

    if not env_path.exists():
        print("No .env file found. Run with --setup first.")
        return False

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

    return True


def test_connection():
    """Test Alpaca API connection"""
    print("=" * 60)
    print("Testing Alpaca API Connection")
    print("=" * 60)
    print()

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
    except ImportError:
        print("Error: alpaca-py not installed")
        print("Run: pip install alpaca-py")
        return False

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("Error: API credentials not found")
        print("Run with --setup first")
        return False

    # Test trading client
    print("1. Testing Trading Client...")
    try:
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()

        print(f"   Account ID: {account.id}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Day Trades: {account.daytrade_count}")
        print(f"   Status: {account.status}")
        print("   Trading Client OK")
    except Exception as e:
        print(f"   Trading Client FAILED: {e}")
        return False

    # Test data client
    print()
    print("2. Testing Data Client...")
    try:
        data_client = StockHistoricalDataClient(api_key, secret_key)
        request = StockLatestQuoteRequest(symbol_or_symbols="AAPL")
        quote = data_client.get_stock_latest_quote(request)["AAPL"]

        print(f"   AAPL Quote:")
        print(f"   Bid: ${float(quote.bid_price):.2f} x {quote.bid_size}")
        print(f"   Ask: ${float(quote.ask_price):.2f} x {quote.ask_size}")
        print("   Data Client OK")
    except Exception as e:
        print(f"   Data Client FAILED: {e}")
        return False

    # Test positions
    print()
    print("3. Checking Positions...")
    try:
        positions = client.get_all_positions()
        if positions:
            print(f"   Found {len(positions)} positions:")
            for pos in positions[:5]:
                print(f"   - {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
        else:
            print("   No open positions")
    except Exception as e:
        print(f"   Position check failed: {e}")

    print()
    print("=" * 60)
    print("All tests passed! Paper trading is ready.")
    print("=" * 60)

    return True


def run_paper_test():
    """Run a small paper trading test"""
    print("=" * 60)
    print("Paper Trading Test")
    print("=" * 60)
    print()

    try:
        from execution.alpaca_engine import ExecutionEngine, Order
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    # Initialize engine
    engine = ExecutionEngine(paper=True)

    if not engine.client:
        print("Failed to connect to Alpaca")
        return False

    # Get account info
    account = engine.get_account_info()
    print(f"Account Equity: ${account['equity']:,.2f}")
    print(f"Buying Power: ${account['buying_power']:,.2f}")
    print()

    # Test order (very small)
    print("Submitting test order: BUY 1 share of SPY...")

    order = Order(
        symbol="SPY",
        side="buy",
        qty=1,
        order_type="market"
    )

    result = engine.submit_order(order)

    print(f"Order Status: {result.status.value}")
    print(f"Order ID: {result.order_id}")

    if result.status.value == 'submitted':
        print()
        print("Test order submitted successfully!")
        print("Check your Alpaca dashboard to confirm.")
        print()

        # Option to cancel
        cancel = input("Cancel test order? (y/n): ").strip().lower()
        if cancel == 'y':
            try:
                engine.client.cancel_order_by_id(result.order_id)
                print("Order cancelled.")
            except Exception as e:
                print(f"Cancel failed (order may have filled): {e}")

    return True


def show_status():
    """Show current account status"""
    print("=" * 60)
    print("Account Status")
    print("=" * 60)
    print()

    try:
        from execution.alpaca_engine import ExecutionEngine
    except ImportError as e:
        print(f"Import error: {e}")
        return

    engine = ExecutionEngine(paper=True)

    if not engine.client:
        print("Not connected to Alpaca")
        return

    # Account info
    account = engine.get_account_info()
    print("Account Summary:")
    print(f"  Equity:       ${account['equity']:,.2f}")
    print(f"  Cash:         ${account['cash']:,.2f}")
    print(f"  Buying Power: ${account['buying_power']:,.2f}")
    print(f"  Day Trades:   {account.get('day_trade_count', 'N/A')}")
    print()

    # Positions
    positions = engine.get_positions()
    if positions:
        print(f"Open Positions ({len(positions)}):")
        total_value = 0
        total_pnl = 0
        for sym, pos in sorted(positions.items()):
            print(f"  {sym:6} {pos.qty:8.2f} @ ${pos.avg_price:8.2f}  "
                  f"Value: ${pos.market_value:10,.2f}  "
                  f"PnL: {pos.unrealized_pnl_pct:+6.2f}%")
            total_value += pos.market_value
            total_pnl += pos.unrealized_pnl
        print(f"  {'Total':6} {'':8} {'':10}  "
              f"Value: ${total_value:10,.2f}  "
              f"PnL: ${total_pnl:+,.2f}")
    else:
        print("No open positions")


def main():
    parser = argparse.ArgumentParser(description='Alpaca API Setup')
    parser.add_argument('--setup', action='store_true', help='Setup API credentials')
    parser.add_argument('--test', action='store_true', help='Test paper trading')
    parser.add_argument('--status', action='store_true', help='Show account status')
    parser.add_argument('--verify', action='store_true', help='Verify connection only')

    args = parser.parse_args()

    # Load existing env
    load_env()

    if args.setup:
        if setup_credentials():
            test_connection()

    elif args.test:
        if test_connection():
            print()
            proceed = input("Run paper trading test? (y/n): ").strip().lower()
            if proceed == 'y':
                run_paper_test()

    elif args.status:
        show_status()

    elif args.verify:
        test_connection()

    else:
        # Default: show help
        parser.print_help()
        print()
        print("Quick start:")
        print("  1. python config/setup_alpaca.py --setup    # Configure keys")
        print("  2. python config/setup_alpaca.py --test     # Test connection")
        print("  3. python trading_system.py --mode paper    # Start trading")


if __name__ == '__main__':
    main()
