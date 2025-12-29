import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

def check_orders():
    load_dotenv()
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    
    client = TradingClient(api_key, secret_key, paper=True)
    
    # 获取最近的订单
    orders = client.get_orders(filter=GetOrdersRequest(status='all', limit=10))
    
    print(f"\n{'Symbol':<10} | {'Qty':<10} | {'Status':<15} | {'Type':<10}")
    print("-" * 50)
    for o in orders:
        print(f"{o.symbol:<10} | {o.qty:<10} | {o.status:<15} | {o.order_type:<10}")

if __name__ == "__main__":
    check_orders()
