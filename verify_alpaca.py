import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

def verify_alpaca():
    load_dotenv()
    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    
    print(f"API Key: {api_key}")
    
    try:
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        print(f"Successfully connected to Alpaca Paper Account!")
        print(f"Equity: ${account.equity}")
        print(f"Buying Power: ${account.buying_power}")
        print(f"Status: {account.status}")
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")

if __name__ == "__main__":
    verify_alpaca()
