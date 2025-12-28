"""
Test API Fetch with User's Keys
"""
import os
from src.data.fetcher_v5_dual_api import DataFetcherV5

# Set user's API keys
os.environ['TIINGO_API_KEY'] = "e250e87869e1ae0eeaefe61fc566a8f23c515ace"
os.environ['POLYGON_API_KEY'] = "iuSlB2A2rdNus58v_YybyTkHzNNa12_A"

print("🔑 Testing Data Fetch with User's API Keys")
print("=" * 60)
print()

fetcher = DataFetcherV5(cache_dir='data_cache_real')

# Test individual symbols
symbols_to_test = ['NVDA', 'AAPL', 'MSFT', 'TSLA']
start_date = '2023-01-01'
end_date = '2024-12-31'

success_count = 0
total_days = 0

for sym in symbols_to_test:
    print(f"Fetching {sym}...")
    df = fetcher.fetch(sym, start_date, end_date)
    
    if not df.empty:
        success_count += 1
        total_days += len(df)
        print(f"✅ {sym}: {len(df)} days")
    else:
        print(f"❌ {sym}: Failed")

print()
print("=" * 60)
print(f"Summary:")
print(f"  Symbols tested: {len(symbols_to_test)}")
print(f"  Successful: {success_count}/{len(symbols_to_test)}")
print(f"  Total days fetched: {total_days}")

print()

if success_count >= 2:
    print("✅ API keys working! Ready for training.")
    print("   → Update train_v5.py to use these symbols.")
elif success_count > 0:
    print("⚠️  Partial API access - Some working.")
    print("   → Use available symbols.")
else:
    print("❌ All APIs failed - Check API keys.")
    print("   → Use synthetic data.")
