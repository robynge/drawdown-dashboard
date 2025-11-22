#!/bin/bash
echo "=========================================="
echo "Regenerating Precomputed Data"
echo "=========================================="
echo ""
echo "Note: Application-level caching has been removed."
echo "All caching is now handled automatically by Streamlit."
echo ""

# Clear precomputed data
echo "Step 1: Clearing precomputed data (data/precomputed/)..."
rm -f data/precomputed/*.pkl data/precomputed/*.csv
if [ $? -eq 0 ]; then
    echo "✓ Cleared data/precomputed/*"
else
    echo "⚠ No precomputed files found"
fi
echo ""

# Regenerate precomputed data
echo "Step 2: Regenerating precomputed data..."
echo "----------------------------------------"
PYTHONPATH=. python precompute_data.py
if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "✓ Precomputed data regenerated successfully"
else
    echo "❌ Error regenerating precomputed data"
    exit 1
fi
echo ""

# Verify regenerated data
echo "Step 3: Verifying regenerated data..."
python3 << 'EOF'
import pickle
from pathlib import Path
try:
    with open('data/precomputed/etf_data.pkl', 'rb') as f:
        data = pickle.load(f)
    arkk_prices = data['ARKK']['prices']
    last_date = arkk_prices['Date'].iloc[-1]
    last_price = arkk_prices['Close'].iloc[-1]
    print(f"✓ ARKK data verified:")
    print(f"  Last date: {last_date.strftime('%Y-%m-%d')}")
    print(f"  Last price: ${last_price:.2f}")
    print(f"  Total rows: {len(arkk_prices)}")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    exit(1)
EOF
echo ""

echo "=========================================="
echo "✓ Precomputed data regenerated!"
echo "=========================================="
echo ""
echo "NEXT STEP:"
echo "Just refresh your browser (F5 or Cmd+R)"
echo ""
echo "The dashboard automatically detects when input files"
echo "have changed and invalidates its cache. No manual"
echo "cache clearing needed!"
echo ""
echo "Note: Precomputed data is optional - the dashboard"
echo "will calculate on-the-fly if precomputed data is"
echo "missing or stale. You only need to regenerate"
echo "precomputed data for faster initial page loads."
echo ""
echo "=========================================="
