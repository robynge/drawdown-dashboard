#!/bin/bash
echo "=========================================="
echo "Clearing All Dashboard Caches"
echo "=========================================="
echo ""

# Clear application caches
echo "Step 1: Clearing application-level cache (data/cache/)..."
rm -f data/cache/*.pkl
if [ $? -eq 0 ]; then
    echo "✓ Cleared data/cache/*.pkl"
else
    echo "⚠ No cache files found in data/cache/"
fi
echo ""

# Clear precomputed data
echo "Step 2: Clearing precomputed data (data/precomputed/)..."
rm -f data/precomputed/*.pkl data/precomputed/*.csv
if [ $? -eq 0 ]; then
    echo "✓ Cleared data/precomputed/*"
else
    echo "⚠ No precomputed files found"
fi
echo ""

# Regenerate precomputed data
echo "Step 3: Regenerating precomputed data..."
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
echo "Step 4: Verifying regenerated data..."
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
echo "✓ All caches cleared and regenerated!"
echo "=========================================="
echo ""
echo "IMPORTANT NEXT STEP:"
echo "You must clear Streamlit's in-memory cache:"
echo ""
echo "Option 1 (In Browser):"
echo "  1. Click the ☰ menu (top-right)"
echo "  2. Click 'Clear cache'"
echo "  3. Click 'Rerun'"
echo ""
echo "Option 2 (Restart App):"
echo "  1. Press Ctrl+C in your terminal"
echo "  2. Run: streamlit run ETF_Analysis.py"
echo ""
echo "=========================================="
