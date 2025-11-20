#!/bin/bash

echo "🔄 Refreshing dashboard data..."
echo ""

# 1. Clear Python cache
echo "1️⃣ Clearing Python cache..."
rm -rf data/cache/*.pkl
echo "   ✅ Python cache cleared"

# 2. Clear Streamlit cache
echo "2️⃣ Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache/*
echo "   ✅ Streamlit cache cleared"

# 3. Clear Python bytecode
echo "3️⃣ Clearing Python bytecode..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   ✅ Python bytecode cleared"

echo ""
echo "✨ All caches cleared!"
echo ""
echo "📝 Next steps:"
echo "   1. Make sure your updated data files are in the 'input' folder"
echo "   2. Restart your Streamlit app with: streamlit run ETF_Analysis.py"
echo "   3. The app will reload all data from scratch"
echo ""
