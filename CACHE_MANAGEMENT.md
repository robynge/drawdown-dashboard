# Cache Management Documentation

## Overview

This dashboard uses **THREE SEPARATE CACHING SYSTEMS** that work independently. When you update input data, ALL THREE must be cleared for changes to appear.

---

## The Three Cache Systems

### 1. Application-Level Pickle Cache (`data/cache/`)

**Location:** `data/cache/*.pkl`

**Managed by:** `src/data_loader.py`

**What it caches:**
- `ark_holdings_*.pkl` - ARK ETF holdings data from Excel files
- `r3000_holdings.pkl` - Russell 3000 holdings data
- `etf_prices_*.pkl` - ETF price data loaded from CSV files
- `industry_info_*.pkl` - Industry mapping data
- `company_name_*.pkl` - Company name mapping data

**How it works:**
```python
# In data_loader.py
def load_etf_prices(etf):
    cache_key = f'etf_prices_{etf}'

    # 1. Check in-memory cache first
    if cache_key in _cache:
        return _cache[cache_key]

    # 2. Check disk cache (data/cache/)
    cached = _load_from_cache(cache_key)
    if cached is not None:
        return cached

    # 3. Load from CSV only if no cache exists
    df = pd.read_csv(file_path)
    _save_to_cache(cache_key, df)  # Save to disk
    return df
```

**Why it causes problems:**
- Even if you update `output/*.csv` files, the old pickle cache is loaded instead
- The cache is checked BEFORE reading the CSV file
- Controlled by `CACHE_ENABLED = True` in `config.py`

**How to clear:**
```bash
rm -f data/cache/*.pkl
```

---

### 2. Precomputed Data (`data/precomputed/`)

**Location:** `data/precomputed/etf_data.pkl` and `data/precomputed/*_drawdowns.csv`

**Managed by:** `precompute_data.py`

**What it contains:**
- All ETF price data (loaded via `data_loader.load_etf_prices()`)
- All ETF drawdown calculations
- Pre-calculated for all 6 ARK ETFs

**How it works:**
```python
# In ETF_Analysis.py
@st.cache_data
def load_all_etf_data(cache_key=None):
    # Try to load from precomputed data first
    if PRECOMP_FILE.exists():
        with open(PRECOMP_FILE, 'rb') as f:
            return pickle.load(f)

    # Fallback: Calculate on the fly
    # (but this uses data_loader which has its own cache!)
```

**Why it causes problems:**
- Depends on `data/cache/` being clean when regenerated
- If you regenerate precomputed data while `data/cache/` has old pickles, it bakes in the old data
- This is why you needed multiple attempts to fix the Nov 21 issue

**How to clear:**
```bash
rm -f data/precomputed/*.pkl data/precomputed/*.csv
```

**How to regenerate (AFTER clearing data/cache):**
```bash
PYTHONPATH=. python precompute_data.py
```

---

### 3. Streamlit In-Memory Cache

**Location:** In memory + `.streamlit/` directory (system temp files)

**Managed by:** Streamlit framework via `@st.cache_data` decorator

**What it caches:**
- `ETF_Analysis.py`: `load_all_etf_data()` - All ETF prices and drawdowns
- `Russell_3000_Analysis.py`: `load_iwv_data()` - IWV prices and drawdowns
- `Stock_Analysis.py`: `get_stock_etf_mapping()` - Stock-ETF mappings
- Many other functions with `@st.cache_data`

**How it works:**
```python
@st.cache_data
def load_all_etf_data(cache_key=None):
    # Streamlit caches the return value in memory
    # Even if the underlying files change, Streamlit returns cached value
    # Cache key determines when to invalidate
```

**Why it causes problems:**
- Persists across code changes and file updates
- Even if you delete `data/cache/` and `data/precomputed/`, Streamlit still serves old data
- Cache survives until you restart the app or manually clear it

**How to clear:**

**Option 1: In Browser (easiest)**
- Click hamburger menu (☰) top-right
- Click "Clear cache"
- Click "Rerun"

**Option 2: Restart App**
```bash
# In terminal where streamlit is running
Ctrl + C  # Stop
streamlit run ETF_Analysis.py  # Restart
```

**Option 3: Clear Streamlit cache directory**
```bash
# This varies by system, Streamlit stores cache in temp directories
rm -rf ~/.streamlit/cache/
```

---

## The Root Cause of Multi-Attempt Cache Clearing

### Problem Flow

1. **You update input data** (e.g., new ARK ETF Excel file, new prices)

2. **First attempt: Clear only `data/cache/`**
   - ❌ `data/precomputed/etf_data.pkl` still has old data
   - ❌ Streamlit cache still has old data
   - Result: No change visible

3. **Second attempt: Clear `data/precomputed/` and regenerate**
   - ⚠️ If `data/cache/` wasn't cleared first, precompute loads from old cache
   - ❌ Streamlit cache still has old data
   - Result: Partial update, but Streamlit still shows old data

4. **Third attempt: Restart Streamlit**
   - ✅ Finally loads new data
   - But inefficient and frustrating

### Dependency Chain

```
Input Files (input/, output/)
    ↓ (loaded by data_loader.py)
    ↓ (cached in data/cache/*.pkl)
    ↓
data/cache/*.pkl
    ↓ (used by precompute_data.py)
    ↓
data/precomputed/etf_data.pkl
    ↓ (loaded by ETF_Analysis.py)
    ↓ (cached by @st.cache_data)
    ↓
Streamlit In-Memory Cache
    ↓
Displayed in Browser
```

**Any cache higher in the chain blocks updates from propagating down.**

---

## The Correct Cache Clearing Procedure

### When You Update Input Data

Follow these steps **IN ORDER**:

```bash
# Step 1: Clear application-level pickle cache
rm -f data/cache/*.pkl

# Step 2: Clear precomputed data
rm -f data/precomputed/*.pkl data/precomputed/*.csv

# Step 3: Regenerate precomputed data (loads fresh from CSV/Excel)
PYTHONPATH=. python precompute_data.py

# Step 4a: Clear Streamlit cache in browser
# Click ☰ menu > Clear cache > Rerun

# OR Step 4b: Restart Streamlit app
# Ctrl+C, then: streamlit run ETF_Analysis.py
```

### Quick Clear All Script

Create a helper script `clear_all_caches.sh`:

```bash
#!/bin/bash
echo "Clearing all caches..."

# Clear application caches
rm -f data/cache/*.pkl
echo "✓ Cleared data/cache/*.pkl"

# Clear precomputed data
rm -f data/precomputed/*.pkl data/precomputed/*.csv
echo "✓ Cleared data/precomputed/*"

# Regenerate precomputed data
echo "Regenerating precomputed data..."
PYTHONPATH=. python precompute_data.py

echo ""
echo "✓ All caches cleared and regenerated"
echo ""
echo "NEXT STEP: Restart your Streamlit app or clear cache in browser"
echo "  Browser: Click ☰ menu > Clear cache > Rerun"
echo "  Terminal: Ctrl+C then run: streamlit run ETF_Analysis.py"
```

Make it executable:
```bash
chmod +x clear_all_caches.sh
```

---

## When to Clear Each Cache

### Clear `data/cache/` when:
- ✅ You update input Excel files (`input/ark_etfs/*.xlsx`, `input/russell_3000/*.xlsx`)
- ✅ You update output CSV files (`output/*_prices.csv`)
- ✅ You update industry/company mapping files
- ✅ You change `START_DATE` or `END_DATE` in `config.py`

### Clear `data/precomputed/` when:
- ✅ After clearing `data/cache/`
- ✅ You modify `drawdown_calculator.py` (calculation logic changed)
- ✅ You modify `precompute_data.py` itself

### Clear Streamlit cache when:
- ✅ After regenerating precomputed data
- ✅ You modify any function with `@st.cache_data` decorator
- ✅ You change the structure of cached data (columns, data types, etc.)
- ✅ Anytime the dashboard shows stale data

---

## Configuration: Disabling Application Cache

If you want to disable the application-level pickle cache during development:

```python
# In config.py
CACHE_ENABLED = False  # Change from True to False
```

**Pros:**
- Data always loads fresh from disk
- No need to clear `data/cache/`

**Cons:**
- Slower load times
- Still need to clear `data/precomputed/` and Streamlit cache

---

## Best Practices

### 1. Always Clear in Order
Application cache → Precomputed → Streamlit (top to bottom of dependency chain)

### 2. Use the Helper Script
Don't manually clear each cache - use `clear_all_caches.sh`

### 3. Check Data After Each Step
```bash
# Verify CSV has latest data
tail -3 output/ARKK_prices.csv

# Verify precomputed has latest data
python3 << 'EOF'
import pickle
with open('data/precomputed/etf_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['ARKK']['prices'].tail(3))
EOF
```

### 4. Use Cache Keys in Streamlit Functions
```python
# Good: Cache invalidates when dates change
@st.cache_data
def load_all_etf_data(cache_key=None):
    ...

# Usage
etf_prices, etf_dd = load_all_etf_data(cache_key=f"{START_DATE}_{END_DATE}")
```

### 5. Document Cache Dependencies
When adding new cached functions, document:
- What data it depends on
- When it should be invalidated
- What cache clearing triggers it needs

---

## Troubleshooting

### Issue: Dashboard shows old data after clearing caches

**Diagnosis:**
```bash
# 1. Check if application cache is clean
ls data/cache/*.pkl
# Should show: ls: data/cache/*.pkl: No such file or directory

# 2. Check precomputed data timestamp
ls -lh data/precomputed/etf_data.pkl
# Should show recent timestamp (within last few minutes)

# 3. Check precomputed data content
python3 << 'EOF'
import pickle
with open('data/precomputed/etf_data.pkl', 'rb') as f:
    data = pickle.load(f)
print("Last date:", data['ARKK']['prices']['Date'].iloc[-1])
print("Last price:", data['ARKK']['prices']['Close'].iloc[-1])
EOF
```

**Solution:**
1. If application cache exists → clear it, regenerate precomputed
2. If precomputed has wrong data → clear it, regenerate
3. If precomputed is correct → clear Streamlit cache in browser

### Issue: Precomputed data is missing latest dates

**Cause:** Application cache (`data/cache/*.pkl`) had stale data when you ran `precompute_data.py`

**Solution:**
```bash
# Clear application cache FIRST
rm -f data/cache/*.pkl

# THEN regenerate precomputed
rm -f data/precomputed/*.pkl data/precomputed/*.csv
PYTHONPATH=. python precompute_data.py
```

### Issue: Changes to code don't appear

**Cause:** Streamlit caches function results, including code logic

**Solution:**
- Change function signature or cache_key to force invalidation
- Or clear Streamlit cache in browser
- Or restart Streamlit app

---

## Summary

**Three independent cache systems:**
1. **Application pickle cache** (`data/cache/`) - caches raw data loading
2. **Precomputed data** (`data/precomputed/`) - caches computed results (depends on #1)
3. **Streamlit in-memory** - caches function returns (depends on #2)

**Clear them in order:** 1 → 2 → 3

**Use the script:** `./clear_all_caches.sh`

**Always verify:** Check data at each step before moving to next
