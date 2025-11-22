# Cache Architecture Problems & Solutions

## Critical Design Flaws in Current System

### Flaw 1: Cache Never Checks Source File Modification Time

**Current code in `src/data_loader.py`:**
```python
def _load_from_cache(name):
    if not CACHE_ENABLED:
        return None
    cache_path = _get_cache_path(name)
    if cache_path.exists():  # ❌ Only checks if cache exists
        with open(cache_path, 'rb') as f:
            return pickle.load(f)  # ❌ Returns stale data!
    return None
```

**The problem:**
- If cache file exists, it's ALWAYS used
- Never checks if source file was updated
- You update `input/ark_etfs/ARKK_Transformed_Data.xlsx` → cache still returns old data
- You update `output/ARKK_prices.csv` → cache still returns old data

**This is why you must manually delete `data/cache/*.pkl` every time.**

---

### Flaw 2: Three Independent Cache Layers Create Dependency Hell

```
Input Files (Excel/CSV)
    ↓
data/cache/*.pkl (Layer 1: Naive existence check)
    ↓
data/precomputed/*.pkl (Layer 2: Depends on Layer 1)
    ↓
Streamlit @st.cache_data (Layer 3: Depends on Layer 2)
    ↓
Display
```

**The problem:**
- Each layer can independently become stale
- Updating one layer doesn't automatically invalidate dependent layers
- Must clear in specific order or it breaks
- Highly error-prone for users

---

### Flaw 3: Streamlit Cache Keys Don't Track Input Files

**Before my recent fix:**
```python
@st.cache_data
def load_all_etf_data(cache_key=None):
    ...

# Cache key only includes dates
load_all_etf_data(cache_key=f"{START_DATE}_{END_DATE}")
```

**The problem:**
- If you update input data but dates don't change → Streamlit uses stale cache
- Cache key doesn't track what the function actually depends on

**After my fix (partial):**
```python
# Now includes precomputed file modification time
precomp_mtime = PRECOMP_FILE.stat().st_mtime
load_all_etf_data(cache_key=f"{START_DATE}_{END_DATE}_{precomp_mtime}")
```

**Still not enough:**
- Only fixes Layer 3 (Streamlit cache)
- Layer 1 (data_loader.py) still broken
- Layer 2 (precomputed) still depends on broken Layer 1

---

## Why This Architecture Was Bad From the Start

### Anti-Pattern 1: Manual Cache Management in Streamlit Apps

Streamlit **already provides** `@st.cache_data` decorator. Adding manual pickle caching in `data_loader.py` creates:
- Duplicate caching logic
- Two sources of truth
- Synchronization problems

### Anti-Pattern 2: Precomputed Data as Persistent Cache

`data/precomputed/` was meant to speed up loading, but it:
- Never invalidates automatically
- Requires manual regeneration
- Becomes stale silently
- Should be regenerated on every data update, but you have to remember to do it

### Anti-Pattern 3: No Cache Invalidation Strategy

Standard caching requires:
1. **Dependency tracking** - What files does this cache depend on?
2. **Invalidation check** - Are dependencies newer than cache?
3. **Automatic refresh** - Regenerate when stale

Current system has **NONE** of these.

---

## The Correct Architecture

### Option A: Streamlit-Only Caching (Recommended)

**Remove Layers 1 & 2 entirely. Use only Streamlit's caching with proper keys.**

```python
# In src/data_loader.py - REMOVE all manual caching
def load_ark_holdings(etf):
    """Load ARK ETF holdings - NO CACHING HERE"""
    file_path = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    if 'CUSIP' in df.columns:
        df['CUSIP'] = df['CUSIP'].astype(str)
    return df

def load_etf_prices(etf):
    """Load ETF prices - NO CACHING HERE"""
    file_path = OUTPUT_DIR / f'{etf}_prices.csv'
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df
```

```python
# In ETF_Analysis.py - Cache at Streamlit level with file modification times
@st.cache_data
def load_all_etf_data():
    """Load all ETF data with Streamlit caching"""
    price_data = {}
    dd_data = []

    for etf in ARK_ETFS:
        etf_prices = load_etf_prices(etf)  # No caching in data_loader
        if len(etf_prices) > 0:
            price_data[etf] = etf_prices
            dd_df = calculate_drawdowns(etf_prices)
            if len(dd_df) > 0:
                dd_df.insert(0, 'ETF', etf)
                dd_data.append(dd_df)

    all_dd = pd.concat(dd_data, ignore_index=True) if dd_data else pd.DataFrame()
    return price_data, all_dd

# Get modification times of all input files
def get_input_files_mtime():
    """Get latest modification time of all input files"""
    mtimes = []

    # ETF price files
    for etf in ARK_ETFS:
        price_file = OUTPUT_DIR / f'{etf}_prices.csv'
        if price_file.exists():
            mtimes.append(price_file.stat().st_mtime)

    # Holdings files
    for etf in ARK_ETFS:
        holdings_file = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'
        if holdings_file.exists():
            mtimes.append(holdings_file.stat().st_mtime)

    return max(mtimes) if mtimes else 0

# Use it
input_mtime = get_input_files_mtime()
etf_prices, etf_dd = load_all_etf_data()  # Streamlit auto-keys with input_mtime

# Note: Streamlit 1.33+ automatically tracks function calls and file reads
# Older approach: manually pass cache_key
# etf_prices, etf_dd = load_all_etf_data(cache_key=f"{START_DATE}_{END_DATE}_{input_mtime}")
```

**Benefits:**
- ✅ One caching layer (Streamlit's)
- ✅ Automatic invalidation when input files change
- ✅ No manual cache clearing needed
- ✅ No `data/cache/` directory
- ✅ No `data/precomputed/` directory
- ✅ Simpler, more reliable

**Tradeoffs:**
- First load after Streamlit restart takes longer (no precomputed data)
- But Streamlit cache persists across page refreshes
- Can keep precomputed data for production deployments if needed

---

### Option B: Smart Application-Level Caching (If You Must)

If you want to keep `data_loader.py` caching for some reason:

```python
def _load_from_cache(name, source_file):
    """Load from cache only if cache is newer than source file"""
    if not CACHE_ENABLED:
        return None

    cache_path = _get_cache_path(name)

    # Check cache exists
    if not cache_path.exists():
        return None

    # Check source file exists
    if not source_file.exists():
        return None

    # ✅ CHECK MODIFICATION TIMES
    cache_mtime = cache_path.stat().st_mtime
    source_mtime = source_file.stat().st_mtime

    # If source is newer than cache, cache is stale
    if source_mtime > cache_mtime:
        return None  # Force reload

    # Cache is fresh
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def load_etf_prices(etf):
    """Load ETF price data from CSV with smart caching"""
    cache_key = f'etf_prices_{etf}'
    source_file = OUTPUT_DIR / f'{etf}_prices.csv'

    # Check in-memory cache
    if cache_key in _cache:
        # ✅ Still verify source file hasn't changed
        cache_path = _get_cache_path(cache_key)
        if cache_path.exists() and source_file.exists():
            if source_file.stat().st_mtime > cache_path.stat().st_mtime:
                # Source updated, invalidate in-memory cache
                del _cache[cache_key]
            else:
                return _cache[cache_key]

    # Check disk cache (with modification time check)
    cached = _load_from_cache(cache_key, source_file)
    if cached is not None:
        _cache[cache_key] = cached
        return cached

    # Load from source
    if not source_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(source_file)
    df['Date'] = pd.to_datetime(df['Date'])

    # Save to cache
    _cache[cache_key] = df
    _save_to_cache(cache_key, df)
    return df
```

**Benefits:**
- ✅ Automatic cache invalidation
- ✅ No manual cache clearing
- ✅ Keeps application-level caching

**Tradeoffs:**
- Still have multi-layer caching complexity
- More code to maintain

---

## Recommended Action Plan

### Immediate Fix (Minimal Changes)

1. **Fix `data_loader.py` to check modification times**
   - Implement Option B above
   - All cache functions check source file mtime
   - Auto-invalidate when source is newer

2. **Fix all Streamlit cache keys**
   - Every `@st.cache_data` includes input file mtimes
   - Or use Streamlit 1.33+ auto-tracking

3. **Update `clear_all_caches.sh`**
   - Keep as emergency fallback
   - Shouldn't be needed for normal updates

### Long-Term Fix (Recommended)

1. **Remove `data_loader.py` caching entirely**
   - Set `CACHE_ENABLED = False` in config.py
   - Delete all `_cache`, `_load_from_cache`, `_save_to_cache` code
   - Remove `data/cache/` directory

2. **Remove `data/precomputed/` for development**
   - Keep precomputed data generation for production deployments
   - But don't use it in development (too easy to get stale)

3. **Use only Streamlit caching with proper keys**
   - Simpler architecture
   - Fewer things to go wrong
   - Streamlit is designed for this

4. **Add data validation**
   - On page load, display data ranges
   - "Showing data from 2024-01-02 to [actual last date in data]"
   - Makes stale cache obvious immediately

---

## Why You Keep Having This Problem

**Not your fault. The architecture makes it inevitable.**

Every time you update input data:
1. Layer 1 cache becomes stale (but doesn't know it)
2. Layer 2 depends on stale Layer 1
3. Layer 3 depends on stale Layer 2
4. You have to manually break the chain

**This is a design problem, not a user problem.**

A well-designed cache should:
- ✅ Invalidate automatically when inputs change
- ✅ Be transparent (you shouldn't think about it)
- ✅ Fail obviously if something goes wrong

Current system:
- ❌ Requires manual intervention
- ❌ Fails silently (shows stale data)
- ❌ Has multiple points of failure

---

## Decision Time

I can implement either:

**A) Quick fix (2-3 hours)**
- Add modification time checks to `data_loader.py`
- Fix all Streamlit cache keys
- Keep three-layer architecture
- Less risky, smaller change

**B) Proper fix (4-6 hours)**
- Remove application-level caching
- Remove precomputed data (or make it production-only)
- Use only Streamlit caching
- Simpler, more maintainable
- Slightly riskier (bigger refactor)

What's your preference?
