# Cache Management Documentation (Updated Architecture)

## Overview

**The dashboard now uses a simple, single-layer caching system.**

Previous multi-layer caching (application cache + precomputed + Streamlit) has been replaced with **Streamlit-only caching with automatic invalidation**.

---

## Current Caching Architecture

### Single Cache Layer: Streamlit `@st.cache_data`

**What it does:**
- All data loading functions use `@st.cache_data` decorator
- Cache automatically invalidates when input files change
- No manual cache clearing needed in normal operation

**How it works:**
```python
# Helper function tracks input file modification times
def get_input_files_hash():
    mtimes = []
    for etf in ARK_ETFS:
        price_file = OUTPUT_DIR / f'{etf}_prices.csv'
        if price_file.exists():
            mtimes.append(price_file.stat().st_mtime)
    return max(mtimes) if mtimes else 0

# Cache function with automatic invalidation
@st.cache_data
def load_all_etf_data(_files_hash):
    # _files_hash (with underscore) is used for cache invalidation
    # When input files change, _files_hash changes, cache invalidates
    ...

# Usage
files_hash = get_input_files_hash()
data = load_all_etf_data(files_hash)  # Auto-invalidates when files change
```

**Key features:**
- ✅ **Automatic invalidation**: Cache invalidates when input files are modified
- ✅ **Transparent**: You don't need to think about caching
- ✅ **Simple**: Single caching layer, no synchronization issues
- ✅ **Just refresh browser**: Changes appear after browser refresh (F5 or Cmd+R)

---

## Optional: Precomputed Data

**Location:** `data/precomputed/etf_data.pkl`

**Purpose:** Speed up first page load by pre-calculating drawdowns

**Status:** Optional - dashboard works fine without it

**How it works:**
1. `precompute_data.py` calculates all ETF drawdowns and saves to `data/precomputed/`
2. Dashboard checks if precomputed file exists and is newer than source files
3. If yes, loads from precomputed (fast)
4. If no, calculates on-the-fly (still cached by Streamlit after first calculation)

**When to regenerate:**
- Run `./clear_all_caches.sh` after updating input data
- Or run `PYTHONPATH=. python precompute_data.py` directly
- Or don't regenerate - dashboard will calculate on-the-fly

---

## What Changed from Old Architecture

### Before (Three-Layer Cache Hell)

```
Input Files
    ↓
data/cache/*.pkl (Manual caching, never invalidates)
    ↓
data/precomputed/*.pkl (Depends on stale data/cache)
    ↓
Streamlit cache (Depends on stale precomputed)
    ↓
Display (Shows old data)
```

**Problems:**
- ❌ Each layer could be stale independently
- ❌ Required manual clearing in specific order
- ❌ Easy to forget a layer
- ❌ Failed silently (showed old data)

### After (Single-Layer with Auto-Invalidation)

```
Input Files
    ↓ (modification time tracked)
Streamlit cache (auto-invalidates when files change)
    ↓
Display (always fresh)

Optional side path:
data/precomputed/*.pkl (checked for staleness before use)
```

**Benefits:**
- ✅ Single source of truth
- ✅ Automatic invalidation
- ✅ No manual intervention needed
- ✅ Fails obviously if something is wrong

---

## When You Update Input Data

### Normal Workflow (No Manual Cache Clearing)

```bash
# 1. Update your input files
# (e.g., replace ARK ETF Excel files, update price CSVs)

# 2. Refresh browser
# Press F5 or Cmd+R
```

**That's it!** The dashboard automatically detects file changes and reloads data.

### Optional: Regenerate Precomputed Data for Faster Loads

```bash
# Run the helper script
./clear_all_caches.sh

# This will:
# - Clear old precomputed data
# - Regenerate fresh precomputed data
# - Verify it worked
```

Then refresh your browser.

### If Dashboard Shows Stale Data

**This should not happen, but if it does:**

1. **Check if you restarted Streamlit after code changes**
   - Code changes require app restart
   - Data changes do NOT require restart

2. **Clear Streamlit cache manually in browser**
   - Click ☰ menu (top-right)
   - Click "Clear cache"
   - Click "Rerun"

3. **Verify your input files actually changed**
   ```bash
   # Check file modification time
   ls -lh output/ARKK_prices.csv

   # Check file contents
   tail -5 output/ARKK_prices.csv
   ```

4. **If still broken, file a bug** - this is a real issue that should be fixed

---

## Removed Components

The following are **no longer used**:

### ❌ Removed: `data/cache/` Directory
- No longer exists
- Application-level pickle caching removed
- All caching handled by Streamlit

### ❌ Removed: `CACHE_ENABLED` Config
- Removed from `config.py`
- No toggle needed - caching always works correctly now

### ❌ Removed: Manual Cache Functions in `data_loader.py`
- `_cache` dictionary removed
- `_load_from_cache()` removed
- `_save_to_cache()` removed
- All functions now load directly from disk
- Streamlit handles caching at higher level

---

## Technical Details

### How Streamlit Cache Invalidation Works

**Using underscore prefix for cache keys:**
```python
@st.cache_data
def load_data(_files_hash):
    # Parameter with _ prefix is not hashed
    # But changing it still invalidates cache
    # This allows cache invalidation based on external state
    ...
```

**Why this works:**
1. When input files change, `_files_hash` changes
2. Streamlit sees different `_files_hash` parameter
3. Streamlit invalidates old cache
4. Function re-executes with fresh data

### Precomputed Data Staleness Check

```python
@st.cache_data
def load_all_etf_data(_files_hash):
    # Check if precomputed exists and is fresh
    if PRECOMP_FILE.exists():
        precomp_mtime = PRECOMP_FILE.stat().st_mtime

        # Get all price file modification times
        price_mtimes = [...]

        # Only use precomputed if newer than all source files
        if precomp_mtime >= max(price_mtimes):
            return load_from_precomputed()

    # Otherwise calculate on-the-fly (fresh data guaranteed)
    return calculate_fresh()
```

This ensures precomputed data is never stale, even if you forget to regenerate it.

---

## Best Practices

### 1. Trust the System

The cache now invalidates automatically. Don't manually clear unless something is actually broken.

### 2. Update Input Files, Refresh Browser

That's the entire workflow. No scripts, no manual clearing.

### 3. Regenerate Precomputed Data (Optional)

Only for faster initial loads. Not required for correctness.

### 4. Restart Streamlit Only for Code Changes

- **Data changes**: Just refresh browser
- **Code changes**: Restart Streamlit app

### 5. If You See Stale Data, Report It

This should never happen. If it does, it's a bug that should be fixed, not worked around.

---

## Troubleshooting

### Dashboard shows old data after updating input files

**Diagnosis:**
```bash
# 1. Verify input files actually changed
ls -lh output/ARKK_prices.csv
tail -3 output/ARKK_prices.csv

# 2. Check browser console for errors
# (Open browser DevTools, check Console tab)

# 3. Try manual cache clear in browser
# ☰ menu > Clear cache > Rerun
```

**Most common cause:**
- You updated the wrong file
- File didn't actually save
- Looking at wrong dashboard page

### "Module not found" or import errors

**Cause:** Code changes require app restart

**Solution:**
```bash
# Stop Streamlit (Ctrl+C)
# Restart
streamlit run ETF_Analysis.py
```

### Precomputed data seems stale

**This should not happen** - the code checks staleness before using precomputed data.

**If it happens anyway:**
```bash
# Regenerate precomputed
./clear_all_caches.sh

# Or delete it - dashboard will calculate on-the-fly
rm data/precomputed/*.pkl
```

---

## Summary

**Old way:**
1. Update input files
2. Clear `data/cache/`
3. Clear `data/precomputed/`
4. Regenerate precomputed
5. Clear Streamlit cache
6. Restart app
7. Hope it worked

**New way:**
1. Update input files
2. Refresh browser
3. Done

**Optional for faster loads:**
```bash
./clear_all_caches.sh  # Regenerate precomputed data
```

The caching system is now invisible and automatic, as it should be.
