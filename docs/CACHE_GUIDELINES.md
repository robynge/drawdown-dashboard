# Precomputed and Cache File Handling Guidelines

This document defines the standards for handling all precomputed and cache files in the project, ensuring consistency between local development and Streamlit Cloud deployment.

---

## 1. Core Principles

### 1.1 Cache-First Principle

**All data loading functions must follow this priority order:**

```
Priority 1: Cache file exists → Return cached data immediately
Priority 2: Source file exists → Compute, save to cache, then return
Priority 3: Neither exists → Raise error or return empty data
```

**Prohibited patterns:**
- ❌ Comparing cache file and source file modification timestamps
- ❌ Attempting to read source files when cache exists
- ❌ Performing heavy computation in Streamlit pages

**Reason:** On Streamlit Cloud, only cache files exist - source files are not available. Timestamp comparison will cause cache invalidation failures.

### 1.2 Standard Code Pattern

```python
def load_data_with_cache():
    """Standard cache loading pattern"""
    cache_path = INPUT_DIR / 'cache_file.parquet'
    source_path = INPUT_DIR / 'source_file.parquet'

    # Priority 1: Return cache if exists
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Priority 2: Compute from source (local development only)
    if source_path.exists():
        data = compute_from_source(source_path)
        data.to_parquet(cache_path, index=False)
        return data

    # Priority 3: No data available
    raise FileNotFoundError(
        "No cache or source data found. Run 'python convert_to_parquet.py' locally first."
    )
```

---

## 2. File Structure

### 2.1 Directory Structure

```
input/
├── ark_etfs/
│   ├── {ETF}_Transformed_Data.parquet    # Source data (committed to git)
│   ├── {ETF}_ticker_list.csv             # Cache: ticker list
│   ├── all_stock_drawdowns_cache.parquet # Cache: all stock drawdowns
│   └── precomputed/                      # Precomputed directory
│       ├── {ETF}_etf_drawdowns.parquet
│       ├── {ETF}_hhi_timeseries.parquet
│       └── ark_holdings_max_drawdowns.parquet
│
├── russell_3000/
│   ├── IWV_Transformed_Data.parquet      # Source data (committed to git)
│   ├── ticker_list.csv                   # Cache: ticker list
│   ├── peer_group_mv_cache.parquet       # Cache: market value weighted
│   ├── peer_group_weighted_cache.parquet # Cache: price weighted
│   ├── iwv_total_mv_cache.parquet        # Cache: IWV total market value
│   └── precomputed/                      # Precomputed directory
│       └── r3000_drawdowns_full.parquet
│
├── industry_mappings/                    # Mapping files (committed to git)
│   ├── ARK ETFs industry info.xlsx
│   └── IWV_industry group.xlsx
│
├── companyname_mappings/                 # Mapping files (committed to git)
│   ├── ARK ETFs company name.xlsx
│   └── R3000 company name.xlsx
│
└── precomputed/
    └── metadata.json                     # Precomputation metadata
```

### 2.2 File Naming Conventions

| File Type | Naming Pattern | Example |
|-----------|---------------|---------|
| Source data | `{NAME}_Transformed_Data.parquet` | `ARKK_Transformed_Data.parquet` |
| Ticker list cache | `{NAME}_ticker_list.csv` | `ARKK_ticker_list.csv` |
| General cache | `{purpose}_cache.parquet` | `peer_group_mv_cache.parquet` |
| Precomputed data | `{name}_{type}.parquet` | `ARKK_etf_drawdowns.parquet` |

### 2.3 Git Commit Rules

**Files that MUST be committed to Git:**
- `*_Transformed_Data.parquet` - Source data (~23MB total)
- `*_ticker_list.csv` - Ticker list caches
- `*_cache.parquet` - All cache files
- All files under `precomputed/` directories
- `industry_mappings/` and `companyname_mappings/` directories

**Files that MUST NOT be committed:**
- Temporary files (`*.tmp`, `~$*`)
- IDE configuration (`.vscode/`, `.idea/`)
- Python cache (`__pycache__/`)

---

## 3. Precomputation Script Standards

### 3.1 convert_to_parquet.py Structure

```python
def main():
    """Precomputation script main entry point"""

    # Phase 1: Basic data conversion
    print("Step 1: Converting Excel to Parquet...")
    convert_excel_to_parquet()

    # Phase 2: Generate cache files
    print("Step 2: Generating ticker lists...")
    generate_ticker_lists()

    # Phase 3: Precompute aggregate data
    print("Step 3: Precomputing peer group data...")
    precompute_peer_group_cache()

    # Phase 4: Precompute analysis data
    print("Step 4: Precomputing ETF drawdowns...")
    precompute_etf_drawdowns()

    # Final: Generate metadata
    print("Step N: Generating metadata...")
    generate_metadata()
```

### 3.2 Precomputation Function Template

```python
def precompute_xxx():
    """Precompute XXX data

    Output file: input/xxx/precomputed/xxx.parquet
    Dependencies: input/xxx/source_data.parquet
    """
    output_path = INPUT_DIR / 'xxx' / 'precomputed' / 'xxx.parquet'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load source data
    source_data = pd.read_parquet(INPUT_DIR / 'xxx' / 'source_data.parquet')

    # 2. Perform computation
    result = expensive_computation(source_data)

    # 3. Save result
    result.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")
```

---

## 4. Data Loading Module Standards

### 4.1 precomputed_loader.py Interface

```python
# Loading functions naming: load_{data_type}()
def load_etf_drawdowns(etf: str) -> pd.DataFrame
def load_hhi_timeseries(etf: str) -> pd.DataFrame
def load_correlation_matrix(etf: str) -> pd.DataFrame
def load_ark_holdings_max_drawdowns(etf: str = None) -> pd.DataFrame

# Filtering functions
def filter_by_period(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame

# Validation functions
def check_precomputed_validity() -> tuple[bool, str]
```

### 4.2 Loading Function Template

```python
def load_xxx(param: str = None) -> pd.DataFrame:
    """Load precomputed XXX data

    Args:
        param: Optional filter parameter

    Returns:
        DataFrame, or empty DataFrame if file does not exist
    """
    path = PRECOMPUTED_DIR / 'xxx.parquet'

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Ensure date columns are datetime type
    df = _ensure_datetime(df, ['date_column'])

    # Optional filtering
    if param is not None:
        df = df[df['param_column'] == param].copy()

    return df
```

### 4.3 Date Handling Standards

```python
def _ensure_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Ensure specified columns are datetime type (may be lost after parquet read)"""
    for col in columns:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    return df
```

---

## 5. Streamlit Page Standards

### 5.1 Page Categories

| Type | Data Source | Example Pages |
|------|-------------|---------------|
| **Precomputed pages** | Read precomputed files only | 5_Drawdown_Distribution, 6_Correlations, 8_HHI_Analysis |
| **Dynamic pages** | Read source data + simple computation | 3_Stock_Analysis, 4_Stock_Comparison |
| **Interactive pages** | User-triggered computation | 7_Correlation_Test, 9_Drawdown_Positions |

### 5.2 Precomputed Page Template

```python
"""Precomputed page template"""
import streamlit as st
from precomputed_loader import load_xxx, filter_by_period, check_precomputed_validity

# Check precomputed data validity
is_valid, message = check_precomputed_validity()
if not is_valid:
    st.error(f"Precomputed data invalid: {message}")
    st.info("Please run `python convert_to_parquet.py` locally and redeploy")
    st.stop()

# Load precomputed data
data = load_xxx()
if len(data) == 0:
    st.warning("No data available")
    st.stop()

# Filter by time period
data = filter_by_period(data, start_date, end_date)

# Display data (no computation)
st.dataframe(data)
```

### 5.3 Prohibited Patterns in Pages

```python
# ❌ PROHIBITED: Heavy computation at page load
@st.cache_data
def expensive_computation():
    # This computation should be moved to convert_to_parquet.py
    for ticker in all_tickers:
        calculate_drawdowns(ticker)  # Slow!

# ❌ PROHIBITED: Comparing file timestamps
if cache_file.stat().st_mtime >= source_file.stat().st_mtime:
    # source_file does not exist on Streamlit Cloud!
    pass

# ❌ PROHIBITED: Assuming source files exist
holdings = pd.read_parquet(source_path)  # May fail
```

---

## 6. Cache Invalidation and Updates

### 6.1 When to Re-precompute

- After source data (Excel/Parquet) is updated
- After computation logic changes
- After adding new ETFs or stocks
- After analysis period changes

### 6.2 Update Workflow

```bash
# 1. Run precomputation script locally
python convert_to_parquet.py

# 2. Verify generated files
ls -la input/*/precomputed/
ls -la input/*/*.parquet

# 3. Commit to Git
git add input/
git commit -m "Update precomputed cache files"
git push
```

### 6.3 metadata.json Structure

```json
{
  "version": "1.0",
  "generated_at": "2026-02-15T12:00:00",
  "source_hashes": {
    "ark_files_hash": 1707912000.0,
    "r3000_files_hash": 1707912000.0
  },
  "etfs_processed": ["ARKK", "ARKF", "ARKG", "ARKQ", "ARKW", "ARKX"],
  "analysis_periods": {
    "2024-2026": {"start": "2024-01-01", "end": "2026-12-31"},
    "2021-2023": {"start": "2021-01-01", "end": "2023-12-31"}
  }
}
```

---

## 7. Troubleshooting FAQ

### Q1: Streamlit Cloud shows "FileNotFoundError"

**Cause:** Code attempts to read files not committed to Git

**Solution:**
1. Verify cache files are committed: `git ls-files input/`
2. Check if `.gitignore` excludes necessary files
3. Re-run `python convert_to_parquet.py` and commit

### Q2: Page loads very slowly

**Cause:** Page is still running computation functions

**Solution:**
1. Check if page uses `precomputed_loader.py`
2. Move computation logic to `convert_to_parquet.py`
3. Remove unnecessary `@st.cache_data` decorators (precomputed data doesn't need them)

### Q3: Local and Cloud results differ

**Cause:** Local has updated source data but didn't re-precompute

**Solution:**
1. Run `python convert_to_parquet.py`
2. Commit all updated cache files
3. Push to GitHub

---

## 8. Checklists

### When adding new precomputed data:

- [ ] Add precomputation function in `convert_to_parquet.py`
- [ ] Add loading function in `src/precomputed_loader.py`
- [ ] Update relevant Streamlit pages to use new loading function
- [ ] Run precomputation script and verify output
- [ ] Add new files to Git and push
- [ ] Test on Streamlit Cloud

### When modifying existing computation logic:

- [ ] Update computation function in `convert_to_parquet.py`
- [ ] Re-run precomputation script
- [ ] Verify output files are correct
- [ ] Commit updated cache files
- [ ] Test on Streamlit Cloud

---

*Last updated: 2026-02-15*
