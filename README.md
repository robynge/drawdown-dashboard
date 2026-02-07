# ARK ETF Drawdown Analysis Dashboard

Interactive dashboard for analyzing drawdowns across ARK ETFs and their holdings.

## Quick Start (Important)

**You MUST run the precomputation script before opening Streamlit for the first time, otherwise the initial load will be extremely slow (5-30 minutes):**

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Precompute all caches (REQUIRED! Takes ~2-3 minutes)
python convert_to_parquet.py

# Step 3: Launch Dashboard
streamlit run ETF_Analysis.py
```

### Why is precomputation required?

The dashboard needs to calculate drawdowns for ~3000 R3000 stocks and all ARK ETF holdings. These calculations are very time-consuming (5-30 minutes if done on-the-fly).

Running `convert_to_parquet.py` will:
1. Convert Excel files to Parquet format (faster data loading)
2. Precompute R3000 peer group market value data
3. Precompute drawdowns for all R3000 stocks (~3000 tickers)
4. Precompute drawdowns for all ARK ETF stocks (~2700 records)

**After precomputation, Streamlit will load all pages instantly on first open.**

### After Updating Data

When you update input data (e.g., `IWV_Transformed_Data.xlsx`), you need to re-run precomputation:

```bash
python convert_to_parquet.py
```

Then refresh your browser to see the updated data.

---

## Project Structure

```
drawdown_dashboard/
├── app.py                  # Main dashboard page
├── app/pages/              # Additional dashboard pages
│   ├── 1_ETF_Analysis.py
│   ├── 2_Stock_Analysis.py
│   ├── 3_Peer_Group.py
│   └── 4_Comparison.py
├── src/                    # Core logic modules
│   ├── data_loader.py
│   ├── drawdown_calculator.py
│   ├── peer_group_analyzer.py
│   └── chart_builder.py
├── input/                  # Input data files
│   ├── ark_etfs/
│   ├── russell_3000/
│   └── industry_mappings/
├── data/                   # Cached and processed data
│   ├── cache/
│   └── processed/
├── config.py               # Configuration settings
└── requirements.txt        # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your browser at http://localhost:8501

## Features

### Main Dashboard
- Overview of all ETF and stock drawdowns
- Filterable summary tables
- Download data as CSV

### ETF Analysis
- Detailed ETF drawdown analysis
- Interactive price charts with drawdown regions
- Top holdings during max drawdown periods

### Stock Analysis
- Individual stock drawdown analysis
- Peer group comparison (MV-weighted or equal-weighted)
- GICS industry-based peer analysis

### Peer Group
- GICS industry group analysis
- Stock composition by industry
- Industry-level market value trends

### Comparison
- Multi-stock comparison (2-10 stocks)
- Normalized performance charts
- Side-by-side drawdown metrics

## Configuration

Edit `config.py` to adjust:
- Analysis date range (START_DATE, END_DATE)
- ETF list
- Cache settings
- File paths

## Data Sources

The dashboard uses data from:
- ARK ETF transformed holdings data
- Russell 3000 (IWV) transformed holdings data
- GICS industry mapping files

## Performance

- Data caching enabled by default for faster load times
- In-memory and disk-based cache layers
- Streamlit's built-in caching for UI components
