"""Methodology Page - Explains the analysis approach"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import START_DATE, END_DATE

st.set_page_config(page_title="Methodology", page_icon="📖", layout="wide")

"""
# Methodology

This page explains the approach and methodology used in this drawdown analysis project.
"""

st.markdown(f"**Analysis Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

""  # Space

# Section 1: Data Preparation
st.subheader("1. Data Preparation")

st.markdown("""
**ETF Data**
- ARK ETFs holdings from Bloomberg
- iShares Russell 3000 ETF (IWV) from Bloomberg
- Historical fund price data from Yahoo Finance
- GICS industry classifications from Bloomberg
- Company name mappings from Bloomberg
""")

""  # Space

# Section 2: Drawdown Calculation Methodology
st.subheader("2. Drawdown Calculation Methodology")

st.markdown("""
A **drawdown** is defined as the decline from a peak to a trough in an asset's value.

**Calculation Method:**
1. **Identify Peaks**: For each date, calculate the maximum price from the start date up to that date (running maximum)
2. **Calculate Drawdown**: Drawdown % = (Current Price - Peak Price) / Peak Price × 100
3. **Identify Drawdown Periods**:
   - A drawdown begins when price falls below a previous peak
   - A drawdown ends when price reaches a new peak (trough date)
   - Multiple consecutive declines from the same peak are considered one drawdown period
4. **Rank Drawdowns**: Sort all historical drawdowns by depth (magnitude) to identify the top 10

**Current Drawdown**: The ongoing drawdown from the most recent peak to the current date.
""")

""  # Space

# Section 3: Peer Group Analysis Methodology
st.subheader("3. Peer Group Analysis Methodology")

st.markdown("""
For GICS industry peer groups and individual stocks, we calculate two types of aggregate values:

**Market Value (MV) Weighted**
- Sum of (shares × price) across all stocks in the group
- Represents total market capitalization of the peer group
- Better reflects the actual dollar value invested

**Weighted Price**
- Weighted average price: Σ(shares × price) / Σ(shares)
- Represents the average price per share weighted by position size
- Useful for understanding average price movements

Both methods allow us to track peer group performance and calculate drawdowns at the industry level.
""")

""  # Space

# Section 4: Analysis Workflow
st.subheader("4. Analysis Workflow")

st.markdown("""
This section provides a step-by-step guide to reproduce the entire analysis from raw data to final dashboard.

**Step 1: Environment Setup**
- Install required Python packages: `pandas`, `numpy`, `yfinance`, `openpyxl`, `streamlit`, `plotly`
- Set up project directory structure:
  - `input/`: Raw data files from Bloomberg
    - `input/ark_etfs/`: ARK ETF holdings (ARKF, ARKG, ARKK, ARKQ, ARKW, ARKX)
    - `input/russell_3000/`: Russell 3000 holdings (IWV)
    - `input/companyname_mappings/`: Company name standardization files
    - `input/industry_mappings/`: GICS industry classification files
  - `output/`: Generated analysis results
  - `src/`: Python source code modules
  - `pages/`: Streamlit dashboard pages

**Step 2: Data Collection & Preparation**
- Download ETF holdings from Bloomberg Terminal:
  - For ARK ETFs: Use Bloomberg function `PORT` to extract daily holdings
  - For Russell 3000 (IWV): Use Bloomberg function `PRTU` to extract constituent holdings
  - Export to Excel format with columns: Date, Ticker, Shares, Market Value
- Create company name mapping file (`input/companyname_mappings/`):
  - Map Bloomberg tickers to standardized company names
  - Handle ticker changes and corporate actions
  - Sheet name: 'value', columns: Ticker, Company Name
- Create industry mapping file (`input/industry_mappings/`):
  - Map tickers to GICS classifications from Bloomberg (`GICS_SECTOR_NAME`, `GICS_INDUSTRY_GROUP_NAME`)
  - Sheet name: 'value', columns: Ticker, GICS Sector, GICS Industry Group

**Step 3: Data Transformation**
- Run transformation scripts to standardize raw data:
  - Read raw holdings Excel files
  - Use `yfinance` library to download historical price data for each ticker
  - Calculate daily market values: MV = Shares × Price
  - Merge with company name and industry mappings
  - Handle missing data: forward-fill prices, drop tickers with insufficient history
  - Output transformed data to `input/ark_etfs/` and `input/russell_3000/` as `*_Transformed_Data.xlsx`

**Step 4: Peer Group Aggregation**
- Group stocks by GICS Industry Group
- For each peer group, calculate:
  - **Market Value (MV) Weighted**: Daily sum of (Shares × Price) across all stocks
  - **Weighted Price**: Daily Σ(Shares × Price) / Σ(Shares)
- Cache peer group time series data in `data/cache/peer_groups/`

**Step 5: Drawdown Calculation**
- For each asset (ETF, peer group, individual stock):
  - Calculate running maximum price (cumulative peak)
  - Calculate daily drawdown: DD% = (Price - Peak) / Peak × 100
  - Identify drawdown periods: sequences from peak to trough
  - Rank all historical drawdowns by depth (most negative first)
  - Separate current drawdown (ongoing) from completed drawdowns
- Output drawdown tables to `output/` as `*_drawdown_YYYY-YYYY.xlsx`
  - Sheet 'Drawdowns': columns = rank, depth_pct, peak_date, trough_date, peak_price, trough_price
  - Sheet 'Prices': columns = Date, Close, Drawdown%

**Step 6: Individual Stock Analysis**
- For each stock in ARK ETF holdings:
  - Calculate individual stock drawdowns
  - Match stock to GICS peer group
  - Prepare comparison data (stock vs peer group MV and Weighted Price)
- Consolidate all stock drawdowns into `output/Consolidated_Stock_Drawdowns_YYYY-YYYY.xlsx`

**Step 7: Visualization Generation**
- Generate Plotly interactive charts:
  - Price time series with shaded regions for top 10 drawdowns (color-coded)
  - Red dashed line showing current drawdown peak
  - Gray shaded area showing current drawdown magnitude
  - Annotation box with current drawdown details
- Charts saved as HTML in `output/` directories
- PNG exports available via chart download button

**Step 8: Dashboard Deployment**
- Launch Streamlit dashboard: `streamlit run ETF_Analysis.py`
- Dashboard pages:
  1. **ETF Analysis**: ARK ETF overview, individual ETF drawdown analysis, comparison table
  2. **Russell 3000 Analysis**: IWV index analysis, GICS peer group selection and analysis
  3. **Stock Analysis**: Individual stock vs peer group comparison with dual-axis charts
  4. **Raw Data**: Display all input data files (holdings, mappings, classifications)
  5. **Methodology**: This documentation page
- All data loaded via cached functions for performance
- Interactive controls: selectbox, pills, tabs for navigation

**Step 9: Analysis Outputs**
- Summary statistics: Current drawdown, max drawdown, RoMaD for each asset
- Comparison tables: Rank assets by drawdown severity and risk-adjusted returns
- Exportable charts: Download as PNG via Plotly modebar
- Consolidated Excel files: Full drawdown history for all analyzed assets

**Key Files & Modules:**
- `config.py`: Global configuration (date range, ETF list, file paths)
- `data_loader.py`: Functions to load and cache ETF/stock data
- `drawdown_calculator.py`: Core drawdown calculation logic
- `peer_group.py`: Peer group aggregation and caching
- `chart_config.py`: Plotly chart configuration and styling
- `ETF_Analysis.py`: Main dashboard entry point
- `pages/*.py`: Individual dashboard pages (Streamlit multi-page app)

**Reproducibility Notes:**
- All data transformations are deterministic given the same input files
- Price data from Yahoo Finance may vary slightly if re-downloaded due to corporate actions
- Bloomberg data requires active Bloomberg Terminal subscription
- Analysis period is configurable in `config.py` (START_DATE, END_DATE)
- Peer group definitions depend on GICS classifications, which may change over time
""")
