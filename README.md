# ARK ETF Drawdown Analysis Dashboard

Interactive dashboard for analyzing drawdowns across ARK ETFs and their holdings.

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
