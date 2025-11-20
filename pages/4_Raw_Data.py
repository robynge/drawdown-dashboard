"""Raw Data Viewer - Display all input data files"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INPUT_DIR

st.set_page_config(
    page_title="Raw Data",
    page_icon="📊",
    layout="wide"
)

"""
# Raw Data

View all input data files used in the analysis.
"""

# Add CSS for left alignment
st.markdown("""
<style>
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

# Section 1: ETF Holdings Data
st.subheader("ETF Holdings Data")

ark_etfs_dir = INPUT_DIR / 'ark_etfs'
ark_files = {
    'ARKF': 'ARKF_Transformed_Data.xlsx',
    'ARKG': 'ARKG_Transformed_Data.xlsx',
    'ARKK': 'ARKK_Transformed_Data.xlsx',
    'ARKQ': 'ARKQ_Transformed_Data.xlsx',
    'ARKW': 'ARKW_Transformed_Data.xlsx',
    'ARKX': 'ARKX_Transformed_Data.xlsx'
}

# Create tabs for each ETF
ark_tabs = st.tabs(list(ark_files.keys()))

for idx, (etf, filename) in enumerate(ark_files.items()):
    with ark_tabs[idx]:
        # Skip temporary Excel files
        if filename.startswith('~$'):
            continue

        file_path = ark_etfs_dir / filename
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)

                # Drop unwanted columns
                cols_to_drop = [col for col in df.columns if 'company' in col.lower() and 'name' in col.lower()]
                cols_to_drop += [col for col in df.columns if 'yfinance' in col.lower() and 'price' in col.lower()]
                df = df.drop(columns=cols_to_drop, errors='ignore')

                st.markdown(f"**File:** `{filename}`")
                st.markdown(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
                st.dataframe(df, use_container_width=True, height=500)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        else:
            st.warning(f"File not found: {filename}")

""  # Space

# Section 2: Company Name Mappings
st.subheader("Company Name Mappings")

companyname_dir = INPUT_DIR / 'companyname_mappings'
companyname_files = {
    'ARK ETFs': 'ARK ETFs company name.xlsx',
    'Russell 3000': 'R3000 company name.xlsx'
}

company_tabs = st.tabs(list(companyname_files.keys()))

for idx, (name, filename) in enumerate(companyname_files.items()):
    with company_tabs[idx]:
        # Skip temporary Excel files
        if filename.startswith('~$'):
            continue

        file_path = companyname_dir / filename
        if file_path.exists():
            try:
                df = pd.read_excel(file_path, sheet_name='value')
                st.markdown(f"**File:** `{filename}`")
                st.markdown(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
                st.dataframe(df, use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        else:
            st.warning(f"File not found: {filename}")

""  # Space

# Section 3: Industry Mappings
st.subheader("Industry Mappings (GICS)")

industry_dir = INPUT_DIR / 'industry_mappings'
industry_files = {
    'ARK ETFs': 'ARK ETFs industry info.xlsx',
    'Russell 3000': 'R3000 industry info.xlsx'
}

industry_tabs = st.tabs(list(industry_files.keys()))

for idx, (name, filename) in enumerate(industry_files.items()):
    with industry_tabs[idx]:
        # Skip temporary Excel files
        if filename.startswith('~$'):
            continue

        file_path = industry_dir / filename
        if file_path.exists():
            try:
                df = pd.read_excel(file_path, sheet_name='value')
                st.markdown(f"**File:** `{filename}`")
                st.markdown(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
                st.dataframe(df, use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        else:
            st.warning(f"File not found: {filename}")

""  # Space

# Section 4: Russell 3000 Data
st.subheader("iShares Russell 3000 ETF (IWV)")

russell_dir = INPUT_DIR / 'russell_3000'
russell_file = 'IWV_Transformed_Data.xlsx'

# Skip temporary Excel files
if not russell_file.startswith('~$'):
    file_path = russell_dir / russell_file
    if file_path.exists():
        try:
            df = pd.read_excel(file_path)

            # Drop unwanted columns
            cols_to_drop = [col for col in df.columns if 'company' in col.lower() and 'name' in col.lower()]
            cols_to_drop += [col for col in df.columns if 'yfinance' in col.lower() and 'price' in col.lower()]
            df = df.drop(columns=cols_to_drop, errors='ignore')

            st.markdown(f"**File:** `{russell_file}`")
            st.markdown(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
            st.dataframe(df, use_container_width=True, height=500)
        except Exception as e:
            st.error(f"Error loading {russell_file}: {e}")
    else:
        st.warning(f"File not found: {russell_file}")
