"""Raw Data Viewer - Display all input data files"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import INPUT_DIR
from session_utils import init_session_state, render_period_selector

st.set_page_config(
    page_title="Raw Data",
    page_icon="📊",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()

"""
# Raw Data

View all input data files used in the analysis.
"""

st.caption("*This page displays raw input data files. Data is not filtered by the selected analysis period.*")

"" # Space

# Add CSS for left alignment
st.markdown("""
<style>
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

# Section 1: Company Name Mappings
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
                st.dataframe(df, width='stretch', height=400)
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
    'Russell 3000': 'IWV_industry group.xlsx'
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
                st.dataframe(df, width='stretch', height=400)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        else:
            st.warning(f"File not found: {filename}")

