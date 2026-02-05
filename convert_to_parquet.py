"""Convert Excel files to Parquet format for faster loading"""
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(__file__).parent / 'input'

def convert_ark_etfs():
    """Convert ARK ETF Excel files to Parquet"""
    ark_dir = INPUT_DIR / 'ark_etfs'

    for xlsx_file in ark_dir.glob('*_Transformed_Data.xlsx'):
        print(f"Converting {xlsx_file.name}...")

        df = pd.read_excel(xlsx_file)
        df['Date'] = pd.to_datetime(df['Date'])

        # Fix CUSIP column type
        if 'CUSIP' in df.columns:
            df['CUSIP'] = df['CUSIP'].astype(str)

        # Save as Parquet
        parquet_file = xlsx_file.with_suffix('.parquet')
        df.to_parquet(parquet_file, index=False)

        # Show size comparison
        xlsx_size = xlsx_file.stat().st_size / (1024 * 1024)
        parquet_size = parquet_file.stat().st_size / (1024 * 1024)
        print(f"  {xlsx_size:.1f} MB -> {parquet_size:.1f} MB ({parquet_size/xlsx_size*100:.0f}%)")

def convert_russell_3000():
    """Convert Russell 3000 Excel file to Parquet"""
    r3000_dir = INPUT_DIR / 'russell_3000'
    xlsx_file = r3000_dir / 'IWV_Transformed_Data.xlsx'

    if not xlsx_file.exists():
        print(f"File not found: {xlsx_file}")
        return

    print(f"Converting {xlsx_file.name}...")

    # Read all sheets and combine
    all_data = []
    xl = pd.ExcelFile(xlsx_file)
    for sheet in xl.sheet_names:
        print(f"  Reading sheet: {sheet}")
        df_sheet = pd.read_excel(xlsx_file, sheet_name=sheet)
        all_data.append(df_sheet)

    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Fix CUSIP column type
    if 'CUSIP' in df.columns:
        df['CUSIP'] = df['CUSIP'].astype(str)

    # Save as Parquet
    parquet_file = xlsx_file.with_suffix('.parquet')
    df.to_parquet(parquet_file, index=False)

    # Show size comparison
    xlsx_size = xlsx_file.stat().st_size / (1024 * 1024)
    parquet_size = parquet_file.stat().st_size / (1024 * 1024)
    print(f"  {xlsx_size:.1f} MB -> {parquet_size:.1f} MB ({parquet_size/xlsx_size*100:.0f}%)")

if __name__ == '__main__':
    print("=" * 50)
    print("Converting Excel files to Parquet")
    print("=" * 50)
    print()

    print("ARK ETFs:")
    convert_ark_etfs()
    print()

    print("Russell 3000:")
    convert_russell_3000()
    print()

    print("=" * 50)
    print("Done! Parquet files created.")
    print("=" * 50)
