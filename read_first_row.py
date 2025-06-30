import pandas as pd
import dask.dataframe as dd

print("=== Reading First Row of Files ===")

# Read first row of CSV file
print("\nFirst row of MeterDataWithHeader.csv:")
csv_path = 'data/MeterDataWithHeader.csv'
csv_df = pd.read_csv(csv_path, nrows=1)
# Display all columns and their values
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap wide tables
print("\nColumns and their values:")
for col in csv_df.columns:
    print(f"{col}: {csv_df[col].iloc[0]}")

# Read first row of parquet file
print("\n\nFirst row of MeterDataWithHeader.parquet:")
parquet_path = 'data/MeterDataWithHeader.parquet'
parquet_ddf = dd.read_parquet(parquet_path)
parquet_df = parquet_ddf.head(1)
# Display all columns and their values
print("\nColumns and their values:")
for col in parquet_df.columns:
    print(f"{col}: {parquet_df[col].iloc[0]}") 