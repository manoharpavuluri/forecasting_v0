import dask.dataframe as dd
import pyodbc
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_object_columns(file_path, nrows=5000):
    """
    Reads a small chunk with pandas and returns columns that contain strings in numeric columns.
    """
    try:
        sample = dd.read_csv(file_path, nrows=nrows)
        object_cols = []
        for col in sample.columns:
            # If any value is a string but the column is mostly numbers, or
            # dtype is object but contains both strings and numbers
            if sample[col].dtype == object:
                if sample[col].apply(lambda x: isinstance(x, str)).any():
                    object_cols.append(col)
        return object_cols
    except Exception as e:
        logger.warning(f"Failed to profile dtypes in {file_path}: {e}")
        return []

def robust_read_csv(file_path: str, use_dask: bool = True) -> dd.DataFrame:
    """
    Robustly read CSV files with progressive fallback strategy.
    """
    if use_dask:
        try:
            # Special case: known problem file
            if 'MeterDataWithHeader.csv' in file_path:
                logger.info("Known dtype issue. Forcing all columns to string for this file.")
                df = dd.read_csv(file_path, assume_missing=True, blocksize="256MB", dtype=str)
            else:
                df = dd.read_csv(file_path, assume_missing=True, blocksize="256MB")
        except Exception as e:
            logger.warning(f"Default read failed, inspecting dtypes: {e}")
            obj_cols = find_object_columns(file_path)
            if obj_cols:
                dtype_map = {col: 'object' for col in obj_cols}
                try:
                    df = dd.read_csv(file_path, assume_missing=True, blocksize="256MB", dtype=dtype_map)
                    logger.info(f"Retrying with columns forced to object: {obj_cols}")
                except Exception as e2:
                    logger.warning(f"Retry failed: {e2}")
                    # Final fallback: all columns as string
                    try:
                        logger.info("Final fallback: reading all columns as strings.")
                        df = dd.read_csv(file_path, assume_missing=True, blocksize="256MB", dtype=str)
                    except Exception as e3:
                        logger.error(f"All-string read failed: {e3}")
                        raise
            else:
                logger.warning("Could not determine mixed-type columns, reading all as string.")
                try:
                    df = dd.read_csv(file_path, assume_missing=True, blocksize="256MB", dtype=str)
                except Exception as e3:
                    logger.error(f"All-string read failed: {e3}")
                    raise
        return df
    else:
        # Use pandas for smaller files
        try:
            df = dd.read_csv(file_path, assume_missing=True, blocksize="256MB")
        except Exception as e:
            logger.warning(f"Pandas read failed, trying with error handling: {e}")
            try:
                df = dd.read_csv(file_path, assume_missing=True, blocksize="256MB", dtype=str)
            except Exception as e2:
                logger.error(f"Pandas read with error handling failed: {e2}")
                raise
        return df

def robust_read_parquet(file_path: str, use_dask: bool = True):
    """
    Robustly read parquet files.
    """
    try:
        if use_dask:
            return dd.read_parquet(file_path)
        else:
            return dd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to read Parquet file {file_path}: {e}")
        raise

# === CSV file path ===
csv_path = '/Users/manohar/Downloads/SampleMeterData.csv'

# === SQL Server connection details ===
server = 'solarforecasting.database.windows.net'
database = 'SolarDB'
username = 'manohar'
password = 'B1zt@lk1977'

# === Build ODBC connection string ===
conn_str = (
    f"DRIVER=/opt/homebrew/lib/libmsodbcsql.17.dylib;"
    f"SERVER={server},1433;"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=60;"
)

# === Load CSV with robust approach ===
logger.info(f"Loading CSV file: {csv_path}")

# Determine if we should use Dask based on file size
file_size = os.path.getsize(csv_path)
use_dask = file_size > 100 * 1024 * 1024  # 100MB threshold

try:
    if csv_path.endswith('.csv'):
        df_ddf = robust_read_csv(csv_path, use_dask=use_dask)
        if use_dask:
            df = df_ddf.compute()
        else:
            df = df_ddf
    else:
        df_ddf = robust_read_parquet(csv_path, use_dask=use_dask)
        if use_dask:
            df = df_ddf.compute()
        else:
            df = df_ddf
    
    logger.info(f"Successfully loaded data with shape: {df.shape}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# === Clean and convert columns with error handling ===
logger.info("Cleaning and converting columns...")

# Convert timestamp with error handling
try:
    df['DataTimeStamp'] = pd.to_datetime(df['DataTimeStamp'], errors='coerce')
    logger.info("Successfully converted DataTimeStamp")
except Exception as e:
    logger.warning(f"Failed to convert DataTimeStamp: {e}")

# Convert float columns with error handling
float_columns = [
    'ApparentPower', 'ReactivePower', 'RealPowerAC',
    'VoltageLN', 'EnergyDelivered'
]
for col in float_columns:
    if col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Successfully converted {col} to numeric")
        except Exception as e:
            logger.warning(f"Failed to convert {col} to numeric: {e}")

# Convert int columns with error handling
int_columns = ['UnitDeviceFaultCodeValue', 'UnitDeviceStateCodeValue']
for col in int_columns:
    if col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            logger.info(f"Successfully converted {col} to integer")
        except Exception as e:
            logger.warning(f"Failed to convert {col} to integer: {e}")

# Fill remaining NaNs in float columns
try:
    df[float_columns] = df[float_columns].fillna(0)
    logger.info("Filled NaN values in float columns")
except Exception as e:
    logger.warning(f"Failed to fill NaN values: {e}")

# === Prepare data for batch insert ===
logger.info("Preparing data for batch insert...")
try:
    data = [tuple(row) for _, row in df.iterrows()]
    total_rows = len(data)
    logger.info(f"Prepared {total_rows} rows for insertion")
except Exception as e:
    logger.error(f"Failed to prepare data for insertion: {e}")
    raise

# === Resume from checkpoint ===
checkpoint_file = 'last_batch_index.txt'
start_index = 0

if os.path.exists(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as f:
            start_index = int(f.read().strip())
            logger.info(f"🔁 Resuming from row {start_index}")
    except Exception as e:
        logger.warning(f"Failed to read checkpoint file: {e}")
        start_index = 0
else:
    logger.info("▶️ Starting fresh insert")

# === Start DB connection ===
try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    logger.info("Successfully connected to database")
except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    raise

# === Insert in batches ===
batch_size = 10000
failed_batches = []

for i in range(start_index, total_rows, batch_size):
    batch = data[i:i + batch_size]
    try:
        cursor.executemany("""
            INSERT INTO dbo.SolarMeterData (
                DataTimeStamp, Site, Substation, DeviceID,
                ApparentPower, ReactivePower, RealPowerAC,
                VoltageLN, UnitDeviceFaultCodeValue,
                UnitDeviceStateCodeValue, EnergyDelivered
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()
        logger.info(f"✅ Inserted rows {i} to {i + len(batch) - 1}")

        # Save checkpoint
        try:
            with open(checkpoint_file, 'w') as f:
                f.write(str(i + batch_size))
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    except Exception as e:
        logger.error(f"❌ Failed to insert batch {i}-{i + len(batch) - 1}: {e}")
        failed_batches.append((i, i + len(batch) - 1, str(e)))

try:
    cursor.close()
    conn.close()
    logger.info("Database connection closed")
except Exception as e:
    logger.warning(f"Failed to close database connection: {e}")

# === Final summary ===
logger.info("✅ Upload complete.")
if failed_batches:
    logger.warning(f"⚠️ {len(failed_batches)} batch(es) failed:")
    for start, end, error in failed_batches:
        logger.warning(f"  - Rows {start} to {end}: {error}")
else:
    logger.info("✅ All batches inserted successfully")
