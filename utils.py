import dask.dataframe as dd
import os

def get_row_count(file_path: str) -> int:
    if file_path.endswith('.parquet'):
        df = dd.read_parquet(file_path)
    else:
        df = dd.read_csv(file_path)
    return df.shape[0].compute()

def unique_values(file_path: str, col_name: str) -> int:
    if file_path.endswith('.parquet'):
        df = dd.read_parquet(file_path)
    else:
        df = dd.read_csv(file_path)
    return df[col_name].nunique().compute()

def file_fresh(output_path, input_paths):
    """Return True if output_path exists and is newer than all input_paths."""
    if not os.path.exists(output_path):
        return False
    output_mtime = os.path.getmtime(output_path)
    for path in input_paths:
        if not os.path.exists(path):
            return False
        if os.path.getmtime(path) > output_mtime:
            return False
    return True

def clean_outputs(*paths):
    """Delete files if they exist."""
    deleted = []
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
            deleted.append(p)
    return deleted
