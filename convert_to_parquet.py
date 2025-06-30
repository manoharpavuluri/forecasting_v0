"""
Script to convert large CSV files to Parquet format.
This helps reduce file size and improve performance.
"""

import pandas as pd
import polars as pl
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import shutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_csv_to_parquet(input_file: str, output_file: str = None, chunk_size: int = 100000):
    """
    Convert a CSV file to Parquet format using chunked processing.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output Parquet file (optional)
        chunk_size: Number of rows to process at once
    """
    input_path = Path(input_file)
    
    # If output file is not specified, use the same name with .parquet extension
    if output_file is None:
        output_path = input_path.with_suffix('.parquet')
    else:
        output_path = Path(output_file)
    
    logger.info(f"Converting {input_path} to Parquet format...")
    logger.info(f"Output will be saved to: {output_path}")
    
    # Get total number of rows for progress bar
    total_rows = sum(1 for _ in open(input_path))
    logger.info(f"Total rows to process: {total_rows:,}")
    
    # Create a temporary directory for chunk files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        chunk_files = []
        
        # Process the file in chunks
        for i, chunk in enumerate(tqdm(pd.read_csv(input_path, chunksize=chunk_size), 
                                     total=total_rows//chunk_size + 1,
                                     desc="Converting chunks")):
            # Convert chunk to PyArrow table
            table = pa.Table.from_pandas(chunk)
            
            # Write chunk to temporary file
            chunk_file = temp_dir_path / f"chunk_{i}.parquet"
            pq.write_table(table, chunk_file)
            chunk_files.append(chunk_file)
        
        # Combine chunks using a streaming approach
        logger.info("Combining chunks...")
        
        # Get schema from first chunk
        schema = pq.read_schema(chunk_files[0])
        
        # Create output file with schema
        with pq.ParquetWriter(output_path, schema) as writer:
            # Process chunks in smaller batches
            batch_size = 100  # Process 100 chunks at a time
            for i in range(0, len(chunk_files), batch_size):
                batch_files = chunk_files[i:i + batch_size]
                for chunk_file in tqdm(batch_files, 
                                     desc=f"Processing batch {i//batch_size + 1}/{(len(chunk_files) + batch_size - 1)//batch_size}"):
                    table = pq.read_table(chunk_file)
                    writer.write_table(table)
                # Force garbage collection after each batch
                import gc
                gc.collect()
    
    # Get file sizes for comparison
    input_size = input_path.stat().st_size / (1024 * 1024)  # Size in MB
    output_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
    
    logger.info(f"Conversion complete!")
    logger.info(f"Original CSV size: {input_size:.2f} MB")
    logger.info(f"Parquet file size: {output_size:.2f} MB")
    logger.info(f"Size reduction: {(1 - output_size/input_size)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to Parquet format')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output', '-o', help='Path to output Parquet file (optional)')
    parser.add_argument('--chunk-size', '-c', type=int, default=100000,
                      help='Number of rows to process at once (default: 100000)')
    
    args = parser.parse_args()
    
    try:
        convert_csv_to_parquet(args.input_file, args.output, args.chunk_size)
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main() 