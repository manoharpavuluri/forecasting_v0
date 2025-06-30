from dataclasses import dataclass

@dataclass
class Config:
    meter_path: str = "./data/MeterDataWithHeader.parquet"
    site_info_path: str = "./data/site_info.parquet"
    output_path: str = "./output/processed_data.parquet"
    output_dir: str = "./temp"  # For intermediate outputs
