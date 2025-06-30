import logging
import dask.dataframe as dd
import time

class DataIngestion:
    def __init__(self, meter_data_path: str, site_info_path: str, output_dir: str):
        self.meter_data_path = meter_data_path
        self.site_info_path = site_info_path
        self.output_dir = output_dir
        self.logger = logging.getLogger("DataIngestionAgent")

    def run(self):
        start = time.time()
        self.logger.info(f"Reading meter data from {self.meter_data_path}")
        meter_data = dd.read_parquet(self.meter_data_path) if self.meter_data_path.endswith('.parquet') else dd.read_csv(self.meter_data_path)
        meter_rows = meter_data.shape[0].compute()
        meter_cols = list(meter_data.columns)
        self.logger.info(f"Loaded meter data: {meter_rows} rows, columns: {meter_cols}")

        self.logger.info(f"Reading site info from {self.site_info_path}")
        site_data = dd.read_parquet(self.site_info_path) if self.site_info_path.endswith('.parquet') else dd.read_csv(self.site_info_path)
        site_rows = site_data.shape[0].compute()
        site_cols = list(site_data.columns)
        self.logger.info(f"Loaded site info: {site_rows} rows, columns: {site_cols}")

        out_meter = f"{self.output_dir}/meter_data.parquet"
        out_site = f"{self.output_dir}/site_info.parquet"
        self.logger.info(f"Saving raw meter data to {out_meter}")
        meter_data.to_parquet(out_meter, write_index=False)
        self.logger.info(f"Saving raw site info to {out_site}")
        site_data.to_parquet(out_site, write_index=False)
        elapsed = time.time() - start
        self.logger.info(f"DataIngestionAgent completed in {elapsed:.2f} seconds.")
        return meter_data, site_data
