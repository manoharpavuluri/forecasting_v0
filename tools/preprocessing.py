import logging
import dask.dataframe as dd
import time
from tools.weather_enrichment import WeatherEnrichment


class Preprocessing:
    def __init__(self, meter_path: str, site_info_path: str, output_path: str):
        self.meter_path = meter_path
        self.site_info_path = site_info_path
        self.output_path = output_path
        self.logger = logging.getLogger("PreprocessingAgent")

    def handle_nulls_and_standardize(self, df: dd.DataFrame) -> dd.DataFrame:
        n_before = df.isnull().sum().compute()
        self.logger.info(f"Nulls before: {n_before.to_dict()}")
        df = df.fillna({'Site': 'Unknown', 'Meter Value': 0})
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        n_after = df.isnull().sum().compute()
        self.logger.info(f"Nulls after: {n_after.to_dict()}")
        self.logger.info(f"Standardized columns: {list(df.columns)}")
        return df

    def merge_site_info(self, df: dd.DataFrame, site_info: dd.DataFrame) -> dd.DataFrame:
        self.logger.info("Merging site info into meter data...")
        df['site'] = df['site'].astype(str)
        site_info['site'] = site_info['site'].astype(str)
        merged = df.merge(site_info, on='site', how='left')
        unmatched_sites = df[~df['site'].isin(site_info['site'])]['site'].nunique().compute()
        self.logger.info(f"Sites in meter data with no match in site info: {unmatched_sites}")
        added_columns = [col for col in site_info.columns if col != "site"]
        self.logger.info(f"Columns added from site info: {added_columns}")
        self.logger.info(f"Columns after merge: {list(merged.columns)}")
        self.logger.info(f"Sample merged row: {merged.head(1).to_dict('records')}")

        # --- Filter: Keep only rows with both latitude AND longitude ---
        before_count = merged.shape[0].compute()
        filtered = merged.dropna(subset=['latitude', 'longitude'])
        after_count = filtered.shape[0].compute()
        self.logger.info(f"Filtered merged data: {after_count} rows with latitude and longitude (removed {before_count - after_count} rows without).")
        self.logger.info(f"Merged and filtered result: {filtered.shape[0].compute()} rows, columns: {list(filtered.columns)}")
        return filtered




    def standardize_timestamps(self, df: dd.DataFrame) -> dd.DataFrame:
        if 'timestamp' in df.columns:
            n_before = df['timestamp'].isnull().sum().compute()
            df['timestamp'] = dd.to_datetime(df['timestamp'], errors='coerce')
            n_after = df['timestamp'].isnull().sum().compute()
            self.logger.info(f"Standardized timestamps. Nulls before: {n_before}, after: {n_after}")
        return df

    def run(self):
        start = time.time()
        self.logger.info(f"Loading meter data from {self.meter_path}")
        meter_df = dd.read_parquet(self.meter_path) if self.meter_path.endswith('.parquet') else dd.read_csv(self.meter_path)
        self.logger.info(f"Loading site info from {self.site_info_path}")
        site_info_df = dd.read_parquet(self.site_info_path) if self.site_info_path.endswith('.parquet') else dd.read_csv(self.site_info_path)

        meter_df = self.handle_nulls_and_standardize(meter_df)
        site_info_df = self.handle_nulls_and_standardize(site_info_df)

        merged_df = self.merge_site_info(meter_df, site_info_df)
        merged_df = self.standardize_timestamps(merged_df)

        self.logger.info(f"Saving processed data to {self.output_path}")
        if self.output_path.endswith('.parquet'):
            merged_df.to_parquet(self.output_path, write_index=False)
        else:
            merged_df.to_csv(self.output_path, single_file=True, index=False)
        elapsed = time.time() - start
        self.logger.info(f"PreprocessingAgent completed in {elapsed:.2f} seconds.")
        
        enriched_path = self.output_path.replace(".parquet", "_weather.parquet") if self.output_path.endswith(".parquet") else self.output_path.replace(".csv", "_weather.csv")
        self.logger.info("Starting weather enrichment step...")
        WeatherEnrichment(
            data_path=self.output_path,
            output_path=enriched_path
        ).run()
        self.logger.info(f"Weather enrichment completed and saved to {enriched_path}")

        return merged_df
