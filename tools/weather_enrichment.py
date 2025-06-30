import logging
import pandas as pd
import dask.dataframe as dd
from datetime import datetime
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry
from tqdm import tqdm
import os
from multiprocessing.pool import ThreadPool

class WeatherEnrichment:
    def __init__(self, data_path: str, output_path: str, api_key: str = None, workers: int = 8):
        self.data_path = data_path
        self.output_path = output_path
        self.api_key = api_key or os.getenv("OPEN_METEO_API_KEY")
        self.logger = logging.getLogger("WeatherEnrichment")
        self.workers = workers

        # Setup Open-Meteo client
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

        self.base_url = "https://customer-archive-api.open-meteo.com/v1/archive"

    def fetch_weather(self, lat, lon, dt):
        date = dt
        next_date = (pd.to_datetime(dt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": next_date,
            "hourly": [
                "temperature_2m", "cloud_cover", "weathercode",
                "wind_speed_10m", "precipitation", "is_day"
            ],
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "apikey": self.api_key
        }
        try:
            response = self.client.weather_api(self.base_url, params=params)[0]
            hourly = response.Hourly()
            times = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )

            df = pd.DataFrame({
                "datetime": times,
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                "cloud_cover": hourly.Variables(1).ValuesAsNumpy(),
                "weathercode": hourly.Variables(2).ValuesAsNumpy(),
                "wind_speed_10m": hourly.Variables(3).ValuesAsNumpy(),
                "precipitation": hourly.Variables(4).ValuesAsNumpy(),
                "is_day": hourly.Variables(5).ValuesAsNumpy()
            })

            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("h")
            df["latitude"] = lat
            df["longitude"] = lon

            return df

        except Exception as e:
            self.logger.error(f"[API ERROR] {lat}, {lon}, {dt} — {e}")
            return pd.DataFrame()

    def run(self, sample_size: int = None):
        self.logger.info(f"Loading data from: {self.data_path}")
        df = dd.read_parquet(self.data_path) if self.data_path.endswith(".parquet") else dd.read_csv(self.data_path)

        df_small = df[['site', 'latitude', 'longitude', 'datatimestamp']].compute()
        df_small['datatimestamp'] = pd.to_datetime(df_small['datatimestamp'], utc=True)

        if sample_size:
            self.logger.info(f"Running weather enrichment on a sample of {sample_size} rows")
            df_small = df_small.head(sample_size)

        df_small['date'] = df_small['datatimestamp'].dt.floor("d").dt.strftime("%Y-%m-%d")
        unique_points = df_small[['site', 'latitude', 'longitude', 'date']].drop_duplicates()
        self.logger.info(f"Requesting weather for {len(unique_points)} unique (site, date) combinations")

        weather_cache = []
        cache_path = Path(self.output_path.replace(".parquet", ".parquet").replace("processed_data", "processed_data_weather"))
        if cache_path.exists():
            weather_cache = pd.read_parquet(cache_path)
            self.logger.info(f"Loaded {len(weather_cache)} rows of cached weather data")
        else:
            def enrich(row):
                self.logger.info(f"Requesting weather for lat: {row['latitude']}, lon: {row['longitude']} and date: {row['date']}")
                return self.fetch_weather(row['latitude'], row['longitude'], row['date'])

            with ThreadPool(self.workers) as pool:
                all_weather = list(tqdm(pool.imap(enrich, [r for _, r in unique_points.iterrows()]),
                                        total=len(unique_points), desc="Fetching weather"))
            weather_cache = pd.concat(all_weather, ignore_index=True)
            weather_cache.to_parquet(cache_path, index=False)
            self.logger.info(f"Saved weather cache to {cache_path}")

        weather_cache["datetime"] = pd.to_datetime(weather_cache["datetime"], utc=True)

        df_full = df.compute()
        df_full['datatimestamp'] = pd.to_datetime(df_full['datatimestamp'], utc=True)
        df_full['hour'] = df_full['datatimestamp'].dt.floor('H')

        merged = pd.merge(
            df_full,
            weather_cache.rename(columns={"datetime": "hour"}),
            on=["latitude", "longitude", "hour"],
            how="left"
        ).drop(columns=["hour"])

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.endswith(".parquet"):
            merged.to_parquet(self.output_path, index=False)
        else:
            merged.to_csv(self.output_path, index=False)

        self.logger.info(f"Saved enriched data to {self.output_path}")
        return {
            "status": "Weather enrichment completed",
            "data": merged
        }

    def merge_only(self):
        self.logger.info(f"Loading base data from: {self.data_path}")
        cache_path = Path(self.output_path.replace(".parquet", ".parquet").replace("processed_data", "processed_data_weather"))
        if not cache_path.exists():
            raise FileNotFoundError("Weather cache not found. Please run full enrichment first.")

        df_full = pd.read_parquet(self.data_path)
        df_full['datatimestamp'] = pd.to_datetime(df_full['datatimestamp'], utc=True)
        df_full['hour'] = df_full['datatimestamp'].dt.floor("h")

        weather_df = pd.read_parquet(cache_path)
        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], utc=True)

        merged = pd.merge(
            df_full,
            weather_df.rename(columns={"datetime": "hour"}),
            on=["latitude", "longitude", "hour"],
            how="left"
        ).drop(columns=["hour"])

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.output_path, index=False)

        self.logger.info(f"Saved merged data to {self.output_path} using cached weather")
        return {
            "status": "Weather merge completed using cached data only",
            "data": merged
        }
