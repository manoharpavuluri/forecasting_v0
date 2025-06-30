"""
Main orchestration script for the Solar Time Series Forecasting system.
Coordinates the execution of all agents in the correct sequence.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl
from dotenv import load_dotenv

from tools.data_ingestion import DataIngestionAgent
from tools.weather_enrichment import WeatherDataAgent
from tools.feature_engineering import FeatureEngineeringAgent
from tools.forecast import ForecastAgent
from tools.anomaly_detector import AnomalyDetectorAgent
from ui import UIAgent
from agents.llm_query import LLMQueryAgent
from tools.preprocessing import PreprocessingAgent

from config import DATA_DIR, MODELS_DIR, Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forecast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForecastPipeline:
    """Main pipeline orchestrating all agents."""
    
    def __init__(
        self,
        input_file: str,
        location: str,
        forecast_horizon: str = "7d"
    ):
        """
        Initialize the forecast pipeline.
        
        Args:
            input_file: Path to input CSV file
            location: Location for weather data
            forecast_horizon: Forecast horizon (e.g., "7d" for 7 days)
        """
        self.input_file = Path(input_file)
        self.location = location
        self.forecast_horizon = forecast_horizon
        self.data = None
        self.weather_data = None
        self.features = None
        self.forecast = None
        self.anomalies = None
        
    def run_data_ingestion(self) -> None:
        """Run the data ingestion agent."""
        logger.info("Starting data ingestion...")
        agent = DataIngestionAgent(self.input_file)
        agent.run()
        self.data = pl.read_parquet(agent.output_path)
        logger.info("Data ingestion completed.")
        
    def run_weather_data(self) -> None:
        """Run the weather data agent."""
        logger.info("Fetching weather data...")
        agent = WeatherDataAgent(self.location)
        self.weather_data = agent.run()
        logger.info("Weather data fetched.")
        
    def run_feature_engineering(self) -> None:
        """Run the feature engineering agent."""
        logger.info("Engineering features...")
        agent = FeatureEngineeringAgent(self.data)
        self.features = agent.run()
        logger.info("Feature engineering completed.")
        
    def run_forecast(self) -> None:
        """Run the forecast agent."""
        logger.info("Generating forecast...")
        agent = ForecastAgent(self.features)
        self.forecast = agent.run()
        logger.info("Forecast generated.")
        
    def run_anomaly_detection(self) -> None:
        """Run the anomaly detection agent."""
        logger.info("Detecting anomalies...")
        agent = AnomalyDetectorAgent(self.forecast, "forecast")
        self.anomalies = agent.run()
        logger.info("Anomaly detection completed.")
        
    def run_ui(self) -> None:
        """Run the UI agent."""
        logger.info("Launching UI...")
        agent = UIAgent({
            'data': self.data,
            'forecast': self.forecast,
            'anomalies': self.anomalies
        })
        agent.run()
        
    def run_llm_query(self, query: str) -> Optional[str]:
        """Run the LLM query agent."""
        logger.info(f"Processing query: {query}")
        agent = LLMQueryAgent({
            'data': self.data,
            'forecast': self.forecast,
            'anomalies': self.anomalies
        })
        return agent.run(query)
        
    def run_pipeline(self) -> None:
        """Run the complete pipeline."""
        try:
            # Create necessary directories
            Path('logs').mkdir(exist_ok=True)
            
            # Run pipeline steps
            self.run_data_ingestion()
            self.run_weather_data()
            self.run_feature_engineering()
            self.run_forecast()
            self.run_anomaly_detection()
            self.run_ui()
            
            logger.info("Pipeline completed successfully.")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def run_pipeline(config: Config, ui_agent: UIAgent = None):
    logger = logging.getLogger("Orchestrator")
    ui = ui_agent or UIAgent()
    ui.notify("Pipeline starting...")

    # Step 1: Ingest Data
    ingestion_agent = DataIngestionAgent(
        meter_data_path=config.meter_path,
        site_info_path=config.site_info_path,
        output_dir="./temp"
    )
    site_info_parquet = ingestion_agent.process_site_info()
    meter_data_parquet = ingestion_agent.process_meter_data()
    joined_parquet = ingestion_agent.join_meter_and_site(meter_data_parquet, site_info_parquet)
    ui.report_progress("Ingestion", 33.0)

    # Step 2: Preprocess Data
    preprocessing_agent = PreprocessingAgent(
        meter_path=joined_parquet,
        site_info_path=site_info_parquet,
        output_path=config.output_path
    )
    processed = preprocessing_agent.run()
    ui.report_progress("Preprocessing", 66.0)

    # Add additional steps as more agents are built (e.g., ForecastingAgent, ReportingAgent)
    ui.notify("Pipeline finished successfully!")
    return f"Success. Final preprocessed data: {config.output_path}"

def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Initialize and run pipeline
    pipeline = ForecastPipeline(
        input_file='data/raw_solar_data.csv',
        location='San Francisco, CA'
    )
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 