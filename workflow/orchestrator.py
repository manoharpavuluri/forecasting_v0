from utils import file_fresh, clean_outputs
from tools.data_ingestion import DataIngestion
from tools.preprocessing import Preprocessing
from tools.weather_enrichment import WeatherEnrichment
import logging
import streamlit as st

def run_pipeline(config, ui_agent=None):
    logger = logging.getLogger("Orchestrator")
    skipped_steps = []
    pipeline_steps = ["Ingestion", "Preprocessing", "Weather Enrichment"]

    meter_parquet = f"{config.output_dir}/meter_data.parquet"
    site_parquet = f"{config.output_dir}/site_info.parquet"
    processed_path = config.output_path
    weather_output_path = processed_path.replace('.parquet', '_weather.parquet')

    if ui_agent:
        ui_agent.notify("Pipeline starting...")

    # --- Ingestion ---
    if file_fresh(meter_parquet, [config.meter_path, config.site_info_path]) and \
       file_fresh(site_parquet, [config.meter_path, config.site_info_path]):
        logger.info("Skipping ingestion: merged files are fresh.")
        skipped_steps.append("Ingestion")
        if ui_agent:
            ui_agent.notify("Ingestion step skipped (outputs are fresh).")
    else:
        deleted = clean_outputs(meter_parquet, site_parquet)
        if deleted:
            logger.info(f"Deleted stale ingestion outputs: {deleted}")
        if ui_agent:
            ui_agent.report_progress("Ingestion", 10)
        ingestion = DataIngestion(
            meter_data_path=config.meter_path,
            site_info_path=config.site_info_path,
            output_dir=config.output_dir
        )
        meter_data, site_data = ingestion.run()

    # --- Preprocessing ---
    if file_fresh(processed_path, [meter_parquet, site_parquet]):
        logger.info("Skipping preprocessing: processed file is fresh.")
        skipped_steps.append("Preprocessing")
        if ui_agent:
            ui_agent.notify("Preprocessing step skipped (output is fresh).")
    else:
        deleted = clean_outputs(processed_path)
        if deleted:
            logger.info(f"Deleted stale preprocessed output: {deleted}")
        if ui_agent:
            ui_agent.report_progress("Preprocessing", 60)
        preprocessing = Preprocessing(
            meter_path=meter_parquet,
            site_info_path=site_parquet,
            output_path=processed_path
        )
        processed = preprocessing.run()

    # --- Weather enrichment ---
    if file_fresh(weather_output_path, [processed_path]):
        logger.info("Skipping weather enrichment: weather file is fresh.")
        skipped_steps.append("Weather Enrichment")
        if ui_agent:
            ui_agent.notify("Weather enrichment step skipped (output is fresh).")
    else:
        deleted = clean_outputs(weather_output_path)
        if deleted:
            logger.info(f"Deleted stale weather output: {deleted}")
        if ui_agent:
            ui_agent.report_progress("Weather Enrichment", 80)
        weather = WeatherEnrichment(
            data_path=processed_path,
            output_path=weather_output_path
        )
        enriched = weather.run()

    # --- Finalizing ---
    if ui_agent:
        ui_agent.report_progress("Finalizing", 100)
        ui_agent.run("Success")
        if len(skipped_steps) == len(pipeline_steps):
            st.info("Pipeline completed! All steps were skipped because outputs are already fresh.")
    logger.info("Pipeline complete. Skipped steps: %s", skipped_steps)
    return {
        "status": "Success",
        "skipped_steps": skipped_steps,
        "output_file": weather_output_path
    }
