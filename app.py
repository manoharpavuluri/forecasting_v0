import streamlit as st
st.set_page_config(page_title="Agentic AI Data Pipeline", layout="wide")  # ← This must be here, immediately after the import.
st.title("Agentic AI Data Pipeline")

import os
import shutil
from logger_config import setup_logger
setup_logger()
import logging
logging.info("This should show up in both terminal and log file.")

from config import Config
from workflow.orchestrator import run_pipeline
from ui import UIAgent
from utils import clean_outputs

DATA_DIR = "./data"
PROCESSED_DIR = "./data/processed"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)



outputs_to_clean = [
    os.path.join(PROCESSED_DIR, "processed_data.parquet"),
    os.path.join(PROCESSED_DIR, "processed_data_weather.parquet")
    # Add more if you have additional outputs to clean
]

if st.button("Reset Pipeline (Delete Outputs)"):
    deleted = clean_outputs(*outputs_to_clean)
    if deleted:
        st.warning(f"Deleted: {deleted}")
    else:
        st.info("No output files found to delete. Ready for fresh run.")

# st.set_page_config(page_title="Agentic AI Data Pipeline", layout="wide")
# st.title("Agentic AI Data Pipeline")

# --- Upload Section ---

st.header("1. Data Upload and Large File Instructions")

st.markdown(
    """
    **To upload data:**
    - For files **up to 1GB** (CSV or Parquet), use the drag-and-drop upload widgets below.
    - For files **over 1GB** (such as large .parquet files):  
        1. **Do NOT use the file uploader below.**
        2. **Instead, copy or move your file directly into the `./data/` folder** using your operating system (Finder, Explorer, terminal, etc).
        3. Then, refresh this page – the file will be selectable below.
    """
)

uploaded_meter = st.file_uploader("Upload Meter Data (≤1GB, CSV or Parquet)", type=['csv', 'parquet'])
uploaded_site = st.file_uploader("Upload Site Info (≤1GB, CSV or Parquet)", type=['csv', 'parquet'])

meter_path, site_info_path = "", ""
if uploaded_meter is not None:
    meter_path = os.path.join(DATA_DIR, uploaded_meter.name)
    with open(meter_path, "wb") as f:
        f.write(uploaded_meter.getbuffer())
    st.success(f"Uploaded meter data to {meter_path}")

if uploaded_site is not None:
    site_info_path = os.path.join(DATA_DIR, uploaded_site.name)
    with open(site_info_path, "wb") as f:
        f.write(uploaded_site.getbuffer())
    st.success(f"Uploaded site info to {site_info_path}")

# --- Show Current Files ---
st.header("2. Files Available")

data_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
processed_files = [f for f in os.listdir(PROCESSED_DIR) if os.path.isfile(os.path.join(PROCESSED_DIR, f))]

st.write("**/data:**", data_files)
st.write("**/data/processed:**", processed_files)

# --- Pipeline Section ---
st.header("3. Preprocess Data")

if data_files:
    meter_file = st.selectbox("Select meter data file", data_files, key="meter_file")
    site_file = st.selectbox("Select site info file", data_files, key="site_file")
else:
    meter_file = site_file = None

output_file = st.text_input("Output filename (in /data/processed/)", "processed_data.parquet")
output_path = os.path.join(PROCESSED_DIR, output_file)

ui_agent = UIAgent(streamlit_mode=True)

if st.button("Run Preprocessing Pipeline"):
    if not (meter_file and site_file):
        st.error("Please upload/select both meter data and site info files.")
    else:
        config = Config(
            meter_path=os.path.join(DATA_DIR, meter_file),
            site_info_path=os.path.join(DATA_DIR, site_file),
            output_path=output_path,
            output_dir=DATA_DIR  # for any intermediates
        )
        with st.spinner("Running pipeline..."):
            result = run_pipeline(config, ui_agent=ui_agent)
        # ---- Enhanced status display ----
        if isinstance(result, dict) and "skipped" in result.get("status", "").lower():
            st.info("All pipeline outputs are already up to date. No steps were rerun.")
            st.write(result["status"])
        elif isinstance(result, dict) and "weather" in result.get("status", "").lower():
            st.success("Weather data has been fetched and merged with your meter data!")
            st.write(result["status"])
        else:
            st.success("Pipeline executed! Meter data, site info, and weather data processed.")
            st.write(result if isinstance(result, str) else result.get("status", "No status"))

st.header("4. Merge Weather from Cache (no new API calls)")
if st.button("Merge Only (Use Cached Weather)"):
    from tools.weather_enrichment import WeatherEnrichment
    weather = WeatherEnrichment(
        data_path=output_path,
        output_path=output_path.replace(".parquet", "_weather.parquet"),
        api_key=os.getenv("OPEN_METEO_API_KEY")
    )
    with st.spinner("Merging using cached weather..."):
        result = weather.merge_only()
    st.success(result["status"])

st.markdown("### Status and Progress")
