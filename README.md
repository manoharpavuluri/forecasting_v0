# ☀️ Solar Energy Forecasting System

This project is a Streamlit-based application designed for forecasting solar energy production. It provides a user-friendly interface to ingest, process, and analyze time-series data from solar power sites. The system can enrich the data with weather information and is built to handle data processing tasks efficiently.

## ✨ Features

-   **Interactive UI**: A Streamlit web application for easy interaction.
-   **File Upload & Management**: Supports direct file uploads and management for large datasets.
-   **Data Ingestion**: Ingests meter data and site information from CSV or Parquet files.
-   **Data Processing**: Cleans data, handles missing values, and ensures data quality.
-   **Weather Data Integration**: Enriches energy data with weather information from external APIs.
-   **Pipeline Orchestration**: An automated pipeline for processing and analysis.
-   **Logging**: Comprehensive logging to track the application's execution and for debugging.

## 📋 Prerequisites

-   Python 3.8+
-   `pip` and `venv`

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd forecasting_v0
```

### 2. Set Up the Environment

You can use the provided scripts to set up the virtual environment and install dependencies.

**For macOS/Linux:**

```bash
sh setup_venv.sh
```

**For Windows:**

```batch
setup_venv.bat
```

These scripts will create a `venv` folder, activate the virtual environment, and install the required packages from `requirements.txt`.

Alternatively, you can perform the steps manually:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux
source venv/bin/activate
# On Windows
.\\venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

This project requires API keys for Google Maps (for geocoding) and Open-Meteo (for weather data).

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API keys to the `.env` file as follows:

    ```
    GOOGLE_MAPS_API_KEY="your_google_maps_api_key"
    OPEN_METEO_API_KEY="your_open_meteo_api_key"
    ```

    - You can obtain a **Google Maps API key** from the [Google Cloud Console](https://console.cloud.google.com/google/maps-apis/overview).
    - The **Open-Meteo API** is free and does not strictly require a key for non-commercial use, but you can register for one if needed.

## 🏃‍♀️ How to Run the Application

1.  Make sure your virtual environment is activated.

    ```bash
    source venv/bin/activate
    ```

2.  Run the Streamlit application from the project root directory.

    ```bash
    streamlit run app.py
    ```

The application will open in your default web browser.

## 🖥️ How to Use the Application

1.  **Upload Data**:
    - For smaller files, use the file uploaders in the UI to upload your meter and site information files.
    - For larger files, place them directly into the `data/` directory and refresh the application page.
2.  **Select Files**: Choose your meter and site data files from the dropdown menus.
3.  **Run Pipeline**: Click the "Run Preprocessing Pipeline" button to start the data processing workflow.
4.  **View Outputs**: Processed files will be saved in the `data/processed/` directory and can be viewed in the UI.

## 📂 Project Structure

Here is a brief overview of the key directories and files:

```
forecasting_v0/
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python package dependencies
├── setup_venv.sh           # Setup script for macOS/Linux
├── setup_venv.bat          # Setup script for Windows
├── .env.example            # Example environment file
├── data/                   # Directory for raw and processed data
├── logs/                   # Contains log files for debugging
├── tools/                  # Contains individual data processing scripts (e.g., weather)
└── workflow/
    └── orchestrator.py     # Controls the main data processing pipeline
```

## 📄 Data Format

The application expects input files with the following columns for proper processing:

**Meter Data File:**
- `DataTimeStamp`: The timestamp for the reading.
- `Site`: The identifier for the site.
- `Substation`: The identifier for the substation.
- `DeviceID`: The unique identifier for the device.
- Power/energy columns (e.g., `RealPowerAC`).

**Site Info File:**
- `Site`: The identifier for the site to join with meter data.
- Columns containing location information (e.g., `Address`, `City`, `State`).

---
Happy Forecasting!
