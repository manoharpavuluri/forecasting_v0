"""
Anomaly Detector Agent: Detects anomalies in forecast residuals.
"""
import logging
from utils import detect_anomalies

class AnomalyDetectorAgent:
    def __init__(self, data, value_col):
        self.data = data
        self.value_col = value_col

    def run(self):
        logging.info("Detecting anomalies...")
        return detect_anomalies(self.data, self.value_col) 